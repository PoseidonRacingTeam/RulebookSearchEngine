"""Build and load the persistent rule index.

Index layout (data/index/):
  rules.json             all parsed rules/sections/chapters + assets + glossary
  extraction_report.json coverage / validation report from the extractor
  assets/*.png           rendered figure / table / formula regions
  docs.json              search documents (one per operative rule) with tokens
  embeddings.npy         float32 [n_docs, dim] normalised embeddings
  meta.json              fingerprints: pdf sha256, parser/index versions, models
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .extract import PARSER_VERSION, extract_pdf, sha256_file
from .textnorm import expand_glossary_in_text, normalize_for_search, raw_tokens, tokenize


class IndexError_(Exception):
    pass


def build_search_docs(data: dict) -> list[dict]:
    """One search document per operative rule. The doc text used for BM25 and
    embeddings combines the heading path, the rule's own text, table/figure
    text and glossary expansions. Display text is kept separately."""
    glossary = data["glossary"]
    assets = data["assets"]
    docs = []
    for r in data["rules"]:
        if r["kind"] != "rule":
            continue
        parts = []
        for b in r["blocks"]:
            if b["type"] in ("p", "li", "note"):
                parts.append(b["text"])
            elif b["type"] in ("table", "figure", "formula"):
                a = assets.get(b["asset"])
                if a:
                    if a["caption"]:
                        parts.append(a["caption"])
                    if a["kind"] == "table":
                        parts.append(" ; ".join(" | ".join(c for c in row if c) for row in a["rows"]))
                    elif a["kind"] == "formula":
                        parts.append(a["text"])
        body = " ".join(parts)
        heading = f"{r['chapter_title']} — {r['section_title']}".strip(" —")
        expansions = expand_glossary_in_text(body + " " + heading, glossary)
        # abbreviations are spelled out for the embedding model and the reranker
        # as well (search text only; display text is untouched)
        abbr = [k for k in glossary if k in {e for e in glossary if glossary[e] in expansions} and glossary[k] in expansions]
        abbr_note = (" Abbreviations: " + "; ".join(f"{k} = {glossary[k]}" for k in abbr)) if abbr else ""
        embed_text = f"{r['code']} {heading}: {body}{abbr_note}"
        bm25_text = f"{heading} {heading} {body} {' '.join(expansions)}"
        docs.append({
            "code": r["code"],
            "key": r["key"],
            "heading": heading,
            "embed_text": embed_text,
            "bm25_text": bm25_text,
            "tokens": tokenize(bm25_text),
            "raw_tokens": raw_tokens(bm25_text),
            "scope": r["applicability"]["scope"],
            "provision_tags": r["applicability"]["provision_tags"],
            "tags": r["applicability"]["tags"],
            "pages": r["pages"],
            "section": r["section"],
            "chapter": r["chapter"],
            "rerank_text": f"{heading}. {body}{abbr_note}",
        })
    return docs


def build_index(pdf_path: Path, index_dir: Path, log=print) -> dict:
    config.offline_env()
    t0 = time.time()
    index_dir.mkdir(parents=True, exist_ok=True)
    log(f"[1/4] Extracting rules from {pdf_path.name} ...")
    data, report = extract_pdf(pdf_path, index_dir)
    (index_dir / "rules.json").write_text(json.dumps(data, ensure_ascii=False))
    (index_dir / "extraction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    c = report["counts"]
    log(f"      {c['rules']} rules, {c['sections']} sections, {c['chapters']} chapters, {c['figures']} figures, {c['tables']} tables, {c['formulas']} formulas")
    problems = len(report["issues"]) + len(report["unassigned_text"]) + len(report["numbering_gaps"]) + len(report["duplicates"]) + len(report["toc_mismatches"])
    log(f"      validation: {problems} problem(s) reported (see extraction_report.json)")

    log("[2/4] Building search documents ...")
    docs = build_search_docs(data)
    (index_dir / "docs.json").write_text(json.dumps(docs, ensure_ascii=False))

    log(f"[3/4] Computing embeddings with {config.EMBEDDING_MODEL} ({len(docs)} docs) ...")
    from sentence_transformers import SentenceTransformer
    device = pick_device()
    model = SentenceTransformer(config.EMBEDDING_MODEL, cache_folder=str(config.MODELS_DIR), device=device)
    emb = model.encode([d["embed_text"] for d in docs], batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(index_dir / "embeddings.npy", emb)

    log("[4/4] Writing index metadata ...")
    meta = {
        "index_format_version": config.INDEX_FORMAT_VERSION,
        "parser_version": PARSER_VERSION,
        "pdf_file": pdf_path.name,
        "pdf_sha256": sha256_file(pdf_path),
        "pdf_size": pdf_path.stat().st_size,
        "pdf_pages": data["document"]["pages"],
        "document_title": data["document"]["title"],
        "document_version": data["document"]["version"],
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": int(emb.shape[1]),
        "reranker_model": config.RERANKER_MODEL,
        "n_docs": len(docs),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": round(time.time() - t0, 1),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"Done in {meta['build_seconds']} s.")
    return meta


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_index(index_dir: Path, pdf_path: Path) -> dict:
    """Raise IndexError_ with a clear message if the index is missing, stale or
    incompatible. Returns meta on success."""
    meta_p = index_dir / "meta.json"
    if not meta_p.exists():
        raise IndexError_("No index found. Run `./setup.sh` (or `uv run python -m app.build`) to build it.")
    meta = json.loads(meta_p.read_text())
    for f in ("rules.json", "docs.json", "embeddings.npy"):
        if not (index_dir / f).exists():
            raise IndexError_(f"Index is incomplete ({f} missing). Rebuild it with `uv run python -m app.build`.")
    if meta.get("index_format_version") != config.INDEX_FORMAT_VERSION or meta.get("parser_version") != PARSER_VERSION:
        raise IndexError_("Index was built by an older version of the app. Rebuild it with `uv run python -m app.build`.")
    if not pdf_path.exists():
        raise IndexError_(f"The rulebook PDF is missing: {pdf_path}")
    if meta.get("pdf_sha256") != sha256_file(pdf_path):
        raise IndexError_("The PDF has changed since the index was built (fingerprint mismatch). Rebuild it with `uv run python -m app.build`.")
    if meta.get("embedding_model") != config.EMBEDDING_MODEL:
        raise IndexError_(f"Index embeddings were built with {meta.get('embedding_model')} but {config.EMBEDDING_MODEL} is configured. Rebuild it with `uv run python -m app.build`.")
    return meta


def load_index(index_dir: Path) -> tuple[dict, list[dict], np.ndarray]:
    data = json.loads((index_dir / "rules.json").read_text())
    docs = json.loads((index_dir / "docs.json").read_text())
    emb = np.load(index_dir / "embeddings.npy")
    if emb.shape[0] != len(docs):
        raise IndexError_("Index is corrupt (embedding count does not match documents). Rebuild it.")
    return data, docs, emb
