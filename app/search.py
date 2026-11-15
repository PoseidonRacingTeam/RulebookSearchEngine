"""Hybrid rule search: deterministic code lookup, BM25 + dense retrieval,
cross-encoder reranking and a conservative decision layer that says
"No reliable match found" instead of inventing an answer.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein

from . import config
from .indexer import load_index, validate_index
from .textnorm import STOPWORDS, expand_glossary_in_query, normalize_for_search, raw_tokens, stem, tokenize

SYNONYMS_PATH = config.DATA_DIR / "synonyms.json"
SYSTEM_WORDLIST = Path("/usr/share/dict/words")

log = logging.getLogger("fs.search")

CODE_QUERY_RE = re.compile(r"^\s*(a|t|cv|ev|in|s|d)\s*[-_ ]?\s*(\d{1,2}(?:\s*\.\s*\d{1,2}){0,2})\s*\.?\s*$", re.I)
CODE_IN_TEXT_RE = re.compile(r"\b(A|T|CV|EV|IN|S|D)\s?(\d{1,2}(?:\.\d{1,2}){0,2})\b")

# Decision thresholds on fused scores (reranker logit + weighted dense cosine).
# Calibrated on the dev split of eval/queries.json (see eval/results). They are
# *not* probabilities and are never shown as confidence percentages.
T_NO_MATCH = -2.5          # top reranker logit below this => "No reliable match found"
T_INCLUDE = -2.5           # extra results must have a reranker logit above this ...
DELTA_INCLUDE = 4.0        # ... and a fused score within this margin of the top
WEAK_RR = 0.0              # top logit below this => result flagged as a weak match
WEAK_COS_MIN = 0.55        # weak match AND low cosine AND few matched query terms => no match
DELTA_AMBIGUOUS = 0.5      # top-2 fused scores within this margin => possibly ambiguous
FUSION_DENSE_WEIGHT = 10.0 # fused = rerank_logit + W * (cosine - 0.5) + W_B * bm25/bm25_max
FUSION_BM25_WEIGHT = 1.0
RERANK_WITH_EXPANSIONS = True  # reranker sees the same expanded query as the dense model
RRF_K = 60


@dataclass
class Timing:
    t0: float = field(default_factory=time.perf_counter)
    stages: dict = field(default_factory=dict)

    def mark(self, name: str, start: float) -> None:
        self.stages[name] = round((time.perf_counter() - start) * 1000, 1)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)


class SearchEngine:
    def __init__(self, index_dir: Path, pdf_path: Path, models_dir: Path):
        self.index_dir = index_dir
        self.pdf_path = pdf_path
        self.models_dir = models_dir
        self.ready = False
        self.state = "init"
        self.progress: list[dict] = []
        self.error: str | None = None
        self.meta: dict = {}
        self.data: dict = {}
        self.docs: list[dict] = []
        self.emb: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.vocab_df: Counter = Counter()
        self.vocab_list: list[str] = []
        self.rules_by_key: dict[str, dict] = {}
        self.rules_in_order: list[dict] = []
        self.doc_index_by_key: dict[str, int] = {}
        self.embedder = None
        self.reranker = None
        self.device = "cpu"
        self.lock = threading.Lock()
        self.pending = 0
        self.pending_lock = threading.Lock()
        self.semantic_available = False
        self.rerank_available = False
        self.debug = False
        self.last_debug: dict = {}

    # ------------------------------------------------------------------ loading
    def _step(self, name: str, status: str = "running", detail: str = "") -> None:
        for s in self.progress:
            if s["name"] == name:
                s["status"], s["detail"] = status, detail
                return
        self.progress.append({"name": name, "status": status, "detail": detail})

    def load(self) -> None:
        """Load index + models. Sets self.ready only after a warm-up search works."""
        config.offline_env()
        self.state = "loading"
        try:
            self._step("Validate index")
            self.meta = validate_index(self.index_dir, self.pdf_path)
            self._step("Validate index", "done", f"{self.meta['document_title']} {self.meta['document_version']}")

            self._step("Load rules and search documents")
            self.data, self.docs, self.emb = load_index(self.index_dir)
            self.rules_in_order = self.data["rules"]
            self.rules_by_key = {r["key"]: r for r in self.rules_in_order}
            self.doc_index_by_key = {d["key"]: i for i, d in enumerate(self.docs)}
            self.bm25 = BM25Okapi([d["tokens"] for d in self.docs])
            for d in self.docs:
                self.vocab_df.update(set(d["raw_tokens"]))
            for k in self.data["glossary"]:
                self.vocab_df[k.lower()] += 1
            self.vocab_list = list(self.vocab_df.keys())
            self.synonyms = self._load_synonyms()
            self.known_words = set(self.vocab_df)
            if SYSTEM_WORDLIST.exists():
                try:
                    words = SYSTEM_WORDLIST.read_text(errors="ignore").split()
                    self.known_words |= {w.lower() for w in words}
                    self.known_stems = {stem(w.lower()) for w in words}
                except OSError:
                    self.known_stems = set()
            else:
                self.known_stems = set()
            self._step("Load rules and search documents", "done", f"{len(self.docs)} rules, {len(self.vocab_list)} vocabulary terms, {len(self.synonyms)} synonym groups")

            import torch
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            from sentence_transformers import CrossEncoder, SentenceTransformer

            self._step(f"Load embedding model ({config.EMBEDDING_MODEL})")
            self.embedder = SentenceTransformer(config.EMBEDDING_MODEL, cache_folder=str(self.models_dir), device=self.device, local_files_only=True)
            self.semantic_available = True
            self._step(f"Load embedding model ({config.EMBEDDING_MODEL})", "done", f"device {self.device}")

            self._step(f"Load reranker ({config.RERANKER_MODEL})")
            self.reranker = CrossEncoder(config.RERANKER_MODEL, cache_folder=str(self.models_dir), device=self.device, local_files_only=True, max_length=512)
            self.rerank_available = True
            self._step(f"Load reranker ({config.RERANKER_MODEL})", "done", f"device {self.device}")

            self._step("Warm-up search")
            t = time.perf_counter()
            res = self.search("driver egress time", "all")
            if res["type"] not in ("results", "no_match", "clarification"):
                raise RuntimeError(f"warm-up search failed: {res.get('message')}")
            self._step("Warm-up search", "done", f"{round((time.perf_counter() - t) * 1000)} ms")
            self.ready = True
            self.state = "ready"
        except Exception as e:  # noqa: BLE001
            log.exception("engine load failed")
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"
            for s in self.progress:
                if s["status"] == "running":
                    s["status"] = "failed"
                    s["detail"] = self.error

    def _load_synonyms(self) -> list[list[str]]:
        import json
        if not SYNONYMS_PATH.exists():
            return []
        data = json.loads(SYNONYMS_PATH.read_text())
        aliases = data.get("aliases", {})
        out = []
        for alias, targets in aliases.items():
            stems = " ".join(stem(w) for w in raw_tokens(alias))
            out.append((stems, [t for t in targets]))
        return out

    def expand_synonyms(self, query_norm: str) -> list[str]:
        """Rulebook wording for colloquial phrases found in the query (directed:
        alias -> canonical only). Matching is on stemmed words so 'knocking
        over a cone' matches 'knocked over cone'. Canonical terms that are
        already in the query are not added."""
        q_stems = " " + " ".join(stem(w) for w in raw_tokens(query_norm)) + " "
        out: list[str] = []
        for alias_stems, targets in self.synonyms:
            if f" {alias_stems} " in q_stems:
                for t in targets:
                    t_stems = " ".join(stem(w) for w in raw_tokens(t))
                    if f" {t_stems} " not in q_stems and t not in out:
                        out.append(t)
        return out[:6]

    # ------------------------------------------------------------------ helpers
    def status(self) -> dict:
        return {
            "state": self.state,
            "ready": self.ready,
            "error": self.error,
            "steps": self.progress,
            "document": {k: self.meta.get(k) for k in ("document_title", "document_version", "pdf_file", "pdf_pages", "pdf_sha256", "built_at")},
            "models": {"embedding": config.EMBEDDING_MODEL, "reranker": config.RERANKER_MODEL, "device": self.device,
                       "semantic_available": self.semantic_available, "rerank_available": self.rerank_available},
            "counts": {"rules": len(self.docs)},
        }

    def normalise_code(self, q: str) -> str | None:
        m = CODE_QUERY_RE.match(q)
        if not m:
            return None
        num = re.sub(r"\s+", "", m.group(2))
        return f"{m.group(1).upper()} {num}"

    def rule_payload(self, r: dict, with_context: bool = True) -> dict:
        assets = self.data["assets"]
        blocks = []
        for b in r["blocks"]:
            bb = dict(b)
            if b["type"] in ("table", "figure", "formula"):
                a = assets.get(b["asset"])
                if a:
                    bb["asset_data"] = {k: a[k] for k in ("id", "kind", "number", "caption", "page", "image", "rows", "extra_images", "pages")}
            blocks.append(bb)
        out = {
            "code": r["code"], "key": r["key"], "kind": r["kind"], "title": r.get("title", ""),
            "part": r["part"], "part_title": r["part_title"], "chapter": r["chapter"], "chapter_title": r["chapter_title"],
            "section": r["section"], "section_title": r["section_title"],
            "pages": r["pages"], "bboxes": r["bboxes"], "blocks": blocks, "display_text": r["display_text"],
            "applicability": r["applicability"], "references": [], "asset_refs": [], "changelog": r["changelog"],
        }
        for c in r["references"]:
            rr = self.rules_by_key.get(re.sub(r"\s+", "", c).upper())
            if rr:
                out["references"].append({"code": rr["code"], "kind": rr["kind"], "title": rr.get("title", ""), "section_title": rr["section_title"],
                                          "pages": rr["pages"], "display_text": rr["display_text"] if rr["kind"] == "rule" else ""})
        for name in r["asset_refs"]:
            aid = self.data["numbered_assets"].get(name)
            if aid and aid in assets:
                a = assets[aid]
                out["asset_refs"].append({k: a[k] for k in ("id", "kind", "number", "caption", "page", "image", "rows", "extra_images", "pages")})
        if with_context:
            out["context"] = self.context_for(r)
        return out

    def context_for(self, r: dict) -> dict:
        """Neighbouring rules in the same section plus prev/next in document order."""
        order = r.get("order", 0)
        siblings = [x for x in self.rules_in_order if x["kind"] == "rule" and x["section"] == r["section"]] if r["kind"] == "rule" else []
        prev_r = next_r = None
        rules_only = [x for x in self.rules_in_order if x["kind"] == "rule"]
        for i, x in enumerate(rules_only):
            if x["key"] == r["key"]:
                prev_r = rules_only[i - 1] if i > 0 else None
                next_r = rules_only[i + 1] if i + 1 < len(rules_only) else None
                break
        return {
            "section_rules": [{"code": s["code"], "snippet": s["display_text"][:140]} for s in siblings],
            "prev": prev_r["code"] if prev_r else None,
            "next": next_r["code"] if next_r else None,
        }

    def get_rule(self, code: str) -> dict | None:
        r = self.rules_by_key.get(re.sub(r"\s+", "", code).upper())
        return self.rule_payload(r) if r else None

    def section_rules(self, code: str) -> list[dict]:
        key = re.sub(r"\s+", "", code).upper()
        sec = self.rules_by_key.get(key)
        if not sec:
            return []
        if sec["kind"] == "rule":
            key = sec["section"].replace(" ", "")
        if sec["kind"] == "chapter":
            return [self.rule_payload(x, with_context=False) for x in self.rules_in_order if x["kind"] == "rule" and x["chapter"].replace(" ", "") == key]
        return [self.rule_payload(x, with_context=False) for x in self.rules_in_order if x["kind"] == "rule" and x["section"].replace(" ", "") == key]

    # ------------------------------------------------------------------ code lookup
    def lookup_code(self, code: str, timing: Timing) -> dict:
        key = code.replace(" ", "").upper()
        r = self.rules_by_key.get(key)
        if r is None:
            # suggest existing codes in the same section / chapter, clearly labelled
            parent = code.rsplit(".", 1)[0] if "." in code else code.split()[0]
            parent_key = parent.replace(" ", "").upper()
            siblings = [x["code"] for x in self.rules_in_order if x["kind"] in ("rule", "section") and x["key"].startswith(parent_key + ".")][:12]
            parent_rule = self.rules_by_key.get(parent_key)
            msg = f"No rule with code “{code}” exists in this rulebook."
            if parent_rule:
                msg += f" {parent_rule['code']} ({parent_rule.get('title') or parent_rule['section_title']}) exists; its entries are listed below."
            return {"type": "no_match", "mode": "code", "query_code": code, "message": msg,
                    "suggestions": siblings, "parent": parent_rule["code"] if parent_rule else None, "timing": timing.stages}
        if r["kind"] == "rule":
            return {"type": "code", "mode": "code", "query_code": code, "results": [self.rule_payload(r)], "timing": timing.stages}
        # section / chapter / part: return the heading and all rules below it
        rules = self.section_rules(r["code"])
        return {"type": "code", "mode": "code", "query_code": code, "heading": {"code": r["code"], "kind": r["kind"], "title": r.get("title", ""), "pages": r["pages"]},
                "results": rules, "timing": timing.stages}

    # ------------------------------------------------------------------ query processing
    def correct_tokens(self, toks: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        corrected, changes = [], []
        for t in toks:
            if (t in self.vocab_df or not t.isalpha() or len(t) < 4 or t in STOPWORDS
                    or t in self.known_words or stem(t) in self.known_stems):
                corrected.append(t)
                continue
            maxd = 1 if len(t) <= 5 else 2
            cands = process.extract(t, self.vocab_list, scorer=DamerauLevenshtein.distance, limit=8, score_cutoff=maxd)
            if cands:
                best = min(cands, key=lambda c: (c[1], -self.vocab_df[c[0]]))
                # require the correction to be reasonably common to avoid odd matches
                if self.vocab_df[best[0]] >= 2 or best[1] == 1:
                    corrected.append(best[0])
                    changes.append((t, best[0]))
                    continue
            corrected.append(t)
        return corrected, changes

    def allowed(self, doc: dict, scope: str) -> bool:
        if scope == "all":
            return True
        if doc["scope"] == "general":
            return True
        return doc["scope"] == scope.upper()

    # ------------------------------------------------------------------ search
    def search(self, query: str, scope: str = "all") -> dict:
        timing = Timing()
        query = (query or "").strip()
        scope = (scope or "all").lower()
        if scope not in ("all", "ev", "cv"):
            scope = "all"
        if not query:
            return {"type": "error", "message": "Please enter a question, a phrase or a rule code.", "timing": timing.stages}
        if len(query) > config.MAX_QUERY_CHARS:
            return {"type": "error", "message": f"Query too long (max {config.MAX_QUERY_CHARS} characters).", "timing": timing.stages}

        code = self.normalise_code(query)
        if code:
            t = time.perf_counter()
            res = self.lookup_code(code, timing)
            timing.mark("lookup", t)
            res["timing"] = timing.stages
            res["total_ms"] = timing.total_ms()
            res["query"] = query
            res["scope"] = scope
            return res

        with self.lock:  # one expensive search at a time; the server bounds the queue
            return self._hybrid(query, scope, timing)

    def _hybrid(self, query: str, scope: str, timing: Timing) -> dict:
        degraded: list[str] = []
        # --- query analysis
        t = time.perf_counter()
        rtoks = raw_tokens(query)
        corrected, changes = self.correct_tokens(rtoks)
        expansions = expand_glossary_in_query(query, self.data["glossary"])
        corrected_query = " ".join(corrected) if changes else query
        syn = self.expand_synonyms(normalize_for_search(corrected_query))
        bm25_query = [stem(w) for w in corrected]
        bm25_query += [stem(w) for e in expansions for w in raw_tokens(e[1])]
        bm25_query += [stem(w) for phrase in syn for w in raw_tokens(phrase) if w not in STOPWORDS]
        extra = [f"{k}: {v}" for k, v in expansions] + syn
        dense_query = corrected_query + ("" if not extra else " (" + "; ".join(extra) + ")")
        mentioned = [f"{m.group(1)} {m.group(2)}" for m in CODE_IN_TEXT_RE.finditer(query)]
        content_tokens = [w for w in corrected if w not in STOPWORDS]
        timing.mark("analyse", t)

        # --- keyword retrieval
        t = time.perf_counter()
        bm = self.bm25.get_scores(bm25_query)
        bm_rank = np.argsort(-bm)
        bm_top = [int(i) for i in bm_rank[: config.BM25_TOP_K] if bm[i] > 0]
        timing.mark("bm25", t)

        # --- dense retrieval
        t = time.perf_counter()
        dense_top: list[int] = []
        sims = None
        if self.semantic_available and self.embedder is not None:
            try:
                qv = self.embedder.encode([config.EMBEDDING_QUERY_PREFIX + dense_query], normalize_embeddings=True, convert_to_numpy=True)[0]
                sims = self.emb @ qv
                dense_top = [int(i) for i in np.argsort(-sims)[: config.DENSE_TOP_K]]
            except Exception as e:  # noqa: BLE001
                log.exception("dense retrieval failed")
                degraded.append(f"semantic search failed ({type(e).__name__}); keyword-only results")
        else:
            degraded.append("semantic model not loaded; keyword-only results")
        timing.mark("dense", t)

        # --- fuse candidates (RRF) and filter by applicability
        t = time.perf_counter()
        rrf: dict[int, float] = {}
        for rank, i in enumerate(bm_top):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + rank)
        for rank, i in enumerate(dense_top):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + rank)
        cand = sorted(rrf, key=lambda i: -rrf[i])
        cand = [i for i in cand if self.allowed(self.docs[i], scope)][: config.RERANK_MAX_CANDIDATES]
        timing.mark("fuse", t)

        if not cand:
            return self._no_match(query, scope, corrected_query, changes, expansions, timing, degraded, hint="no_candidates")

        # --- rerank
        t = time.perf_counter()
        scores: dict[int, float] = {}
        reranked = False
        elapsed = time.perf_counter() - timing.t0
        if self.rerank_available and self.reranker is not None and elapsed < config.RERANK_BUDGET_SECONDS:
            try:
                rq = dense_query if RERANK_WITH_EXPANSIONS else corrected_query
                pairs = [(rq, self.docs[i]["rerank_text"]) for i in cand]
                import torch
                s = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False, activation_fn=torch.nn.Identity())
                s = np.asarray(s, dtype=np.float32).reshape(-1)
                scores = {i: float(v) for i, v in zip(cand, s)}
                reranked = True
            except Exception as e:  # noqa: BLE001
                log.exception("rerank failed")
                degraded.append(f"reranker failed ({type(e).__name__}); results ordered by keyword+semantic fusion only")
        else:
            if not self.rerank_available:
                degraded.append("reranker not loaded; results ordered by keyword+semantic fusion only")
            else:
                degraded.append("retrieval exceeded the time budget; reranking skipped")
        timing.mark("rerank", t)

        if timing.total_ms() > config.SEARCH_BUDGET_SECONDS * 1000:
            return {"type": "error", "query": query, "scope": scope, "timing": timing.stages, "total_ms": timing.total_ms(),
                    "message": f"Search exceeded the {config.SEARCH_BUDGET_SECONDS:.0f} s budget and was abandoned. Try a shorter, more specific query."}

        # --- decide
        t = time.perf_counter()
        fused: dict[int, float] = {}
        if reranked:
            bm_max = float(bm.max()) if len(bm) and bm.max() > 0 else 1.0
            for i in cand:
                cos = float(sims[i]) if sims is not None else 0.5
                fused[i] = scores[i] + FUSION_DENSE_WEIGHT * (cos - 0.5) + FUSION_BM25_WEIGHT * float(bm[i]) / bm_max
            ordered = sorted(cand, key=lambda i: -fused[i])
            top_score = fused[ordered[0]]
            top_rr = max(scores[i] for i in ordered[:3])
            top_doc = self.docs[ordered[0]]
            top_tokens = set(top_doc["tokens"])
            matched = [w for w in content_tokens if stem(w) in top_tokens]
            syn_tokens = {stem(w) for phrase in syn for w in raw_tokens(phrase)} | {stem(w) for e in expansions for w in raw_tokens(e[1])}
            matched_syn = bool(syn_tokens & top_tokens)
            frac = (len(matched) / len(content_tokens)) if content_tokens else 1.0
            top_cos = float(sims[ordered[0]]) if sims is not None else 1.0
            weak = top_rr < WEAK_RR
            if top_rr < T_NO_MATCH or (weak and top_cos < WEAK_COS_MIN and frac < 0.5 and not matched_syn):
                return self._no_match(query, scope, corrected_query, changes, expansions, timing, degraded, hint="low_score",
                                      near=[self.docs[i]["code"] for i in ordered[:3]], scores={self.docs[i]["code"]: round(scores[i], 2) for i in ordered[:3]})
            # keep: rules the reranker judges relevant on their own (logit >= 0),
            # plus near-ties of the top result; never padded to a fixed count
            keep = [i for i in ordered if scores[i] >= WEAK_RR or (scores[i] >= T_INCLUDE and fused[i] >= top_score - DELTA_INCLUDE)][: config.MAX_RESULTS]
            if ordered[0] not in keep:
                keep.insert(0, ordered[0])
            # No clear winner and everything comes from one section (typical for a
            # bare term such as "BSPD"): present the section in rulebook order.
            if len(keep) > 1 and fused[keep[0]] - fused[keep[1]] <= 2 * DELTA_AMBIGUOUS and len({self.docs[i]["section"] for i in keep}) == 1:
                keep.sort(key=lambda i: self.rules_by_key[self.docs[i]["key"]]["order"])
            # Rules explicitly referenced ("see D 10.1.7") by a shown result are
            # pulled in directly after it when the reranker also finds them
            # relevant on their own (positive logit). They supply the referenced
            # conditions the result depends on.
            referenced_by: dict[int, str] = {}
            cand_by_key = {self.docs[i]["key"]: i for i in cand}
            for i in list(keep[:3]):
                r = self.rules_by_key[self.docs[i]["key"]]
                for c in r["references"]:
                    j = cand_by_key.get(c.replace(" ", "").upper())
                    if j is not None and j not in keep and scores.get(j, -9) >= 0.0 and len(keep) < config.MAX_RESULTS and len(referenced_by) < 2:
                        keep.insert(keep.index(i) + 1, j)
                        referenced_by[j] = r["code"]
        else:
            ordered = cand
            keep = ordered[: min(5, config.MAX_RESULTS)]
            top_score = None
            weak = False

        results = []
        for i in keep:
            d = self.docs[i]
            r = self.rules_by_key[d["key"]]
            payload = self.rule_payload(r, with_context=(i == keep[0]))
            payload["why"] = {
                "rerank_score": round(scores[i], 3) if reranked else None,
                "fused_score": round(fused[i], 3) if reranked else None,
                "dense_cosine": round(float(sims[i]), 3) if sims is not None else None,
                "bm25_rank": (bm_top.index(i) + 1) if i in bm_top else None,
                "dense_rank": (dense_top.index(i) + 1) if i in dense_top else None,
                "matched_terms": sorted({w for w in content_tokens if stem(w) in set(d["tokens"])})[:12],
                "referenced_by": referenced_by.get(i) if reranked else None,
            }
            results.append(payload)

        clarification = None
        if reranked and len(ordered) > 1:
            a, b = ordered[0], ordered[1]
            close = fused[a] - fused[b] <= DELTA_AMBIGUOUS
            da, db = self.docs[a], self.docs[b]
            if close and scope == "all" and {da["scope"], db["scope"]} == {"EV", "CV"}:
                clarification = {
                    "kind": "applicability",
                    "question": "This topic has separate EV and CV provisions. Which vehicle class do you mean?",
                    "options": [{"label": f"{da['scope']} — {da['code']}", "scope": da["scope"].lower(), "code": da["code"]},
                                {"label": f"{db['scope']} — {db['code']}", "scope": db["scope"].lower(), "code": db["code"]}],
                }
            elif close and da["section"] != db["section"] and len(content_tokens) <= 2:
                clarification = {
                    "kind": "topic",
                    "question": "Your query is short and matches different sections about equally. Which did you mean?",
                    "options": [{"label": f"{da['code']} — {da['heading']}", "code": da["code"]},
                                {"label": f"{db['code']} — {db['heading']}", "code": db["code"]}],
                }
        timing.mark("decide", t)
        if self.debug:
            self.last_debug = {"dense_query": dense_query, "bm25_query": bm25_query, "syn": syn, "corrections": changes,
                               "cands": [{"code": self.docs[i]["code"], "rr": scores.get(i), "fused": fused.get(i), "cos": float(sims[i]) if sims is not None else None,
                                          "bm": float(bm[i]), "bm_rank": (bm_top.index(i) + 1) if i in bm_top else None, "d_rank": (dense_top.index(i) + 1) if i in dense_top else None}
                                         for i in (ordered if reranked else cand)]}
        return {
            "type": "clarification" if clarification else "results",
            "mode": "hybrid",
            "query": query, "scope": scope,
            "corrected_query": corrected_query if changes else None,
            "corrections": changes,
            "expansions": expansions,
            "synonyms": syn,
            "mentioned_codes": [c for c in mentioned if c.replace(' ', '').upper() in self.rules_by_key],
            "results": results,
            "clarification": clarification,
            "reranked": reranked,
            "weak": weak,
            "degraded": degraded,
            "top_score": round(top_score, 3) if top_score is not None else None,
            "timing": timing.stages,
            "total_ms": timing.total_ms(),
        }

    def _no_match(self, query, scope, corrected_query, changes, expansions, timing, degraded, hint="", near=None, scores=None) -> dict:
        tips = ["Use the technical term the rulebook uses (e.g. “ground clearance” rather than “ride height”).",
                "Add a component or system name (e.g. “brake pedal”, “accumulator container”, “impact attenuator”).",
                "If you know the rule code, type it directly (e.g. T 4.11.1).",
                "Check the applicability filter: EV and CV hide rules that belong only to the other class."]
        if scope != "all":
            tips.insert(0, f"Switch the filter from {scope.upper()} to All to search every rule.")
        if self.debug:
            self.last_debug = {"no_match": True, "hint": hint, "near": near, "scores": scores}
        return {"type": "no_match", "mode": "hybrid", "query": query, "scope": scope,
                "corrected_query": corrected_query if changes else None, "corrections": changes, "expansions": expansions,
                "message": "No reliable match found.", "tips": tips, "hint": hint, "near": near or [], "near_scores": scores or {},
                "degraded": degraded, "timing": timing.stages, "total_ms": timing.total_ms()}
