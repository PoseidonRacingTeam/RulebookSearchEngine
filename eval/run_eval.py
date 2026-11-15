"""Evaluate the search engine against eval/queries.json.

Usage:
  uv run python eval/run_eval.py [--split dev|heldout|all] [--tag NAME]
                                 [--t-no-match X] [--delta-include X] [--delta-ambiguous X] [--fusion-w X]

Reports top-1 accuracy, must-include coverage in the top 3, inappropriate
matches on negative queries, filter violations and latency (median / p95 /
max), including the very first search after readiness. Writes JSON + Markdown
to eval/results/.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app import search as search_mod  # noqa: E402
from app.search import SearchEngine  # noqa: E402


def result_codes(res: dict) -> list[str]:
    return [r["code"] for r in res.get("results", [])]


def judge(q: dict, res: dict) -> dict:
    exp = q["expect"]
    codes = result_codes(res)
    top1 = codes[0] if codes else None
    t = exp["type"]
    out = {"id": q["id"], "split": q["split"], "category": q["category"], "query": q["query"], "scope": q.get("scope", "all"),
           "type": res.get("type"), "top1": top1, "returned": codes[:8], "total_ms": res.get("total_ms"), "expect": exp,
           "corrections": res.get("corrections"), "top_score": res.get("top_score")}
    ok_top1 = ok_cov = ok_neg = ok_filter = None
    if t == "codes":
        ok_top1 = res.get("type") in ("results", "code") and top1 in exp["accept_top1"]
        if exp.get("must_include"):
            ok_cov = all(c in codes[:3] for c in exp["must_include"])
        if res.get("type") == "clarification":
            # a clarification on a query with a clear answer counts as a miss for top-1
            ok_top1 = False
    elif t == "unknown_code":
        ok_neg = res.get("type") == "no_match" and not codes
    elif t == "no_match":
        ok_neg = res.get("type") == "no_match"
    elif t == "ambiguous":
        ok_top1 = res.get("type") == "clarification" or top1 in exp["accept_top1"]
    elif t == "codes_or_no_match":
        ok_filter = True
    if exp.get("must_not_include"):
        ok_filter = all(c not in codes for c in exp["must_not_include"])
    out.update({"ok_top1": ok_top1, "ok_coverage": ok_cov, "ok_negative": ok_neg, "ok_filter": ok_filter})
    out["pass"] = all(v for v in (ok_top1, ok_cov, ok_neg, ok_filter) if v is not None)
    return out


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else "n/a"


def summarise(rows: list[dict]) -> dict:
    s = {}
    for split in sorted({r["split"] for r in rows} | {"all"}):
        rs = rows if split == "all" else [r for r in rows if r["split"] == split]
        t1 = [r for r in rs if r["ok_top1"] is not None]
        cov = [r for r in rs if r["ok_coverage"] is not None]
        neg = [r for r in rs if r["ok_negative"] is not None]
        flt = [r for r in rs if r["ok_filter"] is not None]
        lat = [r["total_ms"] for r in rs if r["total_ms"] is not None]
        lat_sorted = sorted(lat)
        s[split] = {
            "n": len(rs),
            "top1": pct(sum(r["ok_top1"] for r in t1), len(t1)),
            "coverage_top3": pct(sum(r["ok_coverage"] for r in cov), len(cov)),
            "negatives_handled": pct(sum(r["ok_negative"] for r in neg), len(neg)),
            "inappropriate_matches": sum(1 for r in neg if not r["ok_negative"]),
            "filter_respected": pct(sum(r["ok_filter"] for r in flt), len(flt)),
            "pass": pct(sum(r["pass"] for r in rs), len(rs)),
            "latency_ms": {"median": round(statistics.median(lat), 1) if lat else None,
                           "p95": round(lat_sorted[min(len(lat_sorted) - 1, int(round(0.95 * len(lat_sorted))) - 1)], 1) if lat else None,
                           "max": round(max(lat), 1) if lat else None, "n": len(lat)},
        }
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--t-no-match", type=float)
    ap.add_argument("--delta-include", type=float)
    ap.add_argument("--delta-ambiguous", type=float)
    ap.add_argument("--fusion-w", type=float)
    ap.add_argument("--bm25-w", type=float)
    ap.add_argument("--rerank-expansions", type=int)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.t_no_match is not None:
        search_mod.T_NO_MATCH = args.t_no_match
        search_mod.T_INCLUDE = args.t_no_match
    if args.delta_include is not None:
        search_mod.DELTA_INCLUDE = args.delta_include
    if args.delta_ambiguous is not None:
        search_mod.DELTA_AMBIGUOUS = args.delta_ambiguous
    if args.fusion_w is not None:
        search_mod.FUSION_DENSE_WEIGHT = args.fusion_w
    if args.bm25_w is not None:
        search_mod.FUSION_BM25_WEIGHT = args.bm25_w
    if args.rerank_expansions is not None:
        search_mod.RERANK_WITH_EXPANSIONS = bool(args.rerank_expansions)

    qs = json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]
    if args.split != "all":
        qs = [q for q in qs if q["split"] == args.split]

    t0 = time.perf_counter()
    eng = SearchEngine(config.INDEX_DIR, config.PDF_PATH, config.MODELS_DIR)
    eng.load()
    load_s = round(time.perf_counter() - t0, 1)
    if not eng.ready:
        print("engine failed to load:", eng.error)
        return 1

    rows = []
    first_ms = None
    for i, q in enumerate(qs):
        t = time.perf_counter()
        res = eng.search(q["query"], q.get("scope", "all"))
        wall = round((time.perf_counter() - t) * 1000, 1)
        res["total_ms"] = wall  # wall-clock from call to complete result object
        if first_ms is None and res.get("mode") == "hybrid":
            first_ms = wall  # first full (keyword+semantic+rerank) search after readiness
        row = judge(q, res)
        rows.append(row)
        if not args.quiet:
            mark = "PASS" if row["pass"] else "FAIL"
            print(f"{mark} #{q['id']:<3} {wall:6.0f}ms {res.get('type'):<13} top={row['top1']!s:<10} score={res.get('top_score')!s:<7} {q['query']!r} -> {row['returned'][:5]}")
    summary = summarise(rows)
    out = {
        "tag": args.tag, "when": datetime.now().isoformat(timespec="seconds"), "split": args.split,
        "thresholds": {"T_NO_MATCH": search_mod.T_NO_MATCH, "T_INCLUDE": search_mod.T_INCLUDE, "DELTA_INCLUDE": search_mod.DELTA_INCLUDE,
                       "DELTA_AMBIGUOUS": search_mod.DELTA_AMBIGUOUS, "FUSION_DENSE_WEIGHT": search_mod.FUSION_DENSE_WEIGHT,
                       "FUSION_BM25_WEIGHT": search_mod.FUSION_BM25_WEIGHT, "RERANK_WITH_EXPANSIONS": search_mod.RERANK_WITH_EXPANSIONS,
                       "BM25_TOP_K": config.BM25_TOP_K, "DENSE_TOP_K": config.DENSE_TOP_K, "RERANK_MAX_CANDIDATES": config.RERANK_MAX_CANDIDATES},
        "models": {"embedding": config.EMBEDDING_MODEL, "reranker": config.RERANKER_MODEL, "device": eng.device},
        "engine_load_seconds": load_s, "first_search_ms": first_ms,
        "summary": summary, "rows": rows,
        "failures": [r for r in rows if not r["pass"]],
    }
    res_dir = ROOT / "eval" / "results"
    res_dir.mkdir(exist_ok=True)
    name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}{'-' + args.tag if args.tag else ''}-{args.split}"
    (res_dir / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    md = [f"# Eval {name}", "", f"models: {out['models']}", f"thresholds: {out['thresholds']}", f"engine load: {load_s} s, first search: {first_ms} ms", ""]
    md.append("| split | n | top-1 | coverage@3 | negatives handled | inappropriate | filter | pass | latency median/p95/max ms |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for split, s in summary.items():
        l = s["latency_ms"]
        md.append(f"| {split} | {s['n']} | {s['top1']} | {s['coverage_top3']} | {s['negatives_handled']} | {s['inappropriate_matches']} | {s['filter_respected']} | {s['pass']} | {l['median']} / {l['p95']} / {l['max']} |")
    md.append("")
    md.append("## Failures")
    for r in out["failures"]:
        md.append(f"- #{r['id']} [{r['split']}/{r['category']}] {r['query']!r} (scope {r['scope']}) -> type={r['type']} top1={r['top1']} returned={r['returned'][:5]} expected={r['expect']}")
    (res_dir / f"{name}.md").write_text("\n".join(md))
    print()
    print("\n".join(md[:12]))
    print(f"failures: {len(out['failures'])}  (written to eval/results/{name}.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
