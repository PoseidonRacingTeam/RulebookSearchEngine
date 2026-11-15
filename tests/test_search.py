"""Search behaviour tests (deterministic code lookup, filters, resilience).

Relevance quality is measured separately by eval/run_eval.py; these tests
check the contracts that must always hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.search import SearchEngine  # noqa: E402


@pytest.fixture(scope="module")
def eng():
    if not (config.INDEX_DIR / "meta.json").exists():
        pytest.skip("index not built")
    e = SearchEngine(config.INDEX_DIR, config.PDF_PATH, config.MODELS_DIR)
    e.load()
    assert e.ready, e.error
    return e


# ------------------------------------------------------------ code lookup
@pytest.mark.parametrize("q", ["T 4.11.1", "T4.11.1", "t 4.11.1", "T  4 . 11 . 1", "t4.11.1.", " T-4.11.1 "])
def test_code_lookup_normalises_spacing(eng, q):
    r = eng.search(q)
    assert r["type"] == "code" and r["mode"] == "code"
    assert [x["code"] for x in r["results"]] == ["T 4.11.1"]
    assert r["results"][0]["pages"] == [39]


def test_unknown_code_is_not_silently_replaced(eng):
    r = eng.search("T 4.11.9")
    assert r["type"] == "no_match" and r["mode"] == "code"
    assert r["results"] == [] if "results" in r else True
    assert "T 4.11.9" in r["message"]
    assert r["suggestions"] == ["T 4.11.1"]  # offered as navigation, not as the answer
    r = eng.search("D 99.1.1")
    assert r["type"] == "no_match" and not r.get("results")


def test_section_and_chapter_lookup(eng):
    r = eng.search("T 2.2")
    assert r["type"] == "code" and r["heading"]["title"] == "Ground Clearance"
    assert [x["code"] for x in r["results"]] == ["T 2.2.1", "T 2.2.2"]
    r = eng.search("T 6")
    assert r["type"] == "code" and r["heading"]["kind"] == "chapter"
    assert r["results"][0]["code"] == "T 6.1.1" and r["results"][-1]["code"] == "T 6.3.3"


def test_code_lookup_bypasses_ranking(eng):
    r = eng.search("IN 12.1.4")
    assert r["timing"].get("lookup", 0) < 50
    assert "rerank" not in r["timing"]


# ------------------------------------------------------------ payload fidelity
def test_rule_payload_has_everything_the_ui_needs(eng):
    r = eng.search("T 6.3.1")["results"][0]
    assert r["display_text"].startswith("The vehicle must be equipped with one brake light")
    assert r["applicability"]["label"] == "All vehicles; contains EV-only provisions"
    assert [x["code"] for x in r["references"]] == ["EV 2.2.3"]
    assert r["references"][0]["display_text"].startswith("Regenerating energy is allowed")
    assert r["context"]["prev"] == "T 6.2.3" and r["context"]["next"] == "T 6.3.2"
    assert [s["code"] for s in r["context"]["section_rules"]] == ["T 6.3.1", "T 6.3.2", "T 6.3.3"]


def test_asset_payload(eng):
    r = eng.search("A 1.2.8")["results"][0]
    t = [b for b in r["blocks"] if b["type"] == "table"][0]
    assert t["asset_data"]["caption"] == "Table 1: Maximum points awarded"
    assert t["asset_data"]["image"].endswith(".png")


# ------------------------------------------------------------ hybrid behaviour
def test_hybrid_returns_results_with_scores_and_timing(eng):
    r = eng.search("ground clearance")
    assert r["type"] == "results" and r["reranked"] and not r["degraded"]
    assert r["results"][0]["code"] == "T 2.2.1"
    assert set(r["timing"]) >= {"bm25", "dense", "rerank"}
    assert r["total_ms"] < 10000
    assert r["results"][0]["why"]["rerank_score"] is not None
    assert "confidence" not in r["results"][0]["why"]


def test_no_match_for_off_topic(eng):
    r = eng.search("what is the size of the moon")
    assert r["type"] == "no_match"
    assert r["message"] == "No reliable match found."
    assert r["tips"]


def test_scope_filter_hides_other_class_but_keeps_general(eng):
    cv = eng.search("maximum tractive system voltage", "cv")
    assert all(x["code"] != "EV 4.1.1" for x in cv.get("results", []))
    ev = eng.search("engine displacement limit", "ev")
    assert all(x["code"] != "CV 1.1.1" for x in ev.get("results", []))
    ev2 = eng.search("brake light", "ev")
    assert ev2["type"] == "results" and ev2["results"][0]["code"].startswith("T 6.3")
    for x in ev2["results"]:
        assert x["applicability"]["scope"] in ("general", "EV")


def test_typo_and_abbreviation_handling(eng):
    r = eng.search("ground clearence")
    assert r["corrected_query"] == "ground clearance"
    assert r["results"][0]["code"] == "T 2.2.1"
    r = eng.search("BSPD")
    assert ("BSPD", "Brake System Plausibility Device") in r["expansions"]
    assert r["results"][0]["section"] == "T 11.6"


def test_empty_and_overlong_queries(eng):
    assert eng.search("")["type"] == "error"
    assert eng.search("x" * 500)["type"] == "error"


def test_results_are_never_padded(eng):
    r = eng.search("how old must team members be")
    assert r["results"][0]["code"] == "A 4.2.9"
    assert len(r["results"]) <= config.MAX_RESULTS
