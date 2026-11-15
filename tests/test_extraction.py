"""Extraction checks: edge cases, displayed wording and code/page associations.

These tests read the built index (data/index/rules.json) and compare it with
text taken directly from the PDF with PyMuPDF, independent of the parser.
Run with:  uv run pytest -q
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "index"
PDF = ROOT / "data" / "pdf" / "FS_Rules_2027_v1.0.pdf"


@pytest.fixture(scope="module")
def data():
    if not (INDEX / "rules.json").exists():
        pytest.skip("index not built")
    return json.loads((INDEX / "rules.json").read_text())


@pytest.fixture(scope="module")
def rules(data):
    return {r["code"]: r for r in data["rules"]}


@pytest.fixture(scope="module")
def report():
    return json.loads((INDEX / "extraction_report.json").read_text())


@pytest.fixture(scope="module")
def doc():
    if not PDF.exists():
        pytest.skip("pdf missing")
    return pymupdf.open(str(PDF))


def page_text(doc, page: int) -> str:
    return doc[page - 1].get_text("text").replace(" ", " ")


# ---------------------------------------------------------------- coverage
def test_counts_and_no_gaps(data, report):
    assert report["counts"]["rules"] == 1263
    assert report["counts"]["chapters"] == 61
    assert report["counts"]["sections"] == 275
    assert report["duplicates"] == []
    assert report["numbering_gaps"] == []
    assert report["toc_mismatches"] == []
    assert report["empty_rules"] == []
    assert report["unassigned_text"] == []
    assert report["issues"] == []


def test_document_identity(data):
    assert data["document"]["title"] == "Formula Student Rules 2027"
    assert data["document"]["version"] == "Version 1.0"
    assert data["document"]["pages"] == 115
    assert len(data["document"]["sha256"]) == 64


def test_every_rule_code_unique_and_has_heading_path(data):
    seen = set()
    for r in data["rules"]:
        assert r["key"] not in seen
        seen.add(r["key"])
        if r["kind"] == "rule":
            assert r["chapter"] and r["section"], r["code"]
            assert r["pages"], r["code"]
            assert r["display_text"].strip(), r["code"]


# ---------------------------------------------------------------- edge cases
def test_t4111_is_on_page_39_despite_t5_running_header(rules, doc):
    r = rules["T 4.11.1"]
    assert r["pages"] == [39]
    assert r["section"] == "T 4.11" and r["section_title"] == "Driver Egress"
    assert r["chapter"] == "T 4"
    assert "exit to the side of the vehicle in less than 5 s" in r["display_text"]
    assert "T 5 Driver Restraint System" not in r["display_text"]
    # the same sentence is really on page 39 of the PDF
    assert "exit to the side of the vehicle in less than 5 s" in page_text(doc, 39)


def test_t631_continues_from_page_42_to_43(rules, doc):
    r = rules["T 6.3.1"]
    assert r["pages"] == [42, 43]
    items = [b for b in r["blocks"] if b["type"] == "li"]
    assert [b["page"] for b in items] == [43, 43]
    assert items[0]["text"] == "the hydraulic brake system is actuated"
    assert items[1]["text"] == "[EV ONLY] or the electric brake system is actuated, see EV 2.2.3"
    assert r["blocks"][0]["text"] == "The vehicle must be equipped with one brake light that is illuminated if and only if"
    assert "T 7 Powertrain" not in r["display_text"]  # running header of page 43 excluded
    assert "if and only if" in page_text(doc, 42) and "electric brake system is actuated" in page_text(doc, 43)


def test_running_headers_and_footers_absent(data):
    for r in data["rules"]:
        t = r["display_text"]
        assert "Formula Student Rules 2027" not in t, r["code"]
        assert not re.search(r"\b\d{1,3} / 115\b", t), r["code"]


def test_changelog_and_contents_not_rules(data):
    codes = {r["code"] for r in data["rules"]}
    assert "Changelog" not in codes
    assert len(data["changelog"]) >= 100
    assert all("change" in c for c in data["changelog"])
    assert any(c["rule"] == "A 1.2.8" for c in data["changelog"])
    # contents page entries are parsed separately and never appear as rule text
    assert len(data["toc"]) >= 60


def test_glossary(data):
    g = data["glossary"]
    assert g["BSPD"] == "Brake System Plausibility Device"
    assert g["ASES"] == "Accumulator Structural Equivalency Spreadsheet"
    assert g["BPEFS"] == "Business Plan Executive & Financial Summary"
    assert len(g) == 71


# ---------------------------------------------------------------- wording fidelity
@pytest.mark.parametrize("code,page,snippet", [
    ("T 2.2.1", 21, "The minimum static ground clearance of any portion of the vehicle, other than the tires, including a driver, must be 30 mm."),
    ("T 11.6.2", 49, "The action of opening the SDC must occur if the implausibility is persistent for more than 500 ms."),
    ("CV 1.1.1", 61, "displacement not exceeding 710 cm3 per cycle"),
    ("EV 4.1.1", 71, "600 V DC"),
    ("T 5.2.2", 39, "SFI spec harnesses must be replaced following December 31st of the 2nd year after the date of manufacture"),
    ("T 7.2.5", 43, "minimum volume of 10 % of the fluid being contained or 900 mL whichever is greater"),
])
def test_wording_matches_pdf(rules, doc, code, page, snippet):
    r = rules[code]
    assert page in r["pages"]
    assert snippet in r["display_text"]
    flat = re.sub(r"\s+", " ", page_text(doc, page))
    assert snippet in flat


def test_hyphenation_joined_without_corruption(rules):
    # 'configu-ration' is split by a soft hyphen in the PDF (T 5.2.1), 'driver-applied' by a hard hyphen (T 6.1.10)
    assert "configuration is allowed" in rules["T 5.2.1"]["display_text"]
    assert "driver-applied loads" in rules["T 6.1.10"]["display_text"]


def test_mixed_applicability_provisions_preserved(rules):
    r = rules["T 11.6.1"]
    app = r["applicability"]
    assert app["scope"] == "general"
    assert sorted(app["provision_tags"]) == ["CV", "EV"]
    texts = [p["text"] for p in app["provisions"]]
    assert any(t.startswith("[EV ONLY] ≥5 kW") for t in texts)
    assert any(t.startswith("[CV ONLY] the throttle position is more than 25 %") for t in texts)


def test_heading_level_applicability(rules):
    assert rules["A 4.3.1"]["applicability"]["scope"] == "EV"
    assert rules["A 4.3.1"]["applicability"]["source"] == "heading"
    assert rules["T 7.2.1"]["applicability"]["scope"] == "CV"
    assert rules["T 7.2.1"]["applicability"]["source"] == "rule"
    assert rules["CV 2.1.1"]["applicability"]["scope"] == "CV"
    assert rules["EV 5.7.1"]["applicability"]["scope"] == "EV"


# ---------------------------------------------------------------- tables / figures
def test_table_1_rows(data, rules):
    r = rules["A 1.2.8"]
    tbl = [b for b in r["blocks"] if b["type"] == "table"]
    assert len(tbl) == 1
    a = data["assets"][tbl[0]["asset"]]
    assert a["caption"] == "Table 1: Maximum points awarded"
    assert ["Endurance", "250 points", "-"] in a["rows"]
    assert ["Overall", "1000 points", "600 points"] in a["rows"]
    assert (INDEX / a["image"]).exists()


def test_table_2_material_requirements(data, rules):
    a = data["assets"][data["numbered_assets"]["Table 2"]]
    assert a["rows"][0] == ["Item or application", "Minimum wall thickness", "Minimum cross sectional area", "Minimum area moment of inertia"]
    assert ["Main and front hoops, shoulder harness mounting bar", "2.0 mm", "173 mm2", "11 320 mm4"] in a["rows"]


def test_split_table_in_12_1_4(data, rules):
    r = rules["IN 12.1.4"]
    assert r["pages"] == [92, 93]
    tbl = [b for b in r["blocks"] if b["type"] == "table"]
    a = data["assets"][tbl[0]["asset"]]
    assert a["pages"] == [92, 93]
    assert a["rows"][0] == ["", "Acceleration", "Skidpad", "Autocross", "Endurance", "Trackdrive"]
    assert a["rows"][1] == ["Group A", "0.3 s", "0.2 s", "2 s", "30 s", "30 s"]
    assert len(a["extra_images"]) == 1


def test_uncaptioned_table_d_10_1_7(data, rules):
    a = data["assets"][[b for b in rules["D 10.1.7"]["blocks"] if b["type"] == "table"][0]["asset"]]
    assert ["DOO", "2 s", "0.2 s", "2 s", "2 s", "2 s"] in a["rows"]
    assert ["USS", "DQ", "DQ", "DQ", "n/a", "−50 points"] in a["rows"]


def test_all_22_figures_have_images_and_are_referenced(data):
    figs = [a for a in data["assets"].values() if a["kind"] == "figure"]
    assert len(figs) == 22
    for a in figs:
        assert (INDEX / a["image"]).exists(), a["id"]
        assert a["caption"].startswith(f"Figure {a['number']}:")
    assert "Figure 11" in data["numbered_assets"]
    assert "Figure 11" in {x for r in data["rules"] for x in r["asset_refs"]}


def test_figure_first_rules(data, rules):
    for code in ("EV 6.1.1", "CV 4.1.1"):
        r = rules[code]
        assert r["blocks"][0]["type"] == "figure"
        assert r["blocks"][1]["type"] == "p" and r["blocks"][1]["text"].startswith("The Shutdown Circuit (SDC)")


def test_formulas_rendered(data, rules):
    r = rules["D 9.1.1"]
    f = [b for b in r["blocks"] if b["type"] == "formula"]
    assert f and (INDEX / data["assets"][f[0]["asset"]]["image"]).exists()


def test_references_resolved(rules):
    assert "T 13.3" in rules["T 4.11.1"]["references"]
    assert "A 1.5" not in rules["T 3.17.7"]["references"]  # "A 1.5 mm" is a measurement, not a code
    assert "T 3.19.4" in rules["T 3.17.7"]["references"]
