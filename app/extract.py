"""PDF extraction for the Formula Student rulebook.

Parses the Typst-generated rules PDF into complete numbered rules, keeping the
original wording, list structure, tables, figures and formulas, with page
references. Nothing here paraphrases or "fixes" text: every string stored for
display is copied from the PDF at a recorded page/bbox.

Layout facts (measured on FS_Rules_2027_v1.0.pdf, page size 595x842 pt):
  - running header at y~24 (size 12), footer at y~815
  - part heading: size 24   ("T" / "General Technical Requirements")
  - chapter heading: size 18 ("T 5" / "Driver Restraint System")
  - section heading: size 14 ("T 4.11" / "Driver Egress")
  - rule: code at x=51 (size 10), text at x=112
  - list markers at x>=120 (bullet "▪", "(a)", "(i)") with text ~15-22pt right
  - tables: Typst tables, header rule line (x0~111..120), rows ~17pt apart,
    wrapped cell lines ~13.6pt apart
  - figures are vector drawings followed by a "Figure N:" caption
  - formulas use math fonts (NewCMMath / NimbusRoman) and fraction bars
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pymupdf

PARSER_VERSION = "1.4.1"

HEADER_Y_MAX = 35.0      # anything above this is the running header
FOOTER_Y_MIN = 800.0     # anything below this is the footer
CODE_X = 51.0
BODY_X = 112.0
STD_X = {51, 112, 113, 115, 122, 123, 137, 144, 147, 154, 157, 162, 169, 175}
LIST_TEXT_X = {137, 144, 162, 169, 175}
WRAP_GAP = 15.5          # baseline delta below this => wrapped line of the same row/paragraph
PARA_GAP = 16.0          # baseline delta above this between x=112 lines => new paragraph
TABLE_GAP = 26.0         # baseline delta above this => separate table region
MATH_FONTS = ("NewCMMath", "NimbusRoman", "Libertinus")

CHAPTER_PREFIXES = ("A", "T", "CV", "EV", "IN", "S", "D")
CODE_RE = re.compile(r"^(A|T|CV|EV|IN|S|D)[  ](\d+)(?:\.(\d+))?(?:\.(\d+))?$")
# references inside running text, e.g. "T 4.3.4", "EV 6", "see IN 8"
REF_RE = re.compile(r"\b(A|T|CV|EV|IN|S|D)[ \u00a0](\d{1,2}(?:\.\d{1,2}){0,2})(?!\d)(?!\.\d)(?![ \u00a0]?(?:mm|cm|m|kg|g|kW|W|V|A|s|ms|Hz|%|°|N|kN|bar|MPa|L|mL|points?)\b)")
ASSET_REF_RE = re.compile(r"\b(Figure|Table)[  ](\d+)")
CAPTION_RE = re.compile(r"^(Figure|Table)[  ](\d+):\s*(.*)$")
TAG_RE = re.compile(r"\[(EV|CV|DC|HY|Autonomous Mode|Manual Mode) ONLY\]")
MARKER_RE = re.compile(r"^(▪|•|-|–|\([a-z]{1,3}\)|\([ivx]+\)|[a-z]\)|\d{1,2}\.)$")


# ----------------------------------------------------------------------------
# Low-level line model
# ----------------------------------------------------------------------------
@dataclass
class Line:
    page: int          # 1-based page number
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str
    text: str          # raw text, may contain trailing spaces / soft hyphens / nbsp
    role: str = ""     # assigned later: header, footer, table, figure, formula, body...

    @property
    def xr(self) -> int:
        return round(self.x0)

    @property
    def is_math(self) -> bool:
        return self.font.startswith(MATH_FONTS)

    def bbox(self) -> list[float]:
        return [round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)]


def page_lines(page: pymupdf.Page, pno: int) -> list[Line]:
    out: list[Line] = []
    d = page.get_text("dict")
    for b in d["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            text = "".join(s["text"] for s in l["spans"])
            if not text.strip():
                continue
            s0 = l["spans"][0]
            x0, y0, x1, y1 = l["bbox"]
            out.append(Line(pno, x0, y0, x1, y1, float(s0["size"]), s0["font"], text))
    out.sort(key=lambda r: (round(r.y0), r.x0))
    return out


def group_baselines(lines: list[Line], tol: float = 3.0) -> list[list[Line]]:
    """Group lines sharing a baseline (same visual row), ordered by x."""
    groups: list[list[Line]] = []
    for ln in sorted(lines, key=lambda r: (r.y0, r.x0)):
        if groups and abs(groups[-1][0].y0 - ln.y0) <= tol:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    for g in groups:
        g.sort(key=lambda r: r.x0)
    return groups


# ----------------------------------------------------------------------------
# Text helpers (display text is never altered beyond whitespace / hyphen joins)
# ----------------------------------------------------------------------------
def clean_display(s: str) -> str:
    s = s.replace(" ", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def join_lines(parts: list[str]) -> str:
    """Join wrapped lines. A trailing soft hyphen (U+00AD) means the word was
    hyphenated by the typesetter: drop it and join without a space. A trailing
    hard hyphen is kept and joined without a space (e.g. 'driver-' + 'applied')."""
    out = ""
    for p in parts:
        p = p.replace(" ", " ").rstrip()
        if not out:
            out = p
            continue
        if out.endswith("­"):
            out = out[:-1] + p.lstrip()
        elif out.endswith("-") and p and p[0].islower():
            out = out + p.lstrip()
        else:
            out = out + " " + p.lstrip()
    return re.sub(r"[ \t]+", " ", out).strip()


def norm_code(s: str) -> str | None:
    m = CODE_RE.match(s.replace(" ", " ").strip())
    if not m:
        return None
    parts = [m.group(2)] + [g for g in (m.group(3), m.group(4)) if g]
    return f"{m.group(1)} {'.'.join(parts)}"


def code_key(code: str) -> str:
    return re.sub(r"\s+", "", code).upper()


def code_depth(code: str) -> int:
    return code.count(".") + 1  # "T 5"->1, "T 5.1"->2, "T 5.1.1"->3


# ----------------------------------------------------------------------------
# Output model
# ----------------------------------------------------------------------------
@dataclass
class Asset:
    id: str
    kind: str                     # figure | table | formula
    number: int | None
    caption: str
    page: int
    bbox: list[float]
    image: str                    # relative path of PNG
    rows: list[list[str]] = field(default_factory=list)   # tables
    text: str = ""                # formula text / table flattened text (search only)
    pages: list[int] = field(default_factory=list)
    extra_images: list[dict] = field(default_factory=list)  # continuation clips


@dataclass
class Block:
    type: str                  # p | li | table | figure | formula | note
    text: str = ""
    level: int = 0
    marker: str = ""
    asset: str = ""
    page: int = 0
    bbox: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class Rule:
    code: str
    key: str
    kind: str                  # rule | section | chapter | part
    title: str = ""            # headings only
    chapter: str = ""          # e.g. "T 5"
    chapter_title: str = ""
    section: str = ""          # e.g. "T 5.2"
    section_title: str = ""
    part: str = ""             # e.g. "T"
    part_title: str = ""
    pages: list[int] = field(default_factory=list)
    bboxes: list[dict] = field(default_factory=list)   # [{page, bbox}]
    blocks: list[Block] = field(default_factory=list)
    display_text: str = ""
    applicability: dict = field(default_factory=dict)
    references: list[str] = field(default_factory=list)
    asset_refs: list[str] = field(default_factory=list)
    changelog: list[str] = field(default_factory=list)
    order: int = 0


# ----------------------------------------------------------------------------
# Extractor
# ----------------------------------------------------------------------------
class Extractor:
    def __init__(self, pdf_path: Path, out_dir: Path, zoom: float = 2.0):
        self.pdf_path = Path(pdf_path)
        self.out_dir = Path(out_dir)
        self.assets_dir = self.out_dir / "assets"
        self.zoom = zoom
        self.doc = pymupdf.open(str(self.pdf_path))
        self.assets: dict[str, Asset] = {}
        self.rules: list[Rule] = []
        self.glossary: dict[str, str] = {}
        self.changelog: list[dict] = []
        self.toc: list[dict] = []
        self.issues: list[dict] = []
        self.unassigned: list[dict] = []
        self.version = ""
        self.title = ""
        self._asset_counter = 0

    # ---------------- front matter ----------------
    def detect_front_matter(self) -> dict:
        """Find the page ranges of contents / changelog / abbreviations by their
        big (size 24) headings. Returns {'toc': (a,b), 'changelog': (a,b),
        'abbreviations': (a,b), 'rules_start': p}."""
        marks = {}
        for pno in range(self.doc.page_count):
            for ln in page_lines(self.doc[pno], pno + 1):
                if ln.size >= 20 and ln.y0 < 120:
                    t = clean_display(ln.text)
                    if t in ("Changelog", "Abbreviations") and t not in marks:
                        marks[t] = pno + 1
                    elif t == "Formula Student Rules 2027" or t.startswith("Formula Student Rules"):
                        self.title = t
        for ln in page_lines(self.doc[0], 1):
            if ln.y0 > FOOTER_Y_MIN and "Version" in ln.text:
                self.version = clean_display(ln.text)
        first_rule_page = None
        for pno in range(self.doc.page_count):
            for ln in page_lines(self.doc[pno], pno + 1):
                if ln.size >= 20 and ln.xr == 51 and norm_code(ln.text) is None and clean_display(ln.text) in CHAPTER_PREFIXES:
                    first_rule_page = pno + 1
                    break
            if first_rule_page:
                break
        fm = {
            "toc": (1, marks.get("Changelog", 3) - 1),
            "changelog": (marks.get("Changelog", 3), marks.get("Abbreviations", 6) - 1),
            "abbreviations": (marks.get("Abbreviations", 6), first_rule_page - 1),
            "rules_start": first_rule_page,
        }
        return fm

    def parse_toc(self, pages: tuple[int, int]) -> None:
        for pno in range(pages[0] - 1, pages[1]):
            txt = self.doc[pno].get_text("text")
            for line in txt.splitlines():
                m = re.match(r"^((?:A|T|CV|EV|IN|S|D)(?:[  ]\d+)?)\s+(.*?)\s?[. ]{3,}\s*(\d+)\s*$", line.replace(" ", " "))
                if m:
                    self.toc.append({"code": m.group(1).strip(), "title": m.group(2).strip(), "page": int(m.group(3))})
                else:
                    m2 = re.match(r"^((?:A|T|CV|EV|IN|S|D)[  ]\d+)\s*$", line.replace(" ", " "))
                    if m2:
                        self._pending_code = m2.group(1).strip()
                        continue
                    m3 = re.match(r"^\s*(.*?)\s?[. ]{3,}\s*(\d+)\s*$", line)
                    if m3 and getattr(self, "_pending_code", None):
                        self.toc.append({"code": self._pending_code, "title": m3.group(1).strip(), "page": int(m3.group(2))})
                        self._pending_code = None

    def parse_changelog(self, pages: tuple[int, int]) -> None:
        """Changelog is a 3-column table: Rule | Version | Change."""
        for pno in range(pages[0] - 1, pages[1]):
            lines = [l for l in page_lines(self.doc[pno], pno + 1) if HEADER_Y_MAX < l.y0 < FOOTER_Y_MIN and l.size < 20]
            current = None
            for grp in group_baselines(lines):
                rule = next((clean_display(l.text) for l in grp if l.xr < 100), "")
                version = next((clean_display(l.text) for l in grp if 100 <= l.xr < 160), "")
                change = " ".join(clean_display(l.text) for l in grp if l.xr >= 160)
                if rule == "Rule" and version.startswith("Version"):
                    continue
                if version:
                    current = {"rule": norm_code(rule) or rule, "version": version, "change": change, "page": pno + 1}
                    self.changelog.append(current)
                elif current and change:
                    current["change"] = join_lines([current["change"], change])

    def parse_glossary(self, pages: tuple[int, int]) -> None:
        """Abbreviations page: two columns (abbr at x~51 / x~302, expansion at x~100 / x~350)."""
        for pno in range(pages[0] - 1, pages[1]):
            lines = [l for l in page_lines(self.doc[pno], pno + 1) if HEADER_Y_MAX < l.y0 < FOOTER_Y_MIN and l.size < 20]
            last = {"L": None, "R": None}
            for grp in group_baselines(lines, tol=4.0):
                for side, (ax, ex, lim) in {"L": (51, 100, 300), "R": (302, 350, 600)}.items():
                    abbr = [l for l in grp if abs(l.xr - ax) <= 3]
                    exp = [l for l in grp if ex - 3 <= l.xr < lim]
                    if abbr:
                        key = clean_display(abbr[0].text)
                        self.glossary[key] = join_lines([" ".join(l.text for l in exp)])
                        last[side] = key
                    elif exp and last[side]:
                        self.glossary[last[side]] = join_lines([self.glossary[last[side]], " ".join(l.text for l in exp)])
        # normalise soft hyphens inside expansions
        self.glossary = {k: v.replace("­", "") for k, v in self.glossary.items()}

    # ---------------- assets ----------------
    def _render_clip(self, page_no: int, bbox: list[float], name: str) -> str:
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        page = self.doc[page_no - 1]
        rect = pymupdf.Rect(*bbox) & page.rect
        if rect.is_empty or rect.width < 2 or rect.height < 2:
            self.issues.append({"type": "render_skipped", "page": page_no, "bbox": bbox, "asset": name})
            return ""
        pix = page.get_pixmap(matrix=pymupdf.Matrix(self.zoom, self.zoom), clip=rect, alpha=False)
        path = self.assets_dir / f"{name}.png"
        pix.save(str(path))
        return f"assets/{name}.png"

    def _new_asset_id(self, kind: str, page: int) -> str:
        self._asset_counter += 1
        return f"{kind}-p{page}-{self._asset_counter}"

    # ---------------- per-page structural analysis ----------------
    def _hlines(self, page: pymupdf.Page) -> list[tuple[float, float, float]]:
        out = []
        for d in page.get_drawings():
            r = d["rect"]
            if r.width > 120 and r.height < 2.5 and r.x0 < 130 and HEADER_Y_MAX < r.y0 < FOOTER_Y_MIN:
                out.append((r.x0, r.x1, r.y0))
        return out

    @staticmethod
    def _is_body_x(ln: Line) -> bool:
        return ln.xr in STD_X or (BODY_X <= ln.x0 <= 136 and not ln.is_math)

    def _table_regions(self, page: pymupdf.Page, lines: list[Line]) -> list[dict]:
        """Find table regions: seeded by find_tables() bboxes and by Typst header
        rule lines, then expanded over adjacent 'table-like' baseline rows."""
        pno = lines[0].page if lines else page.number + 1
        candidates = [l for l in lines if HEADER_Y_MAX < l.y0 < FOOTER_Y_MIN and l.size >= 9.0]
        groups = group_baselines(candidates)
        # a row is table-like if none of its lines start at a standard body/list x
        # and (it has >=2 cells or all its lines are off-grid)
        def table_like(grp: list[Line]) -> bool:
            if any(l.xr in (51, 112, 113, 115) or l.size >= 13 for l in grp):
                return False
            if any(CAPTION_RE.match(clean_display(l.text)) for l in grp):
                return False
            if len(grp) >= 2 and grp[0].xr in (122, 123, 147, 154, 157) and MARKER_RE.match(clean_display(grp[0].text) or ""):
                return False  # list item (marker + text)
            if all(l.is_math for l in grp):
                return False
            return True

        seeds: list[tuple[float, float]] = []
        try:
            for t in page.find_tables().tables:
                x0, y0, x1, y1 = t.bbox
                if y1 - y0 > 10 and y0 > HEADER_Y_MAX and y1 < FOOTER_Y_MIN:
                    seeds.append((y0, y1))
        except Exception:  # pragma: no cover - find_tables is best effort
            pass
        for (_x0, _x1, y) in self._hlines(page):
            seeds.append((y - 1, y + 1))
        if not seeds:
            return []
        flags = [table_like(g) for g in groups]
        regions: list[dict] = []
        for (sy0, sy1) in sorted(seeds):
            idx = [i for i, g in enumerate(groups) if flags[i] and (sy0 - 20 <= g[0].y0 <= sy1 + 20 or (g[0].y0 <= sy1 and g[0].y1 >= sy0))]
            if not idx:
                continue
            lo, hi = min(idx), max(idx)
            # expand while neighbours are table-like and close
            while lo - 1 >= 0 and flags[lo - 1] and groups[lo][0].y0 - groups[lo - 1][0].y0 < TABLE_GAP:
                lo -= 1
            while hi + 1 < len(groups) and flags[hi + 1] and groups[hi + 1][0].y0 - groups[hi][0].y0 < TABLE_GAP:
                hi += 1
            # also include table-like rows across the seed itself
            region_lines = [l for g in groups[lo:hi + 1] for l in g]
            y0 = min(l.y0 for l in region_lines) - 2
            y1 = max(l.y1 for l in region_lines) + 2
            # merge with overlapping existing region
            merged = False
            for r in regions:
                if not (y1 < r["y0"] - 1 or y0 > r["y1"] + 1):
                    r["y0"], r["y1"] = min(r["y0"], y0), max(r["y1"], y1)
                    merged = True
                    break
            if not merged:
                regions.append({"y0": y0, "y1": y1, "page": pno})
        regions.sort(key=lambda r: r["y0"])
        for r in regions:
            r["lines"] = [l for l in candidates if r["y0"] <= l.y0 <= r["y1"] and table_like_line(l)]
            r["hlines"] = [h[2] for h in self._hlines(page) if r["y0"] - 3 <= h[2] <= r["y1"] + 3]
        return [r for r in regions if r["lines"]]

    def _table_rows(self, region_lines: list[Line], hlines: list[float], col_ranges: list[tuple[float, float]] | None = None):
        """Turn table lines into rows of cell strings. Columns are clusters of
        overlapping x-intervals; wrapped cell lines (baseline gap < WRAP_GAP with
        no rule line between) are merged into the previous row."""
        if not region_lines:
            return [], []
        if col_ranges is None:
            intervals = sorted([(l.x0, l.x1) for l in region_lines])
            cols: list[list[float]] = []
            for a, b in intervals:
                if cols and a <= cols[-1][1] + 1:
                    cols[-1][1] = max(cols[-1][1], b)
                else:
                    cols.append([a, b])
            col_ranges = [(a, b) for a, b in cols]

        def col_of(l: Line) -> int:
            cx = (l.x0 + l.x1) / 2
            best, bestd = 0, 1e9
            for i, (a, b) in enumerate(col_ranges):
                d = 0 if a <= cx <= b else min(abs(cx - a), abs(cx - b))
                if d < bestd:
                    best, bestd = i, d
            return best

        rows: list[list[list[str]]] = []
        prev_y = None
        for grp in group_baselines(region_lines):
            y = grp[0].y0
            new_row = True
            if prev_y is not None and y - prev_y < WRAP_GAP and not any(prev_y < h < y for h in hlines):
                new_row = False
            if new_row or not rows:
                rows.append([[] for _ in col_ranges])
            for l in grp:
                rows[-1][col_of(l)].append(l.text)
            prev_y = y
        out = [[join_lines(c) for c in r] for r in rows]
        return out, col_ranges

    def _figure_regions(self, lines: list[Line], table_regions: list[dict], drawings: list | None = None) -> list[dict]:
        """Each 'Figure N:' caption claims everything between the previous body
        text row (or previous caption / table) and itself. A rule code that has
        no text on its own baseline (the rule starts with the figure) is not a
        boundary; the clip is then narrowed to the drawn area so the code is
        not included in the image."""
        regions = []
        groups = group_baselines([l for l in lines if HEADER_Y_MAX < l.y0 < FOOTER_Y_MIN])
        code_only = {id(g[0]) for g in groups if len(g) == 1 and g[0].xr == 51 and g[0].size < 13 and norm_code(g[0].text)}
        body_rows = [l for l in lines if HEADER_Y_MAX < l.y0 < FOOTER_Y_MIN and l.size >= 9.0 and id(l) not in code_only and
                     (self._is_body_x(l) or CAPTION_RE.match(clean_display(l.text)) or l.size >= 13)]
        for ln in lines:
            m = CAPTION_RE.match(clean_display(ln.text))
            if not m or m.group(1) != "Figure" or not (HEADER_Y_MAX < ln.y0 < FOOTER_Y_MIN):
                continue
            above = [b.y1 for b in body_rows if b.y1 <= ln.y0 - 1 and b is not ln]
            for tr in table_regions:
                if tr["y1"] <= ln.y0:
                    above.append(tr["y1"])
            top = max(above) if above else HEADER_Y_MAX + 5
            x0 = 45.0
            sort_y = top + 1
            code_rows = [l for l in lines if id(l) in code_only and top <= l.y0 <= ln.y0]
            if code_rows:
                sort_y = max(l.y0 for l in code_rows) + 0.5
                xs = [d["rect"].x0 for d in (drawings or []) if top <= d["rect"].y0 and d["rect"].y1 <= ln.y0 + 2 and d["rect"].width < 500]
                x0 = max(100.0, min(xs) - 6) if xs else 100.0
            regions.append({"y0": top + 1, "y1": ln.y1 + 2, "x0": x0, "sort_y": sort_y, "caption_line": ln, "number": int(m.group(2)), "caption": clean_display(ln.text)})
        return regions

    # ---------------- main pass ----------------
    def extract(self) -> dict:
        fm = self.detect_front_matter()
        self.parse_toc(fm["toc"])
        self.parse_changelog(fm["changelog"])
        self.parse_glossary(fm["abbreviations"])

        stream: list[dict] = []   # ordered events across pages
        for pno in range(fm["rules_start"] - 1, self.doc.page_count):
            page = self.doc[pno]
            stream.extend(self._page_events(page, pno + 1))
        self._assemble(stream)
        self._finalise()
        return self.report()

    def _page_events(self, page: pymupdf.Page, pno: int) -> list[dict]:
        lines = page_lines(page, pno)
        events: list[dict] = []
        body = []
        for l in lines:
            if l.y0 < HEADER_Y_MAX:
                l.role = "header"
            elif l.y0 > FOOTER_Y_MIN:
                l.role = "footer"
            else:
                body.append(l)
        tables = self._table_regions(page, body)
        figures = self._figure_regions(body, tables, page.get_drawings())
        for t in tables:
            for l in t["lines"]:
                l.role = "table"
        for f in figures:
            for l in body:
                if f["y0"] - 1 <= l.y0 and l.y1 <= f["y1"] + 1 and l.role == "" and l is not f["caption_line"] and not (l.xr == 51 and norm_code(l.text)):
                    l.role = "figure"
            f["caption_line"].role = "caption"
        # table captions
        for l in body:
            m = CAPTION_RE.match(clean_display(l.text))
            if m and m.group(1) == "Table" and l.role == "":
                l.role = "caption"
        # wrapped caption lines (centered second line directly below a caption)
        caption_text: dict[int, str] = {}
        for i, l in enumerate(body):
            if l.role != "caption":
                continue
            txt = l.text
            y_last = l.y0
            for m in body:
                if m.role == "" and m.size >= 9.5 and 0 < m.y0 - y_last < WRAP_GAP and m.x0 > BODY_X + 20 and not self._is_body_x(m):
                    m.role = "caption_cont"
                    txt = join_lines([txt, m.text])
                    y_last = m.y0
                    for f in figures:
                        if f["caption_line"] is l:
                            f["y1"] = max(f["y1"], m.y1 + 2)
                            f["caption"] = clean_display(txt)
            caption_text[id(l)] = clean_display(txt)
        # footnotes (small text at the very bottom of the page)
        footnotes = [l for l in body if l.role == "" and l.size < 9.5 and l.y0 > 770 and l.x0 < 80]
        for l in footnotes:
            l.role = "footnote"
            events.append({"kind": "footnote", "y": l.y0, "page": pno, "text": clean_display(l.text), "bbox": l.bbox()})
        # formula regions: consecutive off-grid math rows
        formula_regions = self._formula_regions(body)
        for fr in formula_regions:
            for l in fr["lines"]:
                l.role = "formula"

        # emit table events
        for t in tables:
            rows, cols = self._table_rows(t["lines"], t["hlines"])
            aid = self._new_asset_id("table", pno)
            bbox = [45, t["y0"] - 4, page.rect.width - 45, t["y1"] + 4]
            events.append({"kind": "table", "y": t["y0"], "page": pno, "id": aid, "rows": rows, "cols": cols, "bbox": bbox,
                           "at_page_top": t["y0"] < 75, "at_page_bottom": t["y1"] > 770, "_lines": t["lines"], "_hlines": t["hlines"]})
        for f in figures:
            aid = self._new_asset_id("figure", pno)
            bbox = [f.get("x0", 45), f["y0"], page.rect.width - 45, f["y1"]]
            events.append({"kind": "figure", "y": f.get("sort_y", f["y0"]), "page": pno, "id": aid, "number": f["number"], "caption": f["caption"], "bbox": bbox})
        for fr in formula_regions:
            aid = self._new_asset_id("formula", pno)
            events.append({"kind": "formula", "y": fr["y0"], "page": pno, "id": aid, "text": fr["text"], "bbox": [100, fr["y0"] - 3, page.rect.width - 45, fr["y1"] + 3]})
        for l in body:
            m = CAPTION_RE.match(clean_display(l.text))
            if l.role == "caption" and m and m.group(1) == "Table":
                events.append({"kind": "table_caption", "y": l.y0, "page": pno, "number": int(m.group(2)), "caption": caption_text.get(id(l), clean_display(l.text)), "bbox": l.bbox()})

        # text events from remaining lines
        free = [l for l in body if l.role == ""]
        for grp in group_baselines(free):
            first = grp[0]
            text_all = [l for l in grp]
            if first.size >= 13 and first.xr == 51:
                code = norm_code(first.text)
                title = join_lines([l.text for l in grp[1:]])
                if code:
                    kind = "chapter" if first.size >= 17 else "section"
                    events.append({"kind": "heading", "level": kind, "code": code, "title": title, "y": first.y0, "page": pno, "bbox": _union(grp), "size": first.size})
                elif clean_display(first.text) in CHAPTER_PREFIXES:
                    events.append({"kind": "heading", "level": "part", "code": clean_display(first.text), "title": title, "y": first.y0, "page": pno, "bbox": _union(grp), "size": first.size})
                else:
                    self._unassigned(grp, "big heading without code")
                continue
            if first.size >= 13 and first.xr in (112, 113, 115, 116):
                # wrapped heading title line
                events.append({"kind": "heading_cont", "title": join_lines([l.text for l in grp]), "y": first.y0, "page": pno, "size": first.size})
                continue
            if first.xr == 51 and norm_code(first.text):
                code = norm_code(first.text)
                rest = join_lines([l.text for l in grp[1:]])
                events.append({"kind": "rule", "code": code, "text": rest, "y": first.y0, "page": pno, "bbox": _union(grp), "x_text": grp[1].x0 if len(grp) > 1 else BODY_X})
                continue
            if len(grp) >= 2 and MARKER_RE.match(clean_display(first.text)) and grp[1].x0 > first.x0 + 4:
                marker = clean_display(first.text)
                tx = grp[1].x0
                level = 1 if tx < 150 else 2 if tx < 185 else 3
                events.append({"kind": "li", "marker": marker, "level": level, "text": join_lines([l.text for l in grp[1:]]), "y": first.y0, "page": pno, "bbox": _union(grp), "x_text": tx})
                continue
            if first.x0 >= BODY_X - 1 and (self._is_body_x(first) or first.is_math or first.xr in LIST_TEXT_X or first.x0 < 190):
                events.append({"kind": "text", "text": join_lines([l.text for l in grp]), "y": first.y0, "page": pno, "bbox": _union(grp), "x": first.x0, "math": first.is_math})
                continue
            self._unassigned(grp, "off-grid text")
        events.sort(key=lambda e: (e["y"], 0 if e["kind"] in ("heading", "rule") else 1))
        return events

    def _formula_regions(self, body: list[Line]) -> list[dict]:
        regions = []
        free = [l for l in body if l.role == ""]
        cur: list[Line] = []
        for l in sorted(free, key=lambda r: r.y0):
            off_grid_math = l.is_math and not (BODY_X - 1 <= l.x0 <= BODY_X + 1)
            tiny = l.size < 9 and l.is_math
            if off_grid_math or tiny:
                if cur and l.y0 - cur[-1].y1 > 12:
                    regions.append(cur)
                    cur = []
                cur.append(l)
        if cur:
            regions.append(cur)
        out = []
        for reg in regions:
            y0 = min(l.y0 for l in reg)
            y1 = max(l.y1 for l in reg)
            text = " ".join(clean_display(l.text) for l in sorted(reg, key=lambda r: (round(r.y0 / 8), r.x0)))
            out.append({"lines": reg, "y0": y0, "y1": y1, "text": text})
        return out

    def _unassigned(self, grp: list[Line], why: str) -> None:
        self.unassigned.append({"page": grp[0].page, "y": round(grp[0].y0, 1), "x": round(grp[0].x0, 1), "text": clean_display(" ".join(l.text for l in grp))[:120], "why": why})

    # ---------------- assembly ----------------
    def _assemble(self, stream: list[dict]) -> None:
        part = chapter = section = None
        current: Rule | None = None
        last_text_block: Block | None = None
        last_text_y = None
        last_text_page = None
        last_text_x = None
        pending_tables: list[str] = []
        order = 0

        def start(rule: Rule):
            nonlocal current, order
            rule.order = order
            order += 1
            self.rules.append(rule)
            current = rule

        for ev in stream:
            k = ev["kind"]
            if k == "heading":
                last_text_block = None
                title = clean_display(ev["title"])
                if ev["level"] == "part":
                    part = Rule(code=ev["code"], key=code_key(ev["code"]), kind="part", title=title, part=ev["code"], part_title=title)
                    part.pages = [ev["page"]]
                    part.bboxes = [{"page": ev["page"], "bbox": ev["bbox"]}]
                    start(part)
                    chapter = section = None
                elif ev["level"] == "chapter":
                    chapter = Rule(code=ev["code"], key=code_key(ev["code"]), kind="chapter", title=title,
                                   part=part.code if part else ev["code"].split()[0], part_title=part.title if part else "",
                                   chapter=ev["code"], chapter_title=title)
                    chapter.pages = [ev["page"]]
                    chapter.bboxes = [{"page": ev["page"], "bbox": ev["bbox"]}]
                    start(chapter)
                    section = None
                else:
                    section = Rule(code=ev["code"], key=code_key(ev["code"]), kind="section", title=title,
                                   part=part.code if part else "", part_title=part.title if part else "",
                                   chapter=chapter.code if chapter else "", chapter_title=chapter.title if chapter else "",
                                   section=ev["code"], section_title=title)
                    section.pages = [ev["page"]]
                    section.bboxes = [{"page": ev["page"], "bbox": ev["bbox"]}]
                    start(section)
                continue
            if k == "heading_cont":
                if current and current.kind in ("part", "chapter", "section"):
                    current.title = join_lines([current.title, ev["title"]])
                    if current.kind == "chapter":
                        current.chapter_title = current.title
                    elif current.kind == "section":
                        current.section_title = current.title
                    elif current.kind == "part":
                        current.part_title = current.title
                else:
                    self.issues.append({"type": "orphan_heading_line", "page": ev["page"], "text": ev["title"]})
                continue
            if k in ("rule", "li", "text", "heading"):
                pending_tables = []
            if k == "rule":
                r = Rule(code=ev["code"], key=code_key(ev["code"]), kind="rule",
                         part=part.code if part else "", part_title=part.title if part else "",
                         chapter=chapter.code if chapter else "", chapter_title=chapter.title if chapter else "",
                         section=section.code if section else "", section_title=section.title if section else "")
                r.pages = [ev["page"]]
                r.bboxes = [{"page": ev["page"], "bbox": ev["bbox"]}]
                start(r)
                blk = Block(type="p", text=ev["text"], page=ev["page"], bbox=ev["bbox"])
                r.blocks.append(blk)
                last_text_block, last_text_y, last_text_page, last_text_x = blk, ev["y"], ev["page"], ev["x_text"]
                continue
            if current is None:
                self.issues.append({"type": "text_before_first_rule", "page": ev["page"], "text": str(ev.get("text", ev.get("id", "")))[:100]})
                continue
            if k == "footnote":
                current.blocks.append(Block(type="note", text="Footnote: " + ev["text"], page=ev["page"], bbox=ev["bbox"]))
                continue
            if k == "li":
                blk = Block(type="li", text=ev["text"], level=ev["level"], marker=ev["marker"], page=ev["page"], bbox=ev["bbox"])
                current.blocks.append(blk)
                last_text_block, last_text_y, last_text_page, last_text_x = blk, ev["y"], ev["page"], ev["x_text"]
                _touch(current, ev["page"], ev["bbox"])
                continue
            if k == "text":
                x = ev["x"]
                same_page = (ev["page"] == last_text_page)
                gap = (ev["y"] - last_text_y) if (same_page and last_text_y is not None) else None
                cont = False
                starts_math_symbol = bool(ev["math"]) and bool(re.match(r"^[\U0001D400-\U0001D7FF]", ev["text"].lstrip()))
                if last_text_block is not None and last_text_block.type in ("p", "li") and not (starts_math_symbol and abs(x - BODY_X) < 2):
                    if last_text_block.type == "li":
                        # wrapped list item text starts at the item's text x
                        cont = abs(x - (last_text_x or 0)) < 3 and (gap is None or gap < WRAP_GAP + 0.5 or not same_page)
                        if not same_page and abs(x - (last_text_x or 0)) < 3 and ev["y"] < 75:
                            cont = True
                    else:
                        if abs(x - BODY_X) < 2 or (BODY_X < x < 137 and not ev["math"]):
                            cont = (gap is not None and gap < PARA_GAP) or (not same_page and ev["y"] < 75 and abs(x - BODY_X) < 2)
                            if BODY_X + 2 < x < 137:
                                cont = True  # hanging-indent continuation
                        elif abs(x - BODY_X) < 2:
                            cont = gap is not None and gap < PARA_GAP
                if cont and last_text_block is not None:
                    last_text_block.text = join_lines([last_text_block.text, ev["text"]])
                    if ev["page"] not in current.pages:
                        current.pages.append(ev["page"])
                    _touch(current, ev["page"], ev["bbox"])
                else:
                    blk = Block(type="p", text=ev["text"], page=ev["page"], bbox=ev["bbox"])
                    current.blocks.append(blk)
                    last_text_block = blk
                    _touch(current, ev["page"], ev["bbox"])
                last_text_y, last_text_page = ev["y"], ev["page"]
                if last_text_block.type == "p":
                    last_text_x = BODY_X
                continue
            if k == "table":
                aid = ev["id"]
                # continuation of a table split across a page break?
                prev = current.blocks[-1] if current.blocks else None
                if prev is not None and prev.type == "table" and ev["at_page_top"] and prev.page == ev["page"] - 1 and self._asset_at_bottom.get(prev.asset):
                    a = self.assets[prev.asset]
                    rows, _ = self._table_rows_from_event(ev, a)
                    a.rows.extend(rows)
                    a.pages.append(ev["page"])
                    a.extra_images.append({"page": ev["page"], "bbox": ev["bbox"], "image": self._render_clip(ev["page"], ev["bbox"], aid)})
                    a.text = _flatten_rows(a.rows)
                    _touch(current, ev["page"], ev["bbox"])
                    continue
                a = Asset(id=aid, kind="table", number=None, caption="", page=ev["page"], bbox=ev["bbox"],
                          image=self._render_clip(ev["page"], ev["bbox"], aid), rows=ev["rows"], text=_flatten_rows(ev["rows"]), pages=[ev["page"]])
                self.assets[aid] = a
                self._asset_cols[aid] = ev["cols"]
                self._asset_at_bottom[aid] = ev["at_page_bottom"]
                self._asset_lines[aid] = (list(ev["_lines"]), list(ev["_hlines"]), 0)
                blk = Block(type="table", asset=aid, page=ev["page"], bbox=ev["bbox"])
                current.blocks.append(blk)
                last_text_block = None
                pending_tables.append(aid)
                _touch(current, ev["page"], ev["bbox"])
                continue
            if k == "table_caption":
                # caption claims all tables since the last text block
                pending_tables = [aid for aid in pending_tables if self.assets[aid].page == ev["page"] and self.assets[aid].bbox[1] < ev["bbox"][1]]
                if pending_tables:
                    for aid in pending_tables:
                        a = self.assets[aid]
                        a.number = ev["number"]
                        a.caption = ev["caption"]
                    first = self.assets[pending_tables[0]]
                    # one combined image for the captioned table(s), caption included
                    bbox = [45, first.bbox[1], first.bbox[2], ev["bbox"][3] + 3]
                    first.bbox = bbox
                    first.image = self._render_clip(first.page, bbox, first.id)
                    if len(pending_tables) > 1:
                        for aid in pending_tables[1:]:
                            extra = self.assets.pop(aid)
                            first.rows.append(["—"] * max(1, len(first.rows[0]) if first.rows else 1))
                            first.rows.extend(extra.rows)
                            first.text = _flatten_rows(first.rows)
                            current.blocks = [b for b in current.blocks if b.asset != aid]
                    self._numbered_assets[f"Table {ev['number']}"] = first.id
                else:
                    self.issues.append({"type": "table_caption_without_table", "page": ev["page"], "text": ev["caption"]})
                pending_tables = []
                continue
            if k == "figure":
                aid = ev["id"]
                a = Asset(id=aid, kind="figure", number=ev["number"], caption=ev["caption"], page=ev["page"], bbox=ev["bbox"],
                          image=self._render_clip(ev["page"], ev["bbox"], aid), pages=[ev["page"]])
                self.assets[aid] = a
                self._numbered_assets[f"Figure {ev['number']}"] = aid
                current.blocks.append(Block(type="figure", asset=aid, page=ev["page"], bbox=ev["bbox"]))
                last_text_block = None
                pending_tables = []
                _touch(current, ev["page"], ev["bbox"])
                continue
            if k == "formula":
                aid = ev["id"]
                a = Asset(id=aid, kind="formula", number=None, caption="", page=ev["page"], bbox=ev["bbox"],
                          image=self._render_clip(ev["page"], ev["bbox"], aid), text=ev["text"], pages=[ev["page"]])
                self.assets[aid] = a
                current.blocks.append(Block(type="formula", asset=aid, text=ev["text"], page=ev["page"], bbox=ev["bbox"]))
                last_text_block = None
                _touch(current, ev["page"], ev["bbox"])
                continue

    def _table_rows_from_event(self, ev: dict, asset: Asset):
        """Recompute the whole table from the lines of every page it spans.
        Continuation lines are shifted down by one page height per page so
        baseline grouping never interleaves rows from different pages."""
        lines, hlines, shift_n = self._asset_lines[asset.id]
        shift = (shift_n + 1) * 842.0
        new_lines = [Line(l.page, l.x0, l.y0 + shift, l.x1, l.y1 + shift, l.size, l.font, l.text, l.role) for l in ev["_lines"]]
        new_h = [h + shift for h in ev["_hlines"]]
        all_lines, all_h = lines + new_lines, hlines + new_h
        self._asset_lines[asset.id] = (all_lines, all_h, shift_n + 1)
        rows, cols = self._table_rows(all_lines, all_h)
        asset.rows = []
        return rows, cols

    # ---------------- finalisation ----------------
    def _finalise(self) -> None:
        by_key = {r.key: r for r in self.rules}
        for r in self.rules:
            tags_rule: list[str] = []
            for b in r.blocks:
                b.tags = sorted(set(TAG_RE.findall(b.text)))
                b.text = clean_display(b.text)
            for b in r.blocks:
                if b.type == "p" and not b.text:
                    continue
            r.blocks = [b for b in r.blocks if not (b.type == "p" and not b.text)]
            r.display_text = render_display_text(r, self.assets)
            full = " ".join(b.text for b in r.blocks if b.type in ("p", "li")) + " " + r.title
            refs = []
            for m in REF_RE.finditer(full.replace(" ", " ")):
                c = f"{m.group(1)} {m.group(2)}"
                if c != r.code and c not in refs:
                    refs.append(c)
            r.references = refs
            r.asset_refs = sorted({f"{m.group(1)} {m.group(2)}" for m in ASSET_REF_RE.finditer(full)}, key=lambda s: (s.split()[0], int(s.split()[1])))
            r.applicability = derive_applicability(r)
            r.changelog = [c["change"] for c in self.changelog if c["rule"] == r.code]
        # dangling references
        for r in self.rules:
            for c in r.references:
                if code_key(c) not in by_key:
                    self.issues.append({"type": "dangling_reference", "rule": r.code, "ref": c})

    # ---------------- report ----------------
    def report(self) -> dict:
        rules = [r for r in self.rules if r.kind == "rule"]
        sections = [r for r in self.rules if r.kind == "section"]
        chapters = [r for r in self.rules if r.kind == "chapter"]
        # duplicates
        seen: dict[str, int] = {}
        dups = []
        for r in self.rules:
            seen[r.key] = seen.get(r.key, 0) + 1
        dups = [k for k, n in seen.items() if n > 1]
        # numbering gaps within each section
        gaps = []
        from collections import defaultdict
        per_section: dict[str, list[int]] = defaultdict(list)
        for r in rules:
            per_section[r.section].append(int(r.code.rsplit(".", 1)[1]))
        for sec, nums in per_section.items():
            nums_sorted = sorted(nums)
            expected = list(range(1, max(nums_sorted) + 1))
            missing = sorted(set(expected) - set(nums_sorted))
            if missing:
                gaps.append({"section": sec, "missing": [f"{sec}.{n}" for n in missing]})
            if nums_sorted and nums_sorted[0] != 1:
                gaps.append({"section": sec, "missing": [f"{sec}.1"], "note": "first rule not .1"})
        per_chapter: dict[str, list[int]] = defaultdict(list)
        for s in sections:
            per_chapter[s.chapter].append(int(s.code.rsplit(".", 1)[1]))
        for ch, nums in per_chapter.items():
            missing = sorted(set(range(1, max(nums) + 1)) - set(nums))
            if missing:
                gaps.append({"chapter": ch, "missing": [f"{ch}.{n}" for n in missing]})
        # toc cross-check (chapter start pages)
        toc_mismatch = []
        toc_map = {code_key(t["code"]): t for t in self.toc}
        for ch in chapters:
            t = toc_map.get(ch.key)
            if not t:
                toc_mismatch.append({"code": ch.code, "problem": "not in contents"})
            elif t["page"] != ch.pages[0]:
                toc_mismatch.append({"code": ch.code, "toc_page": t["page"], "parsed_page": ch.pages[0]})
        empty = [r.code for r in rules if not r.display_text.strip()]
        no_terminal = [r.code for r in rules if r.blocks and r.blocks[-1].type in ("p", "li") and not re.search(r"[.!?:;)\]”\"]$", r.blocks[-1].text)]
        multi_page = [r.code for r in rules if len(set(r.pages)) > 1]
        return {
            "parser_version": PARSER_VERSION,
            "pdf": str(self.pdf_path),
            "pages": self.doc.page_count,
            "document_title": self.title,
            "document_version": self.version,
            "counts": {"parts": len([r for r in self.rules if r.kind == "part"]), "chapters": len(chapters), "sections": len(sections), "rules": len(rules),
                       "assets": len(self.assets), "figures": len([a for a in self.assets.values() if a.kind == "figure"]),
                       "tables": len([a for a in self.assets.values() if a.kind == "table"]), "formulas": len([a for a in self.assets.values() if a.kind == "formula"]),
                       "glossary": len(self.glossary), "changelog": len(self.changelog), "toc_entries": len(self.toc)},
            "duplicates": dups,
            "numbering_gaps": gaps,
            "toc_mismatches": toc_mismatch,
            "empty_rules": empty,
            "rules_without_terminal_punctuation": no_terminal,
            "multi_page_rules": multi_page,
            "unassigned_text": self.unassigned,
            "issues": self.issues,
        }

    # placeholders set in __init__ via properties
    @property
    def _asset_cols(self) -> dict:
        if not hasattr(self, "_ac"):
            self._ac = {}
        return self._ac

    @property
    def _asset_lines(self) -> dict:
        if not hasattr(self, "_al"):
            self._al = {}
        return self._al

    @property
    def _asset_at_bottom(self) -> dict:
        if not hasattr(self, "_ab"):
            self._ab = {}
        return self._ab

    @property
    def _numbered_assets(self) -> dict:
        if not hasattr(self, "_na"):
            self._na = {}
        return self._na

    # ---------------- serialisation ----------------
    def to_json(self) -> dict:
        rules = []
        for r in self.rules:
            d = asdict(r)
            rules.append(d)
        assets = {k: asdict(v) for k, v in self.assets.items()}
        return {
            "parser_version": PARSER_VERSION,
            "document": {"title": self.title, "version": self.version, "pages": self.doc.page_count, "file": self.pdf_path.name, "sha256": sha256_file(self.pdf_path)},
            "glossary": self.glossary,
            "changelog": self.changelog,
            "toc": self.toc,
            "numbered_assets": self._numbered_assets,
            "rules": rules,
            "assets": assets,
        }


def table_like_line(l: Line) -> bool:
    return l.size >= 9.0 and l.xr not in (51, 112, 113, 115) and not CAPTION_RE.match(clean_display(l.text)) and l.size < 13


def _union(grp: list[Line]) -> list[float]:
    return [round(min(l.x0 for l in grp), 1), round(min(l.y0 for l in grp), 1), round(max(l.x1 for l in grp), 1), round(max(l.y1 for l in grp), 1)]


def _touch(rule: Rule, page: int, bbox: list[float]) -> None:
    if page not in rule.pages:
        rule.pages.append(page)
    for b in rule.bboxes:
        if b["page"] == page:
            b["bbox"] = [min(b["bbox"][0], bbox[0]), min(b["bbox"][1], bbox[1]), max(b["bbox"][2], bbox[2]), max(b["bbox"][3], bbox[3])]
            return
    rule.bboxes.append({"page": page, "bbox": list(bbox)})


def _flatten_rows(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(c for c in r) for r in rows)


def render_display_text(r: Rule, assets: dict[str, Asset]) -> str:
    """Plain-text rendering (for copy / fallback) that keeps list structure."""
    out = []
    for b in r.blocks:
        if b.type == "p":
            out.append(b.text)
        elif b.type == "li":
            indent = "  " * (b.level - 1)
            out.append(f"{indent}{b.marker} {b.text}")
        elif b.type == "table":
            a = assets.get(b.asset)
            if a:
                out.append(f"[{a.caption or 'Table'} — see table on page {a.page}]")
                for row in a.rows:
                    out.append("  " + " | ".join(row))
        elif b.type == "figure":
            a = assets.get(b.asset)
            if a:
                out.append(f"[{a.caption} — see figure on page {a.page}]")
        elif b.type == "formula":
            out.append(f"[Formula: {b.text}]")
        elif b.type == "note":
            out.append(b.text)
    return "\n".join(out)


def derive_applicability(r: Rule) -> dict:
    """Applicability is not inferable from the chapter prefix alone. We record
    the evidence: chapter scope, heading tags, rule-level tags and per-item tags."""
    tags_heading = set(TAG_RE.findall(r.section_title or "")) | set(TAG_RE.findall(r.chapter_title or "")) | set(TAG_RE.findall(r.title or ""))
    first_text = r.blocks[0].text if r.blocks and r.blocks[0].type == "p" else ""
    lead = re.match(r"^\s*((?:\[(?:EV|CV|DC|HY|Autonomous Mode|Manual Mode) ONLY\]\s*)+)", first_text)
    tags_rule = set(TAG_RE.findall(lead.group(1))) if lead else set()
    provisions = []
    for b in r.blocks:
        if b.type in ("p", "li"):
            for t in TAG_RE.findall(b.text):
                if b.type == "li" or (b.type == "p" and b is not r.blocks[0]) or (b is r.blocks[0] and not lead):
                    provisions.append({"tag": t, "text": b.text, "level": b.level, "type": b.type})
    chapter_scope = "general"
    if r.part == "CV" or r.code.startswith("CV "):
        chapter_scope = "CV"
    elif r.part == "EV" or r.code.startswith("EV "):
        chapter_scope = "EV"
    scope = chapter_scope
    source = "chapter" if chapter_scope != "general" else "general"
    for t in ("EV", "CV"):
        if t in tags_heading and scope == "general":
            scope, source = t, "heading"
        if t in tags_rule and scope == "general":
            scope, source = t, "rule"
    mixed = sorted({p["tag"] for p in provisions if p["tag"] in ("EV", "CV")})
    labels = []
    if scope == "EV":
        labels.append("EV only")
    elif scope == "CV":
        labels.append("CV only")
    else:
        labels.append("All vehicles")
    for t in sorted(tags_heading | tags_rule):
        if t in ("DC", "HY", "Autonomous Mode", "Manual Mode"):
            labels.append(f"{t} only")
    if mixed and scope == "general":
        labels.append("contains " + " and ".join(f"{t}-only" for t in mixed) + " provisions")
    return {
        "scope": scope,                      # general | EV | CV
        "source": source,                    # chapter | heading | rule | general
        "tags": sorted(tags_heading | tags_rule),
        "provision_tags": sorted({p["tag"] for p in provisions}),
        "provisions": provisions,
        "label": "; ".join(labels),
    }


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_pdf(pdf_path: Path, out_dir: Path) -> tuple[dict, dict]:
    ex = Extractor(pdf_path, out_dir)
    report = ex.extract()
    data = ex.to_json()
    return data, report


if __name__ == "__main__":  # pragma: no cover
    import sys
    pdf = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    data, report = extract_pdf(pdf, out)
    (out / "rules.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    (out / "extraction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "unassigned_text"}, indent=1, ensure_ascii=False)[:4000])
    print("unassigned:", len(report["unassigned_text"]))
