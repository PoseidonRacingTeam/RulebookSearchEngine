# Rulebook Search Engine

Local, offline search over the **Formula Student Rules 2027, Version 1.0** PDF.
Type a question, a phrase or a rule code; the app shows the matching rule's code,
heading, complete original wording, applicability (All / EV / CV) and PDF page,
so you can verify it in the PDF yourself.

Everything runs on your Mac: a small Python (FastAPI) server on `127.0.0.1:8765`
and a plain HTML/JS page. No internet, API keys or subscriptions are needed after
the one-time setup.

## For teammates: download and run (no terminal needed)

1. On the GitHub page click the green **Code** button → **Download ZIP**, then
   double-click the ZIP in your Downloads folder. You get a folder called
   `Rulebook Search Engine` (keep everything inside it together; move the
   folder wherever you like, e.g. Documents).
2. Open the folder and **double-click `start.command`**.
   - If macOS says it "cannot be opened because it is from an unidentified
     developer": right-click (or Control-click) `start.command`, choose
     **Open**, then **Open** again. Only needed once.
   - If it says the file "could not be executed" (this happens when the project
     was uploaded to GitHub through the website): open **Terminal** (in
     Applications › Utilities), type `bash` followed by a space, drag
     `start.command` from Finder onto the Terminal window, and press Enter.
     The script repairs itself, and from then on double-clicking works.
   - A Terminal window opens and runs the one-time setup by itself: it installs
     a private Python, downloads the two search models (about 1 GB) and builds
     the rule index. This needs internet once and takes 5–10 minutes.
3. The browser opens at `http://127.0.0.1:8765/`. Leave the Terminal window open
   while you use the app; close it (or press Ctrl+C in it) to stop.
4. From then on, double-clicking `start.command` starts the app in a few seconds,
   fully offline.

The rulebook PDF is included in `data/pdf/`, so nothing else is needed.
Requirements: a Mac (Apple silicon or Intel) with about 3 GB of free disk space.

## Quick start (terminal)

```bash
cd "Rulebook Search Engine"
./setup.sh                                       # one-time, needs internet once (~5 min)
./start.command                                  # or: uv run python run.py
```

`setup.sh` uses the PDF in `data/pdf/`; to index a different rulebook pass its
path: `./setup.sh "/path/to/FS_Rules_2027_v1.0.pdf"`.

`start.command` can also be double-clicked in Finder. It starts the server,
opens `http://127.0.0.1:8765/` in your browser and keeps running in that
terminal window. The page shows the setup steps (index check, models, warm-up)
and switches to **Ready** when searches will actually work (about 6 s).

**Stop:** press `Ctrl+C` in the terminal window (or close it).

If port 8765 is taken by another program: `uv run python run.py --port 8766`.
If the app is already running, `start.command` just opens the browser.

### Normal use
- Type and the results appear (or press Enter). Each result shows the rule code
  (click it to open the PDF at that page), chapter › section, applicability
  badges, the full original text with bullets, tables, figures and formulas,
  and a **Page** button that opens the PDF (browser's own viewer) in a side
  pane with a clear **Close** button.
- **Rule codes** such as `T 4.11.1`, `t4.11.1`, `T 2.2` (a section) or `EV 6`
  (a chapter) are looked up exactly; nothing is ranked or guessed. An unknown
  code is reported as such and never replaced by a different rule.
- **Vehicle class** (Settings ⚙ › Preferences): the search is scoped to **EV**
  by default; CV and All are available there. EV and CV still include the
  general rules that apply to every vehicle; rules that belong only to the other
  class are hidden. Rules with separate `[EV ONLY]` / `[CV ONLY]` provisions are
  shown with those items labelled. The current scope is shown as a chip under
  the search field.
- **Keyboard**: `/` focuses the search field, `↓`/`↑` move through the results,
  `Enter` opens the PDF at the selected rule, `→`/`←` show more / less, `P` pins
  it, `Esc` returns to the
  field; typing any letter while a result is selected starts a new search.
- **Pinned rules and recent searches** are shown while the search field is empty
  (pin with the bookmark icon on a card or `P`; recent searches keep the last 10).
  The rule code on each card opens the PDF at that page.
- **Search as you type** (Settings ⚙ › Preferences, on by default): results
  update after a short typing pause (3+ characters); Enter still searches at
  once. At most one live search is in flight, so the server is never flooded.
- Preferences are stored in the browser (`localStorage`).
- **Everything opens in place.** Each result has one **Show more** chevron that
  reveals the rules it refers to, its figures/tables and the whole section it
  belongs to, with **Show less** at the bottom to fold it back. Rule codes inside
  any text ("see T 13.3") and the codes in notices drop down under the line you
  clicked and close with their ×; the result list, the query and the URL never
  change. Drop-downs nest, so you can follow a chain of references without
  leaving the result.
- "No reliable match found" means no rule is supported well enough to show as an
  answer; the tips explain how to refine. A "Weak match" notice means the rules
  shown are the closest found but may not answer the question.
- A clarification is asked when the two best candidates are near-ties with
  different EV/CV applicability or different sections.

### Rebuilding the index (new PDF version)

```bash
uv run python -m app.build "/path/to/new_rules.pdf"
```

The PDF is copied into `data/pdf/` (the original file is never modified), parsed,
validated and embedded (~20 s). The index stores the PDF's SHA-256, parser and
index versions and model names; on start the app refuses a stale or incompatible
index with a message telling you to rebuild. The parser is tuned to the Typst
layout of the 2027 rulebook; check `data/index/extraction_report.json` after
rebuilding for another version.

### Optional: iPhone on the same Wi-Fi
Start with `uv run python run.py --host 0.0.0.0` and open `http://<mac-ip>:8765/`
on the phone. The Mac must stay awake, running the app and on the same network;
the server does no authentication, so only do this on a trusted network.

## How it works
- **Extraction** (`app/extract.py`): PyMuPDF line geometry → parts/chapters/sections/
  1263 rules; running headers, footers, contents and changelog excluded; lists,
  hyphenation, tables (cell-clipped text + image clip), figures and formulas
  (image clips), applicability tags, glossary (71 abbreviations) and references.
  Display text is stored separately from search text and is never rewritten.
- **Search** (`app/search.py`): deterministic code lookup; otherwise BM25
  (stemmed, glossary/alias-expanded) + dense retrieval (`BAAI/bge-base-en-v1.5`),
  80 + 80 candidates fused and reranked by the cross-encoder
  `mixedbread-ai/mxbai-rerank-base-v1`; final order = reranker logit + 10·(cosine − 0.5)
  + normalised BM25. Spelling correction (Damerau-Levenshtein against the
  rulebook vocabulary, guarded by the macOS word list), abbreviation expansion
  from the rulebook glossary, directed aliases in `data/synonyms.json`.
  Thresholds were calibrated on the dev split only (`eval/results`).
  Scores are model logits, never shown as confidence percentages.
- **Server/UI** (`app/server.py`, `static/`): models stay loaded; one expensive
  search at a time with a bounded queue (HTTP 429 when busy); 8 s server budget
  and 12 s client timeout (a timeout is reported as a failed search); newer
  searches cancel and supersede older ones.

## Verification results (this Mac: Apple M5 Pro, 24 GB, macOS 26.6)

**Extraction** (`uv run pytest -q`, 42 tests pass): 1263 rules / 275 sections / 61
chapters, 22 figures, 16 tables, 7 formulas; 0 duplicates, 0 numbering gaps,
0 contents mismatches, 0 unassigned text, 0 dangling references. Edge cases
checked against the PDF: T 4.11.1 on page 39 under the "T 5" running header;
T 6.3.1 continuing from page 42 to 43 with its `[EV ONLY]` bullet; the IN 12.1.4
penalty table split across pages 92–93; Table 1/2/9; figure-first rules EV 6.1.1
and CV 4.1.1; "A 1.5 mm" not read as a rule reference.

**Search quality** (`uv run python eval/run_eval.py`, 68 source-verified queries,
expected answers recorded before looking at output; thresholds tuned on *dev* only):

| split | queries | top-1 correct | all must-include rules in top 3 | negatives handled | filter respected |
|---|---|---|---|---|---|
| dev | 40 | 34/37 (92%) | 4/6 | 3/3 | – |
| held-out | 28 | 22/23 (96%) | 3/3 | 3/3 | 2/2 |

The three starting examples: "How quickly must the driver get out?" → T 4.11.1 ✔;
"How low can the car sit?" → T 2.2.1 ✔; "What happens if I brake and accelerate
together?" → T 11.6.1/T 11.6.2 are shown with their EV/CV provisions labelled, but
**not first** (see limitations).

**Latency** (submit → complete result, after Ready):

| measurement | median | p95 | max |
|---|---|---|---|
| engine, 68 eval queries (incl. code lookups) | 0.79 s | 1.06 s | 1.12 s |
| browser, submit → rendered, online (18 searches) | 0.55 s | – | 1.55 s |
| browser, submit → rendered, offline (18 searches) | 0.81 s | – | 1.85 s |
| first search after Ready (browser) | 1.03 s | | |

Engine load to Ready: 5–7 s. Offline check: Wi-Fi switched off and a dead HTTP
proxy forced; the app restarted, reported Ready and passed the full browser
check (`eval/results/browser_check_offline.json`).

### Known limitations (honest list)
- "What happens if I brake and accelerate together?" returns the ASB rule
  T 15.1.4 first; the BSPD rules T 11.6.1/T 11.6.2 appear in positions 2–5.
  The reranker prefers "manual and autonomous braking simultaneously" over the
  BSPD wording ("hard braking … whilst power is delivered / throttle > 25 %").
- "how many shutdown buttons" shows T 11.4.4 before T 11.4.1 (both in the top 2).
- "jacks lift height" shows T 13.2.3 first; the 75 mm rule T 13.2.1 is third.
- "penalty for knocking over a cone" shows the DOO definition D 10.1.2 but not
  the penalty table D 10.1.7 (reachable via D 10.1.4 → referenced rules / the
  section context).
- The single word "light" returns TSAL rules (EV 4.10.x) rather than asking a
  clarification; the eval marks this as a miss because its accepted list was
  narrower than the TSAL section.
- Table text is reconstructed from cell geometry; spanning cells could be merged
  wrongly in another PDF version. The original image is one click away.
- Clarifications only trigger on near-ties; a confidently wrong top result is not
  detected. Always read the wording and check the page.
- Relevance and the no-match threshold are tuned for this rulebook; another
  season's PDF needs a rebuild and a fresh look at `extraction_report.json`.

## Project layout
```
app/        extract.py (PDF → rules), indexer.py, search.py, server.py, build.py, download_models.py
static/     index.html, app.js, style.css (no external assets)
data/pdf/   the indexed PDF        data/index/  rules.json, docs.json, embeddings.npy, meta.json, assets/*.png
data/synonyms.json   alias expansion list (editable)
models/     cached Hugging Face models (~1 GB, offline)
eval/       queries.json (dev/held-out), run_eval.py, results/
tests/      pytest suite + browser_check.py (Playwright)
run.py setup.sh start.command pyproject.toml uv.lock CHANGELOG.md
```
Dependencies are pinned in `pyproject.toml` / `uv.lock` (Python 3.12 managed by uv).
