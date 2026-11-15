"""End-to-end browser check of the running app with Playwright (Chromium).

  uv run python tests/browser_check.py [http://127.0.0.1:8765]

Verifies: status reaches Ready, a question search renders the expected rule
with its code/heading/text/page, copy button works, code lookup, unknown code,
EV/CV filter, clarification/no-match rendering, stale-response protection and
the client-measured latency (submit -> rendered) for every search.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
OUT = Path(__file__).resolve().parent.parent / "eval" / "results" / "browser_check.json"


def main() -> int:
    report = {"base": BASE, "checks": [], "latencies_ms": {}}
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        report["checks"].append({"name": name, "ok": bool(cond), "detail": detail})
        print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1200, "height": 900}, permissions=["clipboard-read", "clipboard-write"])
        page = ctx.new_page()
        page.goto(BASE, wait_until="domcontentloaded")
        page.wait_for_function("document.querySelector('#status').dataset.state === 'ready'", timeout=120000)
        check("status shows Ready", page.locator("#status-text").inner_text() == "Ready")
        check("rulebook version visible", "Version 1.0" in page.locator("#docinfo").inner_text(), page.locator("#docinfo").inner_text())
        check("search box enabled after ready", page.locator("#q").is_enabled())

        def search(q, scope="all"):
            page.check(f"input[name=scope][value={scope}]")
            page.fill("#q", q)
            t0 = time.perf_counter()
            page.press("#q", "Enter")
            page.wait_for_function("!document.querySelector('#meta').textContent.startsWith('Searching')", timeout=15000)
            page.wait_for_function("document.querySelector('#results').children.length > 0", timeout=15000)
            ms = (time.perf_counter() - t0) * 1000
            report["latencies_ms"][f"{q} [{scope}]"] = round(ms, 1)
            return ms

        # first search after readiness (question)
        ms = search("How quickly must the driver get out?")
        first = page.locator("#results .rule").first
        check("first search under 10 s (submit -> rendered)", ms < 10000, f"{ms:.0f} ms")
        check("top result is T 4.11.1", first.locator(".code").inner_text() == "T 4.11.1", first.locator(".code").inner_text())
        check("heading path shown", "Driver Egress" in first.locator(".path").inner_text(), first.locator(".path").inner_text())
        check("full original text shown", "exit to the side of the vehicle in less than 5 s" in first.locator(".body").inner_text())
        check("page reference shown", "39" in first.locator(".pages").inner_text(), first.locator(".pages").inner_text())
        check("applicability badge shown", "All vehicles" in first.locator(".badges").inner_text())
        first.locator(".copy").click()
        page.wait_for_timeout(200)
        clip = page.evaluate("navigator.clipboard.readText()")
        check("copy code button copies the code", clip == "T 4.11.1", clip)
        first.locator(".refs").click()
        page.wait_for_selector("#results .rule .extra .sub", timeout=5000)
        check("referenced rules expand with text", "T 13.3" in first.locator(".extra").inner_text())
        first.locator(".ctx").click()
        page.wait_for_selector("#results .rule .extra .sub.current", timeout=5000)
        check("section context loads", "T 4.11.1" in first.locator(".extra").inner_text())

        # exact code with spacing variant
        ms = search("t4.11.1")
        check("code lookup renders T 4.11.1", page.locator("#results .rule .code").first.inner_text() == "T 4.11.1", f"{ms:.0f} ms")
        check("code lookup labelled as exact", "Exact rule-code lookup" in page.locator("#meta").inner_text())

        # unknown code
        search("T 4.11.9")
        txt = page.locator("#results").inner_text()
        check("unknown code is reported, not substituted", "No such rule code" in txt and page.locator("#results .rule").count() == 0, txt[:120])

        # mixed applicability + cross-page rule
        search("T 6.3.1")
        body = page.locator("#results .rule").first
        check("cross-page rule complete", "electric brake system is actuated" in body.locator(".body").inner_text())
        check("EV-only provision labelled", body.locator(".tagchip.ev").count() == 1)
        check("pages 42 and 43 listed", "42" in body.locator(".pages").inner_text() and "43" in body.locator(".pages").inner_text())

        # table rendering
        search("A 1.2.8")
        check("table rendered as HTML with original-image toggle", page.locator("#results .asset table").count() == 1 and page.locator("#results .asset button").count() >= 1)

        # EV filter still includes general rules
        search("brake light", "ev")
        codes = [e.inner_text() for e in page.locator("#results .rule .code").all()]
        check("EV filter keeps general brake light rules", codes and codes[0].startswith("T 6.3"), str(codes[:4]))
        # CV filter hides EV-only rule
        search("maximum tractive system voltage", "cv")
        codes = [e.inner_text() for e in page.locator("#results .rule .code").all()]
        check("CV filter hides EV 4.1.1", "EV 4.1.1" not in codes, str(codes[:4]))

        # no match
        search("what is the size of the moon")
        check("no-match message", "No reliable match found" in page.locator("#results").inner_text())

        # stale response protection: fire two searches quickly, the second must win
        page.check("input[name=scope][value=all]")
        page.fill("#q", "accumulator container requirements")
        page.press("#q", "Enter")
        page.fill("#q", "T 2.2.1")
        page.press("#q", "Enter")
        page.wait_for_function("!document.querySelector('#meta').textContent.startsWith('Searching')", timeout=15000)
        page.wait_for_timeout(2500)  # give the older (slower) request time to finish; it must not overwrite
        codes = [e.inner_text() for e in page.locator("#results .rule .code").all()]
        check("older response never overwrites newer search", codes == ["T 2.2.1"], str(codes))

        # PDF pane
        page.locator("#results .rule .pages a").first.click()
        page.wait_for_timeout(500)
        check("PDF pane opens on page link", not page.locator("#pdfpane").is_hidden() and "page=21" in page.locator("#pdfframe").get_attribute("src"))

        # a few more latency samples (subsequent searches)
        for q in ["which colour is the brake light", "how many shutdown buttons", "restrictor diameter for E85", "BSPD", "penalty for knocking over a cone"]:
            search(q)
        lat = list(report["latencies_ms"].values())
        report["latency_summary_ms"] = {"n": len(lat), "median": round(statistics.median(lat), 1), "max": round(max(lat), 1)}
        check("all browser searches under 10 s", max(lat) < 10000, f"median {statistics.median(lat):.0f} ms, max {max(lat):.0f} ms")
        page.screenshot(path=str(OUT.with_name("browser_screenshot.png")), full_page=False)
        browser.close()
    report["all_ok"] = ok_all
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("ALL OK" if ok_all else "SOME CHECKS FAILED", "->", OUT)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
