"""End-to-end check of the static control room (docs/) in a real browser, with Playwright.

    pip install playwright && python -m playwright install chromium
    python tools/site_check.py                          # serves docs/ on a free local port
    python tools/site_check.py --url https://abderrahmanetb.github.io/chowkpilot/
    python tools/site_check.py --channel chrome         # use the installed Chrome instead of Playwright's Chromium

At desktop, tablet and phone widths it checks that the page loads its benchmark and replays, that the headline
numbers are filled in, and that nothing scrolls sideways; then it loads every replay and network file the page links
to, plays the twin for a few seconds, and fails on any console error, page exception or failed request.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import sys
import threading

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READY = ("() => document.getElementById('loadOurs').classList.contains('hidden') && "
         "[...document.querySelectorAll('#kpis .kpi .v')].every(e => e.textContent.trim() !== '…')")
LAYOUT = """() => ({
  overflow: document.documentElement.scrollWidth - innerWidth,
  kpis: [...document.querySelectorAll('#kpis .kpi .v')].map(e => e.textContent.trim()),
  bench: document.querySelectorAll('#benchDelay .sm').length,
  robust: document.querySelectorAll('#benchRobust .sm').length,
  rows: document.querySelectorAll('#benchTable tr').length,
})"""
ALL_DATA = """async () => {
  const list = await getData('scenarios.json'), bad = []; let n = 0;
  for (const s of list) {
    const files = s.replays.map(c => `replays/${s.id}__${c}.json.gz`);
    for (const [k, v] of Object.entries(s.conditions || {})) for (const c of v.controllers) files.push(`replays/${s.id}__${c}__${k}.json.gz`);
    files.push(`network/${s.id}.json.gz`);
    for (const f of files) {
      try { const d = await getData(f); if (f.startsWith('replays') && !(d.frames && d.frames.length > 100)) bad.push(`${f}: too few frames`); n++; }
      catch (e) { bad.push(`${f}: ${e.message}`); }
    }
  }
  return {checked: n, bad};
}"""


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def expected(url):
    """Off GitHub Pages the page probes for the optional live-simulation backend; a 404 there is normal."""
    return url.endswith("/api/healthz")


def serve_docs():
    handler = functools.partial(Quiet, directory=os.path.join(ROOT, "docs"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="check this deployment instead of serving docs/ locally")
    ap.add_argument("--channel", help="browser channel, e.g. chrome")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    srv, url = (None, args.url) if args.url else serve_docs()
    problems, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel=args.channel)
        for width, height, mobile in ((1440, 900, False), (860, 1000, False), (390, 844, True)):
            page = browser.new_page(viewport={"width": width, "height": height}, is_mobile=mobile)
            page.on("console", lambda m, w=width: m.type == "error" and not expected(m.location.get("url", ""))
                    and errors.append(f"{w}px console: {m.text} {m.location.get('url', '')}"))
            page.on("pageerror", lambda e, w=width: errors.append(f"{w}px exception: {e}"))
            page.on("requestfailed", lambda r, w=width: errors.append(f"{w}px request failed: {r.url} {r.failure}"))
            page.goto(url, wait_until="networkidle")
            page.wait_for_function(READY, timeout=60_000)
            lay = page.evaluate(LAYOUT)
            print(f"{width:5d}px  overflow {lay['overflow']}px  kpis {lay['kpis']}  bench {lay['bench']}  "
                  f"robustness {lay['robust']}  table rows {lay['rows']}")
            if lay["overflow"] > 0:
                problems.append(f"{width}px: page scrolls sideways by {lay['overflow']}px")
            if not lay["bench"] or not lay["robust"] or lay["rows"] < 2:
                problems.append(f"{width}px: benchmark panels missing")
            if width == 1440:
                data = page.evaluate(ALL_DATA)
                print(f"data files: {data['checked']} loaded, {len(data['bad'])} bad")
                problems += data["bad"]
                before = page.text_content("#clock")
                page.click("#btnPlay")
                page.wait_for_timeout(3000)
                after = page.text_content("#clock")
                print(f"playback: clock {before} -> {after}")
                if after == before:
                    problems.append("playback: the clock did not advance")
            page.close()
        browser.close()
    if srv:
        srv.shutdown()
    problems += errors
    for msg in problems:
        print("FAIL", msg)
    print("site check:", "failed" if problems else "passed")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
