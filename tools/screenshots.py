"""Capture the README screenshots of the control room with headless Chrome (DevTools protocol).

    python -m http.server 8090 --directory docs        # in another terminal
    python tools/screenshots.py [--base http://localhost:8090/] [--only twin,bench]

Writes docs/img/screen_<name>.png. Set CHROME to the browser binary if it is not found.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request

import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")
PORT = 9333
CANDIDATES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "google-chrome", "chromium", "chromium-browser"]

# name, url hash (deep link), viewport width, device scale, CSS selectors whose union is captured (None = first screen)
SHOTS = [
    ("twin", "#scenario=hyd_hitech&baseline=fixed&t=600&junction=0", 1440, 1, ["#twin"]),
    ("bench", "", 1440, 1, ["#bench"]),
    ("lab", "", 1440, 1, ["#lab"]),
    ("how", "", 1440, 1, ["#how"]),
    ("hero", "#scenario=hyd_hitech&baseline=fixed&t=600", 1440, 1, None),
    ("mobile", "#scenario=hyd_hitech&baseline=fixed&t=600", 390, 2, None),
]
READY = ("document.getElementById('loadOurs').classList.contains('hidden') && "
         "document.getElementById('kpis').querySelectorAll('.kpi').length > 0")


def find_chrome() -> str:
    for c in [os.environ.get("CHROME")] + CANDIDATES:
        if c and (os.path.isfile(c) or shutil.which(c)):
            return shutil.which(c) or c
    raise SystemExit("Chrome/Chromium not found; set CHROME=/path/to/chrome")


class CDP:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def __call__(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr):
        r = await self("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")


async def shoot(cdp, base, name, hsh, width, dpr, sels):
    first = 900 if width > 500 else 844
    await cdp("Emulation.setDeviceMetricsOverride", width=width, height=first, deviceScaleFactor=dpr, mobile=width < 500)
    await cdp("Page.navigate", url=f"{base}?shot={name}{hsh}")  # the query string forces a fresh load per shot
    t0 = time.time()
    while not await cdp.js(READY):
        if time.time() - t0 > 30:
            print(f"{name}: page not ready after 30 s, capturing anyway")
            break
        await asyncio.sleep(0.5)
    await asyncio.sleep(2.0)
    if sels is None:
        clip = {"x": 0, "y": 0, "width": width, "height": first}
    else:
        # make the viewport as tall as the page so the section renders without scrolling
        h = await cdp.js("document.documentElement.scrollHeight")
        await cdp("Emulation.setDeviceMetricsOverride", width=width, height=int(h), deviceScaleFactor=dpr, mobile=False)
        await asyncio.sleep(2.0)
        rs = json.loads(await cdp.js("JSON.stringify(%s.map(s => { const r = document.querySelector(s).getBoundingClientRect(); "
                                     "return [r.left, r.top + scrollY, r.right, r.bottom + scrollY]; }))" % json.dumps(sels)))
        pad = 16
        x0, y0 = max(0, min(r[0] for r in rs) - pad), max(0, min(r[1] for r in rs) - pad)
        x1, y1 = min(width, max(r[2] for r in rs) + pad), max(r[3] for r in rs) + pad
        clip = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
    shot = await cdp("Page.captureScreenshot", format="png", clip={**clip, "scale": 1})
    path = os.path.join(OUT, f"screen_{name}.png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(shot["data"]))
    print(f"{os.path.relpath(path, ROOT)}  {os.path.getsize(path) / 1e3:.0f} kB")


async def run(base, only):
    os.makedirs(OUT, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="chowk_chrome_")
    proc = subprocess.Popen([find_chrome(), "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--hide-scrollbars", "--disable-gpu", "--no-first-run", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=2))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("Chrome DevTools endpoint did not come up")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = CDP(ws)
            await cdp("Page.enable")
            await cdp("Runtime.enable")
            for s in SHOTS:
                if not only or s[0] in only:
                    await shoot(cdp, base, *s)
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8090/")
    ap.add_argument("--only", default="", help="comma-separated shot names")
    args = ap.parse_args()
    asyncio.run(run(args.base, set(args.only.split(",")) - {""}))


if __name__ == "__main__":
    main()
