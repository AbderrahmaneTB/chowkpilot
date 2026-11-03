"""Narrated walkthrough video of ChowkPilot (1920x1080, about 2.5 minutes).

    python -m http.server 8090 --directory docs        # in another terminal
    python tools/make_walkthrough.py                   # -> docs/img/chowkpilot_walkthrough.mp4
    python tools/make_walkthrough.py --preview DIR     # a few stills per scene, no TTS or encoding

Narration uses Kokoro, an offline neural voice with Apache-2.0 weights, when TTS_PYTHON points at a Python with
kokoro-onnx and KOKORO_DIR holds its model files (see tools/tts_kokoro.py); otherwise the built-in Windows voice.
Each sentence is synthesised separately, so captions and on-screen highlights change exactly when it starts.

The live scenes step the control room page frame by frame through the real recorded SUMO replays; the diagram and
deployment scenes are captured from the page; the problem, results and impact scenes are drawn here. Every number
comes from results/benchmark.json or the replay files.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import websockets
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screenshots import CDP, READY, find_chrome  # noqa: E402

IMG = os.path.join(ROOT, "docs", "img")
DATA = os.path.join(ROOT, "docs", "data")
OUT = os.path.join(IMG, "chowkpilot_walkthrough.mp4")
BASE = "http://localhost:8090/"
SCENARIO = "hyd_hitech"
W, H, FPS = 1920, 1080, 25
VIEW_H = 930          # scene area; the caption band sits below it
DISSOLVE = 10         # frames of cross-dissolve between scenes
VOICE = "Microsoft Zira Desktop"

# site tokens (docs/style.css)
BG, PANEL, LINE, LINE2 = (7, 11, 22), (15, 23, 43), (31, 43, 75), (42, 57, 98)
TEXT, MUTED, FAINT = (232, 237, 255), (141, 154, 196), (92, 106, 150)
OURS, OURS_TXT, MARK = (22, 163, 74), (74, 222, 128), (107, 122, 166)
SIG_RED, SIG_AMBER = (110, 34, 44), (202, 160, 40)
NAMES = {"fixed": "Fixed-time", "webster": "Webster*", "actuated": "Actuated", "maxpressure": "Max-pressure",
         "chowkpilot": "ChowkPilot"}

# ------------------------------------------------------------------ type and drawing helpers
FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
STYLES = {"display": ("bahnschrift.ttf", "Bold Condensed"), "display-semi": ("bahnschrift.ttf", "SemiBold SemiCondensed"),
          "body": ("segoeui.ttf", None), "semi": ("seguisb.ttf", None), "mono": ("CascadiaMono.ttf", "Regular")}
_fonts = {}


def font(size, style="body"):
    """Bahnschrift (a DIN face close to the site's Barlow Condensed) for display, Segoe UI for text."""
    if (size, style) not in _fonts:
        name, var = STYLES[style]
        f = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
        if var:
            f.set_variation_by_name(var)
        _fonts[size, style] = f
    return _fonts[size, style]


def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ramp(t, start, dur=0.5):
    return ease((t - start) / dur)


def mix(c, under, a):
    return tuple(int(u + (v - u) * a) for v, u in zip(c, under))


def wrap_px(text, f, width):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cur and f.getlength(trial) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def fit_into(img, box, resample=Image.LANCZOS):
    """Scale img to fit box (x0, y0, x1, y1); returns the scaled image, its paste position and the scale."""
    x0, y0, x1, y1 = box
    s = min((x1 - x0) / img.width, (y1 - y0) / img.height)
    im = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), resample)
    return im, (x0 + (x1 - x0 - im.width) // 2, y0 + (y1 - y0 - im.height) // 2), s


def spotlight(base, regions, pad=12, radius=18):
    """Dim everything except the regions (boxes in base pixels) and ring them in green."""
    if not regions:
        return base
    ov = Image.new("RGBA", base.size, BG + (178,))
    d = ImageDraw.Draw(ov)
    boxes = [[b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad] for b in regions]
    for b in boxes:
        d.rounded_rectangle(b, radius, fill=(0, 0, 0, 0))
    for b in boxes:
        d.rounded_rectangle(b, radius, outline=OURS_TXT + (255,), width=4)
    return Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")


def pct(v, display=False):
    """Signed percentage; Bahnschrift has no minus sign, so display type uses an en dash."""
    return f"{v:+.0f}%".replace("-", "\u2013" if display else "\u2212")


# ------------------------------------------------------------------ numbers
WORDS = {0: "none", 1: "one", 2: "two", 3: "three"}


def numbers():
    b = json.load(open(os.path.join(ROOT, "results", "benchmark.json"), encoding="utf-8"))
    s = b["summary"]
    rob = {c["label"]: c["values"]["chowkpilot"] for c in b["robustness"]["conditions"]}
    juncs = []
    for sid in b["test_scenarios"]:
        cell = b["cells"][sid]
        means = {c: cell[c]["mean_delay"]["mean"] for c in b["controllers"]}
        name, _, sub = b["titles"][sid].partition(": ")
        juncs.append({"name": name, "sub": sub, "means": means,
                      "cis": {c: cell[c]["mean_delay"]["ci"] for c in b["controllers"]},
                      "change": {c: 100 * (means["chowkpilot"] / means[c] - 1) for c in means if c != "chowkpilot"},
                      "paired": b["paired"][sid], "lowest": min(means, key=means.get)})
    replay = {c: json.load(gzip.open(os.path.join(DATA, "replays", f"{SCENARIO}__{c}.json.gz")))["mean_delay"]
              for c in ("fixed", "chowkpilot")}
    return {"fixed": -s["delay_vs_fixed"], "mp": -s["delay_vs_maxpressure"], "act": -s["delay_vs_actuated"],
            "amb": -s["ambulance_vs_fixed"], "co2": -s["co2_vs_fixed"], "seeds": b["seeds"],
            "noise": -next(v for k, v in rob.items() if k.lower().startswith("camera noise")),
            "vh": sum(v["vehicle_hours_saved_vs_fixed"] for v in b["impact"].values()),
            "sig_fixed": sum(j["paired"]["fixed"]["p_value"] < 0.05 for j in juncs),
            "webster_best": sum(j["lowest"] == "webster" for j in juncs),
            "juncs": juncs, "replay": replay, "controllers": b["controllers"]}


# ------------------------------------------------------------------ script
def S(say, show=""):
    """A narrated sentence: what the voice says, and the caption (defaults to the spoken text)."""
    return (say, (show or say).replace("Chowk Pilot", "ChowkPilot") if show is not None else None)


def segments(n):
    f = lambda x: f"{x:.0f}"  # noqa: E731
    sig = ("significant at all three junctions" if n["sig_fixed"] == 3
           else f"significant at {WORDS[n['sig_fixed']]} of the three junctions")
    results = [
        S(f"On three held-out Hyderabad areas, {n['seeds']} paired seeds each, we compared it with four baselines."),
        S(f"Chowk Pilot cut average delay by {f(n['fixed'])} percent versus fixed-time signals, {sig}.",
          f"Chowk Pilot cut average delay by {f(n['fixed'])}% versus fixed-time signals, {sig}."),
        S(f"It is {f(n['mp'])} percent lower than max-pressure, and level with vehicle-actuated control.",
          f"It is {f(n['mp'])}% lower than max-pressure, and level with vehicle-actuated control."),
    ]
    if n["webster_best"]:
        results.append(S(f"A Webster plan given each run's exact demand is still lowest at {WORDS[n['webster_best']]} "
                         "areas, and we report that openly."))
    rd = {k: f(v) for k, v in n["replay"].items()}
    return [
        {"name": "coldopen", "chapter": "Hitech City, Hyderabad: recorded SUMO simulation at peak hour",
         "live": (440, 560), "lines": [
             S("Same junction. Same vehicles. Same minute of rush hour."),
             S("On the left, the fixed-time signal plan most junctions run today. On the right, Chowk Pilot.",
               "Left: the fixed-time signal plan most junctions run today. Right: ChowkPilot.")]},
        {"name": "title", "full": True, "tail": 1.4, "lines": [
            S("Chowk Pilot: one A.I. signal brain for every chowk.", None)]},
        {"name": "problem", "chapter": (1, "The problem"), "lines": [
            S("Hyderabad drivers lose 123 hours a year to rush-hour traffic."),
            S("Nearly three in four vehicles in Telangana are two-wheelers, in mixed, lane-less traffic."),
            S("Yet most junctions still run fixed-time plans, blind to the queues in front of them.")]},
        {"name": "how", "chapter": (2, "How it works"), "lines": [
            S("Chowk Pilot is a reinforcement-learning controller. It sees only what a roadside camera can count: "
              "queues, flow, waiting, and ambulances."),
            S("A small shared network scores each movement and sums the scores per phase, so one set of weights "
              "runs any junction."),
            S("It starts from max-pressure, a classical rule with a proven guarantee, and learns a correction with "
              "deep reinforcement learning."),
            S("A fixed safety layer enforces minimum and maximum green, clearance, and an ambulance green corridor."),
            S("Trained on thirteen layouts, it was tested on three Hyderabad areas it had never seen.",
              "Trained on 13 layouts (11 synthetic, 2 real), it was tested on three Hyderabad areas it had never "
              "seen.")]},
        {"name": "twin", "chapter": (3, "The control room"), "live": (330, 600), "lines": [
            S("In the live control room on our website, both views replay the same recorded simulation of Hitech "
              "City, with exactly the same vehicles."),
            S("Stopped vehicles turn red, and the counters track the queues."),
            S(f"In this run, mean delay fell from {rd['fixed']} seconds to {rd['chowkpilot']}.",
              f"In this run, mean delay per vehicle fell from {rd['fixed']} s to {rd['chowkpilot']} s.")]},
        {"name": "explain", "chapter": (3, "The control room"), "live": (600, 700), "lines": [
            S("Every decision is explained: the learned score for each phase, the stopped vehicles it would release, "
              "how long they waited, and the max-pressure prior.")]},
        {"name": "results", "chapter": (4, "Results on unseen junctions"), "lines": results},
        {"name": "impact", "chapter": (5, "Impact"), "lines": [
            S(f"Ambulance delay fell {f(n['amb'])} percent, and carbon dioxide {f(n['co2'])} percent.",
              f"Ambulance delay fell {f(n['amb'])}%, and CO\u2082 {f(n['co2'])}%."),
            S(f"With twenty percent camera counting error, delay was still {f(n['noise'])} percent below fixed-time.",
              f"With 20% camera counting error, delay was still {f(n['noise'])}% below fixed-time."),
            S(f"One half-hour peak at the three junctions saves about {f(n['vh'])} vehicle-hours of delay.")]},
        {"name": "deploy", "chapter": (6, "Path to deployment"), "lines": [
            S("A decision takes under a millisecond on a laptop processor, so it fits a roadside box."),
            S("Next: a calibrated twin of one corridor, a shadow-mode trial with traffic engineers, then a "
              "supervised pilot at three to five junctions.",
              "Next: a calibrated twin of one corridor, a shadow-mode trial with traffic engineers, then a "
              "supervised pilot at 3\u20135 junctions.")]},
        {"name": "end", "full": True, "tail": 3.0, "lines": [
            S("Everything is open source. Try the live control room at the link on screen.", None)]},
    ]


# ------------------------------------------------------------------ narration
def synthesize(texts):
    """One wav per sentence, cached by text and voice; returns (int16 arrays, silence trimmed), sample rate, credit."""
    py, kdir = os.environ.get("TTS_PYTHON"), os.environ.get("KOKORO_DIR")
    kokoro = bool(py and kdir)
    key = (f"kokoro {os.environ.get('KOKORO_VOICE', 'af_heart')} {os.environ.get('KOKORO_SPEED', '1.0')}"
           if kokoro else VOICE)
    cache = os.environ.get("WALK_TTS_CACHE", os.path.join(tempfile.gettempdir(), "chowk_tts_cache"))
    os.makedirs(cache, exist_ok=True)
    jobs = [{"text": t, "path": os.path.join(cache, hashlib.sha1(f"{key}|{t}".encode()).hexdigest()[:16] + ".wav")}
            for t in texts]
    todo = [j for j in jobs if not os.path.exists(j["path"])]
    if todo:
        jpath = os.path.join(cache, "jobs.json")
        json.dump(todo, open(jpath, "w", encoding="utf-8"))
        if kokoro:
            subprocess.run([py, os.path.join(ROOT, "tools", "tts_kokoro.py"), jpath], check=True)
        else:
            ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.SelectVoice('{VOICE}'); $s.Rate = 0; "
                  f"foreach ($j in (Get-Content -Raw -Encoding UTF8 '{jpath}' | ConvertFrom-Json)) "
                  "{ $s.SetOutputToWaveFile($j.path); $s.Speak($j.text) }; $s.SetOutputToNull(); $s.Dispose()")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    credit = "Narration: synthetic voice (Kokoro TTS, Apache-2.0)" if kokoro else "Narration: synthetic voice (Windows)"
    clips, sr = [], None
    for j in jobs:
        with wave.open(j["path"]) as w:
            sr = sr or w.getframerate()
            assert w.getframerate() == sr and w.getnchannels() == 1 and w.getsampwidth() == 2
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        loud = np.flatnonzero(np.abs(a.astype(np.int32)) > 0.02 * np.abs(a).max())
        m = int(0.04 * sr)
        clips.append(a[max(0, loud[0] - m): loud[-1] + m])
    return clips, sr, credit


def timeline(segs, durs):
    """Per scene: sentence (start, end) times in seconds and the frame count; sentences sit in a lead/gap/tail frame."""
    out = []
    it = iter(durs)
    for i, seg in enumerate(segs):
        t, spans = (0.25 if i == 0 else 0.5), []
        for _ in seg["lines"]:
            d = next(it)
            spans.append((t, t + d))
            t += d + 0.32
        t += seg.get("tail", 0.6) - 0.32
        out.append((spans, int(round(t * FPS))))
    return out


# ------------------------------------------------------------------ page capture
RECTS = ("JSON.stringify((%s).map(el => { const r = el.getBoundingClientRect(); "
         "return [r.left + scrollX, r.top + scrollY, r.right + scrollX, r.bottom + scrollY]; }))")
VIEWS = "[document.getElementById('cvBase').parentElement, document.getElementById('cvOurs').parentElement]"
ZOOM = "fitCamera(); state.cam.scale *= %s; Object.values(views).forEach(v => (v.dirty = true)); render(); 1"


def jobs_for(segs, plan, preview=False):
    by = {s["name"]: (s, p) for s, p in zip(segs, plan)}
    jobs = []
    for name, clips, subs, dsf, setup, extra in [
        ("coldopen", [VIEWS], {}, 1.5, ZOOM % 1.5, [0]),
        ("twin", ["[document.querySelector('.controls'), document.getElementById('runSummary')]"],
         {"hud": "[document.getElementById('hudBase'), document.getElementById('hudOurs')]",
          "summary": "[document.getElementById('runSummary')]"}, 1.5, None, [0]),
        ("explain", ["[document.getElementById('cvOurs').parentElement]",
                     "[document.querySelector('.below > .card:last-child')]"], {}, 2, ZOOM % 1.3, [0, 0]),
    ]:
        seg, (_, nf) = by[name]
        jobs.append({"name": name, "clips": clips, "subs": subs, "dsf": dsf, "setup": setup, "extra": extra,
                     "t0": seg["live"][0], "t1": seg["live"][1], "n": min(nf, 8) if preview else nf})
    jobs.append({"name": "diagram", "clips": ["[document.querySelector('#how svg')]"], "subs": {}, "dsf": 3,
                 "extra": [0]})
    return jobs


async def capture(jobs, folder):
    """Fill each job with 'frames' (lists of image paths per clip, per frame) and 'subs' (boxes in clip pixels)."""
    prof = tempfile.mkdtemp(prefix="chowk_walk_")
    proc = subprocess.Popen([find_chrome(), "--headless=new", "--remote-debugging-port=9336", f"--user-data-dir={prof}",
                             "--hide-scrollbars", "--no-first-run", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                page = next(t for t in json.load(urllib.request.urlopen("http://127.0.0.1:9336/json/list", timeout=2))
                            if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("Chrome DevTools endpoint did not come up")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = CDP(ws)
            await cdp("Page.enable")
            for job in jobs:
                dsf = job["dsf"]
                # tall enough that the explanation card (about y = 1300-1650) is inside the viewport for live frames
                await cdp("Emulation.setDeviceMetricsOverride", width=1440, height=2400, deviceScaleFactor=dsf,
                          mobile=False)
                await cdp("Page.navigate", url=f"{BASE}?walk={job['name']}#scenario={SCENARIO}&baseline=fixed"
                                               f"&t={job.get('t0', 600)}&junction=0")
                t_start = time.time()
                while not await cdp.js(READY) and time.time() - t_start < 40:
                    await asyncio.sleep(0.5)
                await asyncio.sleep(2.0)
                if job.get("setup"):
                    await cdp.js(job["setup"])
                boxes = []
                for expr, extra in zip(job["clips"], job["extra"]):
                    rs = json.loads(await cdp.js(RECTS % expr))
                    boxes.append([min(r[0] for r in rs) - 8, min(r[1] for r in rs) - 8,
                                  max(r[2] for r in rs) + 8, max(r[3] for r in rs) + 8 + extra])
                x0, y0 = boxes[0][:2]
                job["subs"] = {k: [[(r[0] - x0) * dsf, (r[1] - y0) * dsf, (r[2] - x0) * dsf, (r[3] - y0) * dsf]
                                   for r in json.loads(await cdp.js(RECTS % e))] for k, e in job["subs"].items()}
                live = "t0" in job
                n = job["n"] if live else 1
                job["frames"] = []
                for i in range(n):
                    if live:
                        t = job["t0"] + (job["t1"] - job["t0"]) * i / max(1, n - 1)
                        await cdp.js(f"state.t = {t:.2f}; lastPanel = 0; render(); 1")
                    paths = []
                    for c, b in enumerate(boxes):
                        clip = {"x": b[0], "y": b[1], "width": b[2] - b[0], "height": b[3] - b[1], "scale": 1}
                        shot = await cdp("Page.captureScreenshot", format="jpeg" if live else "png", quality=93,
                                         clip=clip, captureBeyondViewport=not live)
                        p = os.path.join(folder, f"{job['name']}_{i:04d}_{c}.{'jpg' if live else 'png'}")
                        with open(p, "wb") as fh:
                            fh.write(base64.b64decode(shot["data"]))
                        paths.append(p)
                    job["frames"].append(paths)
                print(f"captured {job['name']}: {n} frame(s)", flush=True)
    finally:
        proc.terminate()
        shutil.rmtree(prof, ignore_errors=True)
    return {j["name"]: j for j in jobs}


# ------------------------------------------------------------------ scenes
class Ctx:
    def __init__(self, k, n, spans):
        self.k, self.n, self.t = k, n, k / FPS
        self.starts, self.ends = [s for s, _ in spans], [e for _, e in spans]
        self.si = max([i for i, s in enumerate(self.starts) if self.t >= s - 0.15] or [0])

    def since(self, i, delay=0.0, dur=0.5):
        return ramp(self.t, self.starts[i] - 0.15 + delay, dur) if i < len(self.starts) else 0.0


def live_frame(A, name, ctx):
    """Captured frame for this video frame (previews capture only a few, spread over the scene)."""
    fr = A[name]["frames"]
    return fr[min(len(fr) - 1, round(ctx.k * (len(fr) - 1) / max(1, ctx.n - 1)))]


def backdrop():
    """Faint street network of the Secunderabad area behind the title and end cards."""
    g = json.load(gzip.open(os.path.join(DATA, "network", "hyd_secunderabad.json.gz")))
    x0, y0, x1, y1 = g["bounds"]
    s = min(W * 0.66 / (x1 - x0), H * 0.96 / (y1 - y0))
    ox, oy = W - 40 - (x1 - x0) * s, (H - (y1 - y0) * s) / 2
    pt = lambda x, y: (ox + (x - x0) * s, oy + (y1 - y) * s)  # noqa: E731
    net = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(net)
    for poly in g["junctions"]:
        if len(poly) > 2:
            d.polygon([pt(x, y) for x, y in poly], fill=(26, 38, 70))
    for lane in g["lanes"]:
        d.line([pt(x, y) for x, y in lane["s"]], fill=LINE2, width=max(1, round(lane["w"] * s * 0.7)))
    for sg in g["signals"]:
        cx, cy = pt(sg["x"], sg["y"])
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=OURS)
    ramp_x = np.clip((np.arange(W) - 0.42 * W) / (0.3 * W), 0, 1) * 0.6
    mask = Image.fromarray((np.tile(ramp_x, (H, 1)) * 255).astype(np.uint8), "L")
    return Image.composite(net, Image.new("RGB", (W, H), BG), mask)


def scene_title(ctx, A):
    img = A["backdrop"].copy()
    d = ImageDraw.Draw(img)
    a1, a2, a3 = ramp(ctx.t, 0.1, 0.8), ramp(ctx.t, 0.7, 0.6), ramp(ctx.t, 1.2, 0.6)
    d.text((150, 250 + int(24 * (1 - a1))), "ChowkPilot", font=font(210, "display"), fill=mix(TEXT, BG, a1))
    d.text((156, 500), "One AI signal brain for every chowk.", font=font(58, "semi"), fill=mix(OURS_TXT, BG, a2))
    d.text((156, 590), "A learned traffic-signal controller for India's mixed, lane-less traffic,",
           font=font(34), fill=mix(MUTED, BG, a2))
    d.text((156, 636), "tested on real Hyderabad junctions it never saw in training.", font=font(34),
           fill=mix(MUTED, BG, a2))
    d.text((156, 760), "chowk (Hindi\u2013Urdu): a crossroads, a junction", font=font(26), fill=mix(FAINT, BG, a3))
    d.text((156, H - 130), "Team Blackwater", font=font(30, "semi"), fill=mix(TEXT, BG, a3))
    d.text((156, H - 88), "NeuralStorm 2026, problem FT-01 Smart Traffic Signal Optimisation", font=font(26),
           fill=mix(MUTED, BG, a3))
    return img


def scene_end(ctx, A):
    img = A["backdrop"].copy()
    d = ImageDraw.Draw(img)
    a1, a2, a3 = ramp(ctx.t, 0.1, 0.6), ramp(ctx.t, 0.6, 0.6), ramp(ctx.t, 1.1, 0.6)
    d.text((150, 230), "Try the live control room", font=font(110, "display"), fill=mix(TEXT, BG, a1))
    d.text((154, 370), "abderrahmanetb.github.io/chowkpilot", font=font(66, "display-semi"), fill=mix(OURS_TXT, BG, a1))
    d.text((156, 480), "Code, benchmark and model: github.com/AbderrahmaneTB/chowkpilot", font=font(36),
           fill=mix(TEXT, BG, a2))
    d.text((156, 620), "Team Blackwater", font=font(34, "semi"), fill=mix(TEXT, BG, a3))
    d.text((156, 668), "Abderrahmane Tayeb Boudjema and Ouail Imad Eddine Zitouni", font=font(34),
           fill=mix(MUTED, BG, a3))
    d.text((156, H - 96), f"Simulation: Eclipse SUMO. Map data \u00a9 OpenStreetMap contributors (ODbL). {A['credit']}.",
           font=font(24), fill=mix(FAINT, BG, a3))
    return img


def scene_coldopen(ctx, A):
    img = Image.new("RGB", (W, VIEW_H), BG)
    fr = Image.open(live_frame(A, "coldopen", ctx)[0]).convert("RGB")
    im, pos, _ = fit_into(fr, (60, 150, W - 60, VIEW_H - 20))
    img.paste(im, pos)
    a = ramp(ctx.t, 0.15, 0.7)
    ImageDraw.Draw(img).text((64, 34), "Same vehicles, same minute, two signal brains.", font=font(84, "display"),
                             fill=mix(TEXT, BG, a))
    return img


PROBLEM = [("123 h", "lost per driver each year to rush-hour traffic in Hyderabad", "TomTom Traffic Index 2026"),
           ("73%", "of Telangana's vehicles are two-wheelers, in mixed, lane-less traffic",
            "Telangana Transport Department, 2025"),
           ("17+ min", "average ambulance response time in Delhi, rising since 2014", "The Hindu, 2025")]


def scene_problem(ctx, A):
    img = Image.new("RGB", (W, VIEW_H), BG)
    d = ImageDraw.Draw(img)
    for i, (value, label, src) in enumerate(PROBLEM):
        a = ctx.since(min(i, 1), delay=1.2 * (i == 2))  # the ambulance figure joins the second sentence
        if a <= 0:
            continue
        x, y = 120 + i * 580, 70 + int(20 * (1 - a))
        d.text((x - 4, y), value, font=font(170, "display"), fill=mix(TEXT, BG, a))
        yy = y + 230
        for ln in wrap_px(label, font(32), 500):
            d.text((x, yy), ln, font=font(32), fill=mix(MUTED, BG, a))
            yy += 44
        d.text((x, yy + 12), src, font=font(22), fill=mix(FAINT, BG, a))
    a = ctx.since(2)
    if a > 0:  # a fixed-time plan drawn as a signal timing diagram: 4 approaches, one rigid cycle
        x0, x1, y0, cyc = 300, W - 120, 640, 120.0
        sx = (x1 - x0) / (3 * cyc)
        d.text((120, y0 - 62), "Illustrative fixed-time plan: each approach gets the same green, every cycle",
               font=font(26, "semi"), fill=mix(MUTED, BG, a))
        for r, name in enumerate(["North", "East", "South", "West"]):
            y = y0 + r * 56
            d.text((120, y + 2), name, font=font(26), fill=mix(MUTED, BG, a))
            d.rectangle([x0, y, x1, y + 32], fill=mix(SIG_RED, BG, a))
            for c in range(3):
                g0 = c * cyc + r * 30
                d.rectangle([x0 + g0 * sx, y, x0 + (g0 + 26) * sx, y + 32], fill=mix(OURS, BG, a))
                d.rectangle([x0 + (g0 + 26) * sx, y, x0 + (g0 + 29) * sx, y + 32], fill=mix(SIG_AMBER, BG, a))
        ph = x0 + ((ctx.t - ctx.starts[2]) * 45 % (3 * cyc)) * sx
        d.line([(ph, y0 - 10), (ph, y0 + 3 * 56 + 42)], fill=mix(TEXT, BG, a), width=3)
    return img


SVG_BOX = {"camera": (10, 20, 160, 90), "movements": (10, 130, 160, 200), "mask": (10, 240, 160, 310),
           "encoder": (235, 120, 395, 210), "sum": (460, 120, 580, 210), "score": (620, 120, 710, 210),
           "safety": (570, 242, 710, 334), "shared": (244, 28, 600, 92)}
HOW = [("1  Sense like a camera", ["camera", "movements"]),
       ("2  Score every phase", ["encoder", "sum", "mask", "shared"]),
       ("2  Score every phase", ["sum", "score"]),
       ("3  Stay safe and fair", ["safety"]),
       ("4  Generalise to junctions it has never seen", [])]


def prep_how(A):
    src = Image.open(A["diagram"]["frames"][0][0]).convert("RGB")
    im, pos, f = fit_into(src, (120, 150, W - 120, VIEW_H - 24))
    s = min(src.width / 720, src.height / 336)
    ox, oy = (src.width - 720 * s) / 2, (src.height - 336 * s) / 2
    box = lambda b: [(ox + b[0] * s) * f, (oy + b[1] * s) * f, (ox + b[2] * s) * f, (oy + b[3] * s) * f]  # noqa
    A["how"] = (pos, [spotlight(im, [box(SVG_BOX[k]) for k in keys]) for _, keys in HOW])


def blend_states(ctx, states):
    cur = states[min(ctx.si, len(states) - 1)]
    if ctx.si == 0:
        return cur
    return Image.blend(states[ctx.si - 1], cur, ctx.since(ctx.si, dur=0.45))


def scene_how(ctx, A):
    img = Image.new("RGB", (W, VIEW_H), BG)
    pos, states = A["how"]
    img.paste(blend_states(ctx, states), pos)
    d = ImageDraw.Draw(img)
    a = ctx.since(ctx.si, dur=0.4) if ctx.si else 1.0
    label = HOW[ctx.si][0]
    if ctx.si and HOW[ctx.si - 1][0] != label:
        d.text((120, 52), HOW[ctx.si - 1][0], font=font(54, "display-semi"), fill=mix(TEXT, BG, 1 - a))
        d.text((120, 52), label, font=font(54, "display-semi"), fill=mix(TEXT, BG, a))
    else:
        d.text((120, 52), label, font=font(54, "display-semi"), fill=TEXT)
    return img


def scene_twin(ctx, A):
    img = Image.new("RGB", (W, VIEW_H), BG)
    job = A["twin"]
    fr = Image.open(live_frame(A, "twin", ctx)[0]).convert("RGB")
    im, pos, s = fit_into(fr, (40, 24, W - 40, VIEW_H - 24))
    img.paste(im, pos)
    d = ImageDraw.Draw(img)
    for key, si in (("hud", 1), ("summary", 2)):
        a = ctx.since(si) if ctx.si == si else 0.0
        for b in job["subs"][key] if a > 0 else []:
            d.rounded_rectangle([pos[0] + b[0] * s - 6, pos[1] + b[1] * s - 6, pos[0] + b[2] * s + 6,
                                 pos[1] + b[3] * s + 6], 12, outline=mix(OURS_TXT, BG, a), width=4)
    return img


def scene_explain(ctx, A):
    img = Image.new("RGB", (W, VIEW_H), BG)
    view, panel = (Image.open(p).convert("RGB") for p in live_frame(A, "explain", ctx))
    for src, box in ((view, (40, 40, 820, VIEW_H - 40)), (panel, (860, 30, W - 40, VIEW_H - 30))):
        im, pos, _ = fit_into(src, box)
        img.paste(im, pos)
    return img


def emphasis(si, c, n_lines):
    sets = [None, {"fixed", "chowkpilot"}, {"maxpressure", "actuated", "chowkpilot"}, {"webster", "chowkpilot"}]
    s = sets[min(si, len(sets) - 1)] if si < n_lines else None
    return 1.0 if s is None or c in s else 0.3


def scene_results(ctx, A):
    n = A["n"]
    img = Image.new("RGB", (W, VIEW_H), BG)
    d = ImageDraw.Draw(img)
    d.text((120, 36), "Mean delay per vehicle at three junctions it never saw", font=font(56, "display-semi"), fill=TEXT)
    d.text((122, 106), f"Seconds, lower is better. {n['seeds']} paired seeds per junction; whiskers are 95% confidence "
                       "intervals.", font=font(26), fill=MUTED)
    pw, gap, top, bot = 544, 24, 170, VIEW_H - 64
    nl = len(ctx.starts)
    for p, j in enumerate(n["juncs"]):
        x = 120 + p * (pw + gap)
        d.rounded_rectangle([x, top, x + pw, bot], 18, fill=PANEL, outline=LINE, width=2)
        d.text((x + 28, top + 22), j["name"], font=font(32, "semi"), fill=TEXT)
        d.text((x + 28, top + 64), j["sub"] or " ", font=font(22), fill=FAINT)
        vmax = max(j["means"][c] + j["cis"][c] for c in n["controllers"])
        L = pw - 56
        for r, c in enumerate(n["controllers"]):
            prev = emphasis(ctx.si - 1, c, nl) if ctx.si else 1.0
            em = prev + (emphasis(ctx.si, c, nl) - prev) * (ctx.since(ctx.si, dur=0.4) if ctx.si else 1.0)
            grow = ctx.since(0, delay=0.25 + 0.15 * p + 0.06 * r, dur=1.1)
            ry = top + 116 + r * 74
            ours = c == "chowkpilot"
            d.text((x + 28, ry), NAMES[c], font=font(25, "semi" if ours else "body"),
                   fill=mix(TEXT if ours else MUTED, PANEL, em))
            val = f"{j['means'][c]:.0f} s"
            d.text((x + pw - 28 - font(24, "mono").getlength(val), ry + 2), val, font=font(24, "mono"),
                   fill=mix(TEXT, PANEL, em * grow))
            bl = L * j["means"][c] / vmax * grow
            if bl > 2:
                d.rounded_rectangle([x + 28, ry + 38, x + 28 + bl, ry + 58], 4,
                                    fill=mix(OURS if ours else MARK, PANEL, em))
                lo, hi = (x + 28 + L * max(0.0, j["means"][c] + k * j["cis"][c]) / vmax for k in (-1, 1))
                wc = mix(TEXT, PANEL, 0.75 * em * grow)
                d.line([(lo, ry + 48), (hi, ry + 48)], fill=wc, width=2)
                d.line([(lo, ry + 42), (lo, ry + 54)], fill=wc, width=2)
                d.line([(hi, ry + 42), (hi, ry + 54)], fill=wc, width=2)
            if c == j["lowest"] and ctx.si == 3:
                tag = "lowest here"
                tx = x + 28 + font(25, "semi" if ours else "body").getlength(NAMES[c]) + 14
                d.rounded_rectangle([tx, ry + 2, tx + font(20, "semi").getlength(tag) + 20, ry + 32], 8,
                                    outline=mix(TEXT, PANEL, ctx.since(3)), width=2)
                d.text((tx + 10, ry + 4), tag, font=font(20, "semi"), fill=mix(TEXT, PANEL, ctx.since(3)))
        by = top + 116 + 5 * 74 + 6
        lines = []
        if ctx.si == 1:
            pf = j["paired"]["fixed"]
            lines = [(f"{pct(j['change']['fixed'])} vs fixed-time", OURS_TXT),
                     (f"lower in {pf['wins']} of {pf['n']} paired seeds", MUTED)]
        elif ctx.si == 2:
            for c in ("maxpressure", "actuated"):
                ns = "" if j["paired"][c]["p_value"] < 0.05 else " (not significant)"
                lines.append((f"{pct(j['change'][c])} vs {NAMES[c].lower()}{ns}", TEXT if not ns else MUTED))
        a = ctx.since(ctx.si, dur=0.4)
        for i, (txt, col) in enumerate(lines):
            d.text((x + 28, by + i * 40), txt, font=font(28, "semi" if i == 0 else "body"), fill=mix(col, PANEL, a))
    d.text((120, VIEW_H - 46), "* Webster: a fixed plan re-timed with each run's exact demand, which a deployed signal "
                               "would not know in advance.", font=font(22), fill=FAINT)
    return img


def scene_impact(ctx, A):
    n = A["n"]
    tiles = [(pct(-n["amb"], True), "ambulance delay with the green corridor", 0),
             (pct(-n["co2"], True), "CO\u2082 emissions", 0),
             (pct(-n["noise"], True), "delay with 20% camera counting error and 10% of vehicles missed", 1),
             (f"{n['vh']:.0f} h", "vehicle-hours of delay saved in one simulated half-hour peak, three junctions", 2)]
    img = Image.new("RGB", (W, VIEW_H), BG)
    d = ImageDraw.Draw(img)
    for i, (value, label, si) in enumerate(tiles):
        a = ctx.since(si, delay=0.35 * (i == 1))
        if a <= 0:
            continue
        x, y = 120 + i * 430, 200 + int(20 * (1 - a))
        d.text((x - 4, y), value, font=font(150, "display"), fill=mix(OURS_TXT, BG, a))
        yy = y + 200
        for ln in wrap_px(label, font(30), 370):
            d.text((x, yy), ln, font=font(30), fill=mix(MUTED, BG, a))
            yy += 42
    d.text((120, VIEW_H - 76), "All compared with the standard fixed-time plan and averaged over the three unseen "
                               "junctions. The camera test uses 5 seeds per junction.", font=font(24), fill=FAINT)
    return img


def scene_deploy(ctx, A):
    """The deployment section of the site, set at video size: what already fits, then the staged pilot."""
    fits = [("Runs on a roadside box", "About 0.16 ms per decision on a laptop CPU. No GPU at inference."),
            ("Camera counts are enough", f"With 20% counting error and 10% of vehicles missed, delay still fell "
                                         f"{A['n']['noise']:.0f}% vs fixed-time."),
            ("Safety stays outside the network", "If the cameras fail, the junction reverts to its fixed plan.")]
    steps = [("Calibrated twin", "2–4 weeks", "One corridor from OpenStreetMap and a site survey, with demand "
                                                    "from real counts. Re-run this benchmark on it."),
             ("Shadow mode", "about 4 weeks", "Reads live counts and logs what it would do next to what the signal "
                                              "did. Engineers review the explanations."),
             ("Supervised pilot", "3–5 junctions", "For example the Secunderabad corridor modelled here. An "
                                                        "operator can switch back to the fixed plan at any time."),
             ("Scale by corridor", "after the pilot", "Still to build: camera counting, pedestrian phases, the "
                                                     "controller interface and approvals.")]
    img = Image.new("RGB", (W, VIEW_H), BG)
    d = ImageDraw.Draw(img)
    d.text((120, 40), "From simulation to a Hyderabad junction", font=font(56, "display-semi"), fill=TEXT)
    dim = 1.0 - 0.55 * ctx.since(1, dur=0.5)
    for i, (title, body) in enumerate(fits):
        shown = ctx.since(0, delay=0.35 * i)
        if shown <= 0:
            continue
        a = shown * dim
        x, y = 120 + i * 570, 140
        d.rounded_rectangle([x, y, x + 540, y + 230], 18, fill=mix(PANEL, BG, shown), outline=mix(LINE2, BG, a),
                            width=2)
        d.text((x + 30, y + 26), title, font=font(32, "semi"), fill=mix(TEXT, PANEL, a))
        yy = y + 80
        for ln in wrap_px(body, font(27), 480):
            d.text((x + 30, yy), ln, font=font(27), fill=mix(MUTED, PANEL, a))
            yy += 38
    if len(ctx.starts) > 1:
        s0, s1 = ctx.starts[1], ctx.ends[1]
        y = 470
        for i, (title, tag, body) in enumerate(steps):  # one step per quarter of the sentence
            a = ramp(ctx.t, s0 - 0.15 + (s1 - s0) * i / len(steps), 0.5)
            x = 120 + i * 430
            if i:
                d.line([(x - 404 + 60, y + 28), (x - 8, y + 28)], fill=mix(LINE2, BG, a), width=3)
            d.ellipse([x, y, x + 56, y + 56], fill=mix(OURS, BG, a))
            num = str(i + 1)
            d.text((x + 28 - font(32, "display-semi").getlength(num) / 2, y + 9), num, font=font(32, "display-semi"),
                   fill=mix(TEXT, BG, a))
            d.text((x, y + 84), title, font=font(34, "semi"), fill=mix(TEXT, BG, a))
            d.text((x, y + 130), tag, font=font(26, "semi"), fill=mix(OURS_TXT, BG, a))
            yy = y + 178
            for ln in wrap_px(body, font(26), 380):
                d.text((x, yy), ln, font=font(26), fill=mix(MUTED, BG, a))
                yy += 36
    return img


SCENES = {"coldopen": scene_coldopen, "title": scene_title, "problem": scene_problem, "how": scene_how,
          "twin": scene_twin, "explain": scene_explain, "results": scene_results, "impact": scene_impact,
          "deploy": scene_deploy, "end": scene_end}


def band(img, chapter, caption):
    d = ImageDraw.Draw(img)
    d.rectangle([0, VIEW_H, W, H], fill=BG)
    d.line([(0, VIEW_H), (W, VIEW_H)], fill=LINE, width=2)
    x = 80
    if isinstance(chapter, tuple):
        d.text((x, VIEW_H + 16), f"{chapter[0]}/6", font=font(24, "display-semi"), fill=OURS_TXT)
        d.text((x + 50, VIEW_H + 16), chapter[1], font=font(24, "display-semi"), fill=MUTED)
    elif chapter:
        d.text((x, VIEW_H + 16), chapter, font=font(24, "display-semi"), fill=MUTED)
    if caption:
        for size in (36, 32, 28):
            lines = wrap_px(caption, font(size, "semi"), W - 2 * x)
            if len(lines) <= 2:
                break
        for i, ln in enumerate(lines[:2]):
            d.text((x, VIEW_H + 52 + i * int(size * 1.3)), ln, font=font(size, "semi"), fill=TEXT)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="write a few stills per scene to this folder instead of the video")
    args = ap.parse_args()
    n = numbers()
    segs = segments(n)
    lines = [ln for s in segs for ln in s["lines"]]
    tmp = tempfile.mkdtemp(prefix="chowk_walk_")
    clips, sr, credit = synthesize([say for say, _ in lines])
    durs = [len(c) / sr for c in clips]
    plan = timeline(segs, durs)
    total = sum(nf for _, nf in plan)
    print(f"{len(segs)} scenes, {len(lines)} sentences, {total / FPS:.1f} s", flush=True)

    A = asyncio.run(capture(jobs_for(segs, plan, bool(args.preview)), tmp))
    A.update(n=n, credit=credit, backdrop=backdrop())
    prep_how(A)

    if not args.preview:
        silent = os.path.join(tmp, "video.mp4")
        writer = imageio.get_writer(silent, fps=FPS, codec="libx264", quality=None, macro_block_size=8,
                                    ffmpeg_params=["-crf", "20", "-preset", "medium"])
    else:
        os.makedirs(args.preview, exist_ok=True)
    prev = Image.new("RGB", (W, H), BG)
    it = iter(lines)
    for seg, (spans, nf) in zip(segs, plan):
        seg_lines = [next(it) for _ in seg["lines"]]
        keep = set() if not args.preview else {int(s * FPS) + 12 for s, _ in spans} | {nf - 1}
        last = None
        for k in range(nf):
            if args.preview and k not in keep and k >= DISSOLVE:
                continue
            ctx = Ctx(k, nf, spans)
            scene = SCENES[seg["name"]](ctx, A)
            if seg.get("full"):
                frame = scene
            else:
                frame = Image.new("RGB", (W, H), BG)
                frame.paste(scene, (0, 0))
                band(frame, seg.get("chapter"), seg_lines[ctx.si][1])
            last = frame
            if k < DISSOLVE:
                frame = Image.blend(prev, frame, ease((k + 1) / DISSOLVE))
            if args.preview:
                if k in keep:
                    frame.save(os.path.join(args.preview, f"{seg['name']}_{k:04d}.png"))
            else:
                writer.append_data(np.asarray(frame))
        prev = last
        print(f"{seg['name']:9s} {nf / FPS:5.1f} s", flush=True)
    if args.preview:
        shutil.rmtree(tmp, ignore_errors=True)
        return
    writer.close()

    audio = []
    for spans, nf in plan:
        buf = np.zeros(nf * sr // FPS, dtype=np.int16)
        for st, _ in spans:
            c = clips.pop(0)
            i = int(st * sr)
            buf[i: i + len(c)] = c[: max(0, len(buf) - i)]
        audio.append(buf)
    wav = os.path.join(tmp, "narration.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.concatenate(audio).tobytes())
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-c:v", "copy",
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", "-shortest", OUT], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB, {total / FPS:.0f} s")


if __name__ == "__main__":
    main()
