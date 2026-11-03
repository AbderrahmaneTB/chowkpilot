"""Narrated walkthrough video of the control room (Windows: narration uses the built-in offline speech voice).

    python -m http.server 8090 --directory docs        # in another terminal
    python tools/screenshots.py                        # refresh docs/img/screen_*.png first
    python tools/make_walkthrough.py                   # -> docs/img/chowkpilot_walkthrough.mp4

The twin and explanation segments are captured live from the page (real recorded SUMO replays, stepped frame by
frame); the other segments use the site screenshots. Every number in the narration comes from results/benchmark.json.
"""

from __future__ import annotations

import asyncio
import base64
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
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chowkpilot.eval.make_video import card, _font, BG, TEXT, MUTED, GREEN_TXT  # noqa: E402
from screenshots import CDP, READY, find_chrome  # noqa: E402

IMG = os.path.join(ROOT, "docs", "img")
OUT = os.path.join(IMG, "chowkpilot_walkthrough.mp4")
BASE = "http://localhost:8090/"
W, H, FPS = 1280, 720, 25
VIEW_H = 596  # visual area above the caption band
VOICE = "Microsoft Zira Desktop"


def numbers():
    b = json.load(open(os.path.join(ROOT, "results", "benchmark.json"), encoding="utf-8"))
    s = b["summary"]
    rob = {c["label"]: c["values"]["chowkpilot"] for c in b["robustness"]["conditions"]}
    noise = next(v for k, v in rob.items() if k.lower().startswith("camera noise"))
    return {"fixed": -s["delay_vs_fixed"], "mp": -s["delay_vs_maxpressure"], "act": -s["delay_vs_actuated"],
            "amb": -s["ambulance_vs_fixed"], "co2": -s["co2_vs_fixed"], "noise": -noise, "seeds": b["seeds"]}


def segments(n):
    """(kind, source, narration as spoken, caption as shown)."""
    f = lambda x: f"{x:.0f}"  # noqa: E731
    return [
        ("card", "title",
         "Chowk Pilot. One A I signal brain for every chowk. Built by Team Blackwater for NeuralStorm 2026, "
         "problem F T zero one: smart traffic signal optimisation.", None),
        ("still", ("screen_hero.png", 0.0, 0.0),
         "Hyderabad drivers lose 123 hours a year to rush-hour traffic. Most junctions still run fixed-time plans, "
         "and the traffic is mixed and lane-less: nearly three in four vehicles in Telangana are two-wheelers.",
         "Hyderabad drivers lose 123 hours a year to rush-hour traffic. Most junctions still run fixed-time plans, "
         "and the traffic is mixed and lane-less: nearly three in four vehicles in Telangana are two-wheelers."),
        ("still", ("screen_how.png", 0.0, 0.0),
         "Chowk Pilot is a reinforcement learning controller. Every five seconds it decides which approach gets "
         "green, using only what a roadside camera can count. One small policy runs any junction layout, and we "
         "tested it on three real Hyderabad areas it never saw in training.",
         "ChowkPilot is a reinforcement-learning controller. Every 5 seconds it decides which approach gets green, "
         "using only what a roadside camera can count. One small policy runs any junction layout, tested on three "
         "real Hyderabad areas it never saw in training."),
        ("live", ("twin", 240, 640),
         "This is the control room. Both views replay the same recorded simulation of Hitech City at peak hour, "
         "with exactly the same vehicles. On the left, the standard fixed-time plan. On the right, Chowk Pilot. "
         "Stopped vehicles turn red. Watch the queues grow on the left, while Chowk Pilot keeps them short.",
         "The control room replays the same recorded SUMO simulation of Hitech City at peak hour, with exactly the "
         "same vehicles. Left: the standard fixed-time plan. Right: ChowkPilot. Stopped vehicles turn red."),
        ("live", ("explain", 640, 760),
         "Every decision is explained. For each signal phase, the panel shows the learned score, the stopped "
         "vehicles it would release, how long they have waited, and the max-pressure score the network uses as a prior.",
         "Every decision is explained: for each phase, the learned score, the stopped vehicles it would release, "
         "how long they have waited, and the max-pressure score the network uses as a prior."),
        ("still", ("screen_bench.png", 0.0, 0.30),
         f"On the three unseen areas, with {n['seeds']} paired seeds per junction, Chowk Pilot cut average delay by "
         f"{f(n['fixed'])} percent compared with fixed-time signals, and by {f(n['mp'])} percent compared with "
         f"max-pressure control. Against vehicle-actuated control it is level, {f(n['act'])} percent lower and not "
         "significant. A Webster plan given each run's exact demand is still best at two of the three areas, "
         "and we report that openly.",
         f"On the three unseen areas ({n['seeds']} paired seeds each), average delay is {f(n['fixed'])}% lower than "
         f"fixed-time and {f(n['mp'])}% lower than max-pressure. Against vehicle-actuated control it is level "
         f"({f(n['act'])}% lower, not significant). A Webster plan given each run's exact demand is best at 2 of 3 areas."),
        ("still", ("screen_bench.png", 0.22, 0.22),
         f"With the green corridor, ambulance delay fell by {f(n['amb'])} percent, and carbon dioxide by "
         f"{f(n['co2'])} percent. Even with twenty percent camera counting error, delay stayed {f(n['noise'])} "
         "percent below fixed-time.",
         f"Ambulance delay {f(n['amb'])}% lower and CO₂ {f(n['co2'])}% lower than fixed-time. With 20% camera "
         f"counting error, delay is still {f(n['noise'])}% below fixed-time."),
        ("still", ("screen_deploy.png", 0.0, 0.0),
         "A decision takes under a millisecond on a laptop processor. The path to Hyderabad is a calibrated twin of "
         "one corridor, a shadow-mode trial with traffic engineers, and then a supervised pilot at three to five junctions.",
         "A decision takes under a millisecond on a laptop CPU. Path to Hyderabad: a calibrated twin of one corridor, "
         "a shadow-mode trial with traffic engineers, then a supervised pilot at 3–5 junctions."),
        ("card", "end",
         "Everything is open source and reproducible. Try the live control room at the link on screen.", None),
    ]


# ------------------------------------------------------------------ narration
def speak(text, path):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        txt = f.name
    ps = (f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{VOICE}'); $s.Rate = 0; $s.SetOutputToWaveFile('{path}'); "
          f"$s.Speak((Get-Content -Raw -Encoding UTF8 '{txt}')); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    os.remove(txt)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def join_audio(parts, gaps, path):
    params, frames = None, []
    for p, gap in zip(parts, gaps):
        with wave.open(p) as w:
            params = params or w.getparams()
            frames.append(w.readframes(w.getnframes()))
            frames.append(b"\0" * int(gap * w.getframerate()) * w.getsampwidth() * w.getnchannels())
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(b"".join(frames))


# ------------------------------------------------------------------ frames
def fit(img, w=W, h=VIEW_H, y_frac=None, contain=False):
    """Scale an image to the frame width (or to fit both sides with contain); pan vertically (y_frac 0..1 of the
    overflow) when it is taller."""
    s = min(w / img.width, h / img.height) if contain else w / img.width
    if contain:
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), BG)
        canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
        return canvas
    img = img.resize((w, max(1, int(img.height * s))), Image.LANCZOS)
    if img.height > h:
        y = int((img.height - h) * (y_frac or 0.0))
        return img.crop((0, y, w, y + h))
    canvas = Image.new("RGB", (w, h), BG)
    canvas.paste(img, (0, (h - img.height) // 2))
    return canvas


def caption_chunks(text, n):
    words, out, cur = text.split(), [], ""
    for w_ in words:  # about 2 lines of 78 characters per chunk, broken at sentence ends when possible
        cur = (cur + " " + w_).strip()
        if len(cur) > 120 or (len(cur) > 60 and cur.endswith((".", ":"))):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    total = sum(len(c) for c in out)
    starts, acc = [], 0
    for c in out:
        starts.append(acc / total * n)
        acc += len(c)
    return list(zip(starts, out))


def wrap(text, n=78):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        if len(cur) + len(w_) + 1 > n:
            lines.append(cur)
            cur = w_
        else:
            cur = (cur + " " + w_).strip()
    return lines + [cur] if cur else lines


def compose(view, caption):
    frame = Image.new("RGB", (W, H), BG)
    frame.paste(view, (0, 0))
    if caption:
        d = ImageDraw.Draw(frame)
        d.line([(0, VIEW_H), (W, VIEW_H)], fill=(31, 43, 75), width=1)
        for i, line in enumerate(wrap(caption)[:3]):
            d.text((40, VIEW_H + 18 + i * 32), line, font=_font(24), fill=TEXT)
    return frame


def end_card():
    return card([("Try it yourself", 44, True, TEXT), ("abderrahmanetb.github.io/chowkpilot", 32, True, GREEN_TXT),
                 ("github.com/AbderrahmaneTB/chowkpilot", 28, False, TEXT),
                 ("Team Blackwater: Abderrahmane Tayeb Boudjema · Ouail Imad Eddine Zitouni", 22, False, MUTED)],
                sub="Simulation: Eclipse SUMO · Map data © OpenStreetMap contributors (ODbL) · Narration: synthetic voice")


def title_card():
    return card([("ChowkPilot", 84, True, TEXT), ("One AI signal brain for every chowk.", 40, False, GREEN_TXT),
                 ("A learned traffic-signal controller for India's lane-less mixed traffic,", 24, False, MUTED),
                 ("tested on real Hyderabad junctions it never saw.", 24, False, MUTED)],
                sub="NeuralStorm 2026 · FT-01 Smart Traffic Signal Optimisation · Team Blackwater")


async def live_frames(kind, t0, t1, n_frames):
    """Step the recorded replay from t0 to t1 and capture the page region for each video frame."""
    prof = tempfile.mkdtemp(prefix="chowk_walk_")
    proc = subprocess.Popen([find_chrome(), "--headless=new", "--remote-debugging-port=9335", f"--user-data-dir={prof}",
                             "--hide-scrollbars", "--no-first-run", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = []
    try:
        for _ in range(60):
            try:
                page = next(t for t in json.load(urllib.request.urlopen("http://127.0.0.1:9335/json/list", timeout=2))
                            if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = CDP(ws)
            await cdp("Page.enable")
            await cdp("Emulation.setDeviceMetricsOverride", width=1440, height=2400, deviceScaleFactor=1, mobile=False)
            await cdp("Page.navigate", url=f"{BASE}?walk={kind}#scenario=hyd_hitech&baseline=fixed&t={t0}&junction=0")
            t_start = time.time()
            while not await cdp.js(READY) and time.time() - t_start < 40:
                await asyncio.sleep(0.5)
            await asyncio.sleep(2.0)
            sel = [".controls", "#runSummary"] if kind == "twin" else [".below > .card:last-child"]
            rect = json.loads(await cdp.js("JSON.stringify(%s.map(s => { const r = document.querySelector(s)"
                                           ".getBoundingClientRect(); return [r.left, r.top + scrollY, r.right, "
                                           "r.bottom + scrollY]; }))" % json.dumps(sel)))
            x0, y0 = min(r[0] for r in rect) - 8, min(r[1] for r in rect) - 8
            x1, y1 = max(r[2] for r in rect) + 8, max(r[3] for r in rect) + 8
            clip = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "scale": 1}
            for i in range(n_frames):
                t = t0 + (t1 - t0) * i / max(1, n_frames - 1)
                await cdp.js(f"state.t = {t:.2f}; lastPanel = 0; render(); 1")
                shot = await cdp("Page.captureScreenshot", format="jpeg", quality=92, clip=clip)
                frames.append(Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB"))
    finally:
        proc.terminate()
        shutil.rmtree(prof, ignore_errors=True)
    return frames


def main():
    n = numbers()
    tmp = tempfile.mkdtemp(prefix="chowk_walk_audio_")
    segs = segments(n)
    wavs, durs = [], []
    for i, (_, _, say, _) in enumerate(segs):
        p = os.path.join(tmp, f"s{i}.wav")
        durs.append(speak(say, p))
        wavs.append(p)
    gap = 0.7
    join_audio(wavs, [gap] * len(wavs), os.path.join(tmp, "narration.wav"))
    silent = os.path.join(tmp, "video.mp4")
    writer = imageio.get_writer(silent, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for (kind, src, _, cap), dur in zip(segs, durs):
        n_frames = int(round((dur + gap) * FPS))
        chunks = caption_chunks(cap, n_frames) if cap else []

        def caption_at(k):
            cur = None
            for start, c in chunks:
                if k >= start:
                    cur = c
            return cur

        if kind == "card":
            img = np.asarray(title_card() if src == "title" else end_card())
            for _ in range(n_frames):
                writer.append_data(img)
        elif kind == "still":
            name, ya, yb = src
            base = Image.open(os.path.join(IMG, name)).convert("RGB")
            for k in range(n_frames):
                view = fit(base, y_frac=ya + (yb - ya) * k / max(1, n_frames - 1))
                writer.append_data(np.asarray(compose(view, caption_at(k))))
        else:
            what, t0, t1 = src
            for k, fr in enumerate(asyncio.run(live_frames(what, t0, t1, n_frames))):
                writer.append_data(np.asarray(compose(fit(fr, contain=what == "explain"), caption_at(k))))
        print(f"{kind:5s} {dur:5.1f} s", flush=True)
    writer.close()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", silent, "-i", os.path.join(tmp, "narration.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", OUT], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB, {sum(durs) + gap * len(durs):.0f} s")


if __name__ == "__main__":
    main()
