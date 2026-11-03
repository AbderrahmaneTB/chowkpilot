"""Render the ~75 s demo video (MP4) for the submission.

    python -m chowkpilot.eval.make_video

Title and problem cards, two real side-by-side digital-twin sequences rendered from
docs/data/replays, then the benchmark results read from results/benchmark.json.
"""

from __future__ import annotations

import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from ..scenarios import registry
from .make_gif import BASE, BG, MUTED, NAMES, OURS, TEXT, Panel, _font, stopped_minutes
from .replay import REPLAY_DIR, load_replay, network_geometry

W, H, FPS = 1280, 720, 25
GREEN_TXT = (74, 222, 128)
OUT = os.path.join(registry.ROOT, "docs", "img", "chowkpilot_demo.mp4")


def card(lines, sub=None):
    """lines: list of (text, size, bold, color)."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 150
    for text, size, bold, color in lines:
        f = _font(size, bold)
        d.text((90, y), text, font=f, fill=color)
        y += int(size * 1.45)
    if sub:
        d.text((90, H - 80), sub, font=_font(18), fill=MUTED)
    return img


def stats_card(title, tiles, footer):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((90, 90), title, font=_font(40, True), fill=TEXT)
    for i, (value, label, source) in enumerate(tiles):
        x, y = 90 + (i % 2) * 560, 200 + (i // 2) * 200
        d.line([(x, y), (x, y + 150)], fill=(42, 57, 98), width=3)
        d.text((x + 22, y), value, font=_font(46, True), fill=TEXT)
        yy = y + 64
        for chunk in _wrap(label, 44):
            d.text((x + 22, yy), chunk, font=_font(20), fill=MUTED)
            yy += 28
        d.text((x + 22, yy + 4), source, font=_font(15), fill=(92, 106, 150))
    d.text((90, H - 70), footer, font=_font(18), fill=MUTED)
    return img


def _wrap(text, n):
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + [cur] if cur else out


def twin_frames(scenario, baseline, t0, t1, step, caption):
    geom = network_geometry(scenario)
    base = load_replay(os.path.join(REPLAY_DIR, f"{scenario}__{baseline}.json.gz"))
    ours = load_replay(os.path.join(REPLAY_DIR, f"{scenario}__chowkpilot.json.gz"))
    pw, ph, top = W // 2 - 15, 520, 92
    panels = [Panel(geom, pw, ph), Panel(geom, pw, ph)]
    title = registry.get(scenario).title
    for t in range(t0, t1 + 1, step):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((12, 12), f"{title}", font=_font(26, True), fill=TEXT)
        d.text((12, 50), f"{caption} · simulated clock {t // 60:02d}:{t % 60:02d}", font=_font(17), fill=MUTED)
        vals = []
        for k, (rec, ctrl, col) in enumerate(((base, baseline, BASE), (ours, "chowkpilot", OURS))):
            x0 = 10 + k * (pw + 10)
            idx = max(0, min(len(rec["frames"]) - 1, t - rec["frames"][0]["t"]))
            frame = rec["frames"][idx]
            img.paste(panels[k].draw(frame), (x0, top))
            d.rectangle([x0, top, x0 + pw - 1, top + ph - 1], outline=(31, 43, 75))
            tag = NAMES[ctrl]
            f_tag = _font(16, True)
            d.rounded_rectangle([x0 + 10, top + 10, x0 + 26 + d.textlength(tag, font=f_tag), top + 40], radius=7,
                                fill=tuple(int(c * 0.3) for c in col), outline=col)
            d.text((x0 + 18, top + 15), tag, font=f_tag, fill=TEXT)
            halted = sum(1 for i in range(4, len(frame["v"]), 5) if frame["v"][i] >= 10)
            v = [halted, stopped_minutes(rec, t), rec["mean_delay"]]
            vals.append(v)
            labels = ["stopped now", "vehicle-minutes stopped", "mean delay, whole run"]
            fmt = [f"{v[0]}", f"{v[1]:,.0f}", f"{v[2]:.0f} s"]
            for n in range(3):
                sx = x0 + 8 + n * (pw // 3)
                better = k == 1 and v[n] < vals[0][n]
                d.text((sx, top + ph + 10), fmt[n], font=_font(28, True), fill=GREEN_TXT if better else TEXT)
                d.text((sx, top + ph + 48), labels[n], font=_font(15), fill=MUTED)
        yield img


def results_card(b):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    s = b["summary"]
    d.text((90, 60), "Results on 3 real Hyderabad areas never seen in training", font=_font(34, True), fill=TEXT)
    d.text((90, 110), f"{b['seeds']} paired seeds per junction · peak demand · network fully drained",
           font=_font(18), fill=MUTED)
    tiles = [(s["delay_vs_fixed"], "mean delay vs standard fixed-time plan"),
             (s["delay_vs_webster"], "vs Webster-optimised fixed-time"),
             (s["delay_vs_actuated"], "vs vehicle-actuated control"),
             (s["delay_vs_maxpressure"], "vs max-pressure control")]
    for i, (v, label) in enumerate(tiles):
        x = 90 + i * 280
        d.text((x, 170), f"{'+' if v > 0 else '−'}{abs(v):.0f}%", font=_font(54, True),
               fill=GREEN_TXT if v < 0 else TEXT)
        for j, chunk in enumerate(_wrap(label, 24)):
            d.text((x, 245 + j * 26), chunk, font=_font(18), fill=MUTED)
    # per-junction bars: mean delay by controller
    ctrls = b["controllers"]
    y = 360
    for sid in b["test_scenarios"]:
        cells = b["cells"][sid]
        d.text((90, y), b["titles"][sid], font=_font(18, True), fill=TEXT)
        vmax = max(cells[c]["mean_delay"]["mean"] for c in ctrls)
        for k, c in enumerate(ctrls):
            v = cells[c]["mean_delay"]["mean"]
            bx, by = 520, y + k * 18 - 2
            d.rectangle([bx, by, bx + int(560 * v / vmax), by + 12], fill=OURS if c == "chowkpilot" else (107, 122, 166))
            d.text((bx + int(560 * v / vmax) + 8, by - 3), f"{v:.0f} s", font=_font(13), fill=TEXT if c == "chowkpilot" else MUTED)
            d.text((350, by - 3), NAMES[c].replace(" (AI)", ""), font=_font(13), fill=TEXT if c == "chowkpilot" else MUTED)
        y += 18 * len(ctrls) + 22
    return img


def fade(img, n_frames, writer):
    arr = np.asarray(img, dtype=np.float32)
    bg = np.asarray(Image.new("RGB", (W, H), BG), dtype=np.float32)
    k = min(8, n_frames // 4)
    for i in range(n_frames):
        a = min(1.0, (i + 1) / k, (n_frames - i) / k)
        writer.append_data((bg + (arr - bg) * a).astype(np.uint8))


def main():
    b = json.load(open(os.path.join(registry.ROOT, "results", "benchmark.json"), encoding="utf-8"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    fade(card([("ChowkPilot", 84, True, TEXT), ("One AI signal brain for every chowk.", 40, False, GREEN_TXT),
               ("A learned traffic-signal controller for India's lane-less mixed traffic,", 24, False, MUTED),
               ("tested zero-shot on real Hyderabad junctions it never saw.", 24, False, MUTED)],
              sub="NeuralStorm 2026 · Track 1 Future Tech · FT-01 Smart Traffic Signal Optimisation · Team Blackwater"),
         int(4 * FPS), writer)
    fade(stats_card("The problem", [
        ("123 h", "lost per driver per year to rush-hour congestion in Hyderabad", "TomTom Traffic Index 2026"),
        ("$22 bn", "yearly cost of congestion in Delhi, Mumbai, Bengaluru and Kolkata", "BCG, Unlocking Cities (2018)"),
        ("73%", "of Telangana's vehicles are two-wheelers, in traffic with no lane discipline", "Telangana Transport Dept. (2025)"),
        (">17 min", "average ambulance response time in Delhi", "The Hindu (2025)")],
        "Most junctions still run fixed-time plans that cannot react to the traffic in front of them."),
        int(6 * FPS), writer)
    fade(card([("How ChowkPilot works", 44, True, TEXT),
               ("1. Camera-style counts from the last 150 m of every lane", 26, False, TEXT),
               ("2. A learned version of max-pressure scores every signal phase", 26, False, TEXT),
               ("3. One 38k-parameter policy for any junction layout (deep RL, Double DQN)", 26, False, TEXT),
               ("4. Safety layer: min/max green, yellow, all-red, max-red, ambulance corridor", 26, False, TEXT),
               ("Simulated in SUMO digital twins built from OpenStreetMap,", 22, False, MUTED),
               ("with two-wheelers filtering between cars and left-hand driving.", 22, False, MUTED)]),
         int(7 * FPS), writer)
    for img in twin_frames("hyd_hitech", "fixed", 150, 690, 2, "Same vehicles, two signal controllers, real SUMO run"):
        writer.append_data(np.asarray(img))
    for img in twin_frames("hyd_secunderabad", "actuated", 200, 560, 2, "5-signal corridor vs vehicle-actuated control"):
        writer.append_data(np.asarray(img))
    fade(results_card(b), int(9 * FPS), writer)
    fade(card([("Try it yourself", 44, True, TEXT), ("abderrahmanetb.github.io/chowkpilot", 32, True, GREEN_TXT),
               ("github.com/AbderrahmaneTB/chowkpilot", 28, False, TEXT),
               ("Digital twin replays · phase-by-phase explanations · stress tests · full benchmark", 22, False, MUTED)],
              sub="Simulation: Eclipse SUMO · Map data © OpenStreetMap contributors (ODbL)"),
         int(4 * FPS), writer)
    writer.close()
    print(OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
