"""Render a side-by-side demo GIF (baseline vs ChowkPilot) from recorded replays.

    python -m chowkpilot.eval.make_gif --scenario hyd_hitech --baseline fixed

Every frame is drawn from real SUMO output (docs/data/replays), not animated by hand.
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib
from PIL import Image, ImageDraw, ImageFont

from ..scenarios import registry
from .replay import REPLAY_DIR, load_replay, network_geometry

BG, LANE, JUNC, TEXT, MUTED = (7, 11, 22), (30, 40, 70), (26, 36, 64), (232, 237, 255), (141, 154, 196)
BASE, OURS = (57, 135, 229), (22, 163, 74)
TYPE_COLORS = [(251, 191, 36), (96, 165, 250), (52, 211, 153), (244, 114, 182), (167, 139, 250), (255, 255, 255)]
SIGNAL = {"G": (34, 197, 94), "y": (245, 158, 11), "r": (239, 68, 68)}
NAMES = {"fixed": "Fixed-time (standard plan)", "actuated": "Vehicle-actuated", "maxpressure": "Max-pressure",
         "webster": "Fixed-time (Webster)", "chowkpilot": "ChowkPilot (AI)"}


def _font(size, bold=False):
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(path, size)


class Panel:
    def __init__(self, geom, w, h, margin=80.0):
        self.g, self.w, self.h = geom, w, h
        pts = geom["signals"] or [{"x": 0, "y": 0}]
        x0, x1 = min(p["x"] for p in pts), max(p["x"] for p in pts)
        y0, y1 = min(p["y"] for p in pts), max(p["y"] for p in pts)
        m = max(margin, 0.15 * max(x1 - x0, y1 - y0))
        x0, x1, y0, y1 = x0 - m, x1 + m, y0 - m, y1 + m
        self.s = min(w / (x1 - x0), h / (y1 - y0))
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.bg = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(self.bg)
        for poly in geom["junctions"]:
            d.polygon([self.xy(x, y) for x, y in poly], fill=JUNC)
        for lane in geom["lanes"]:
            d.line([self.xy(x, y) for x, y in lane["s"]], fill=LANE, width=max(1, int(lane["w"] * self.s)))

    def xy(self, x, y):
        return ((x - self.cx) * self.s + self.w / 2, self.h / 2 - (y - self.cy) * self.s)

    def draw(self, frame):
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        s = self.s
        for sig in self.g["signals"]:
            st = frame["s"][sig["j"]] if sig["j"] < len(frame["s"]) else ""
            col = SIGNAL["r"]
            for i in sig["links"]:
                c = st[i] if i < len(st) else "r"
                if c in "Gg":
                    col = SIGNAL["G"]
                    break
                if c in "yY":
                    col = SIGNAL["y"]
            px, py = self.xy(sig["x"], sig["y"])
            half = max(2.5, sig["w"] * s * 0.5)
            a = -sig["a"]
            ux, uy = -math.sin(a) * half, math.cos(a) * half  # along the stop line
            d.line([(px - ux, py - uy), (px + ux, py + uy)], fill=col, width=max(2, int(1.2 * s)))
        v = frame["v"]
        types = self.g["types"]
        for i in range(0, len(v), 5):
            x, y, ang, code = v[i + 1] / 10, v[i + 2] / 10, math.radians(v[i + 3]), v[i + 4]
            ty, halted = code % 10, code >= 10
            T = types[ty] if ty < len(types) else types[1]
            L, W = max(2.2 / s, T["length"]), max(1.3 / s, T["width"])
            dx, dy = math.sin(ang), math.cos(ang)       # heading in world coordinates
            px_, py_ = math.cos(ang) * W / 2, -math.sin(ang) * W / 2
            fx, fy, rx, ry = x, y, x - dx * L, y - dy * L
            poly = [self.xy(fx + px_, fy + py_), self.xy(fx - px_, fy - py_),
                    self.xy(rx - px_, ry - py_), self.xy(rx + px_, ry + py_)]
            col = TYPE_COLORS[ty] if ty < len(TYPE_COLORS) else (200, 200, 200)
            if halted and ty != 5:
                col = tuple(int(0.45 * c + 0.55 * r) for c, r in zip(col, (239, 68, 68)))
            d.polygon(poly, fill=col)
        return img


def stopped_minutes(rec, t):
    s = rec["series"]
    return sum(h * 10 for tt, h in zip(s["t"], s["halting"]) if tt <= t) / 60


def render(scenario, baseline, t0=120, t1=720, step=2, width=1100, out=None):
    geom = network_geometry(scenario)
    base = load_replay(os.path.join(REPLAY_DIR, f"{scenario}__{baseline}.json.gz"))
    ours = load_replay(os.path.join(REPLAY_DIR, f"{scenario}__chowkpilot.json.gz"))
    pw, ph, top, bottom = width // 2 - 15, int(width * 0.36), 58, 64
    H = top + ph + bottom
    panels = [Panel(geom, pw, ph), Panel(geom, pw, ph)]
    f_title, f_tag, f_num, f_lbl, f_small = _font(17, True), _font(12, True), _font(20, True), _font(11), _font(11)
    title = registry.get(scenario).title
    frames = []
    for t in range(t0, t1 + 1, step):
        img = Image.new("RGB", (width, H), BG)
        d = ImageDraw.Draw(img)
        d.text((12, 10), f"ChowkPilot digital twin · {title}", font=f_title, fill=TEXT)
        d.text((12, 34), f"Same vehicles, two signal controllers · peak hour · t = {t // 60:02d}:{t % 60:02d}",
               font=f_small, fill=MUTED)
        for k, (rec, ctrl, col) in enumerate(((base, baseline, BASE), (ours, "chowkpilot", OURS))):
            x0 = 10 + k * (pw + 10)
            idx = max(0, min(len(rec["frames"]) - 1, t - rec["frames"][0]["t"]))
            frame = rec["frames"][idx]
            img.paste(panels[k].draw(frame), (x0, top))
            d.rectangle([x0, top, x0 + pw - 1, top + ph - 1], outline=(31, 43, 75))
            d.rounded_rectangle([x0 + 8, top + 8, x0 + 8 + 12 + d.textlength(NAMES[ctrl], font=f_tag), top + 30],
                                radius=6, fill=tuple(int(c * 0.3) for c in col), outline=col)
            d.text((x0 + 14, top + 12), NAMES[ctrl], font=f_tag, fill=TEXT)
            halted = sum(1 for i in range(4, len(frame["v"]), 5) if frame["v"][i] >= 10)
            values = [halted, stopped_minutes(rec, t), rec["mean_delay"]]
            if k == 0:
                base_values = values
            stats = [(f"{values[0]}", "stopped now"), (f"{values[1]:,.0f}", "vehicle-minutes stopped"),
                     (f"{values[2]:.0f} s", "mean delay (whole run)")]
            for n, (num, lbl) in enumerate(stats):
                sx = x0 + 6 + n * (pw // 3)
                better = k == 1 and values[n] < base_values[n]  # green only when ChowkPilot is actually better
                d.text((sx, top + ph + 8), num, font=f_num, fill=(74, 222, 128) if better else TEXT)
                d.text((sx, top + ph + 34), lbl, font=f_lbl, fill=MUTED)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=96))
    out = out or os.path.join(registry.ROOT, "docs", "img", f"demo_{scenario}.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="hyd_hitech")
    ap.add_argument("--baseline", default="fixed")
    ap.add_argument("--t0", type=int, default=120)
    ap.add_argument("--t1", type=int, default=720)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--width", type=int, default=1100)
    args = ap.parse_args()
    path = render(args.scenario, args.baseline, args.t0, args.t1, args.step, args.width)
    print(path, f"{os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
