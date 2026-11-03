"""README figures (light and dark variants) from results/benchmark.json and models/train_log.json.

    python -m chowkpilot.eval.figures

Emphasis form: ChowkPilot in the accent green, baselines in a recessive gray, values
labelled directly. Colours were checked with a colour-vision-deficiency validator on
each mode's surface (green vs gray: deutan dE 11.7 light / 15.7 dark).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..scenarios import registry  # noqa: E402

OUT = os.path.join(registry.ROOT, "docs", "img")
THEMES = {
    "light": dict(bg="#ffffff", text="#1f2328", muted="#59636e", grid="#d8dee4", ours="#15803d", other="#737d96"),
    "dark": dict(bg="#0d1117", text="#e6edf3", muted="#9198a1", grid="#262c36", ours="#16a34a", other="#6b7aa6"),
}
PANEL_TITLES = {"syn_single_2lane_inc": "4-arm junction (synthetic)", "syn_arterial3": "3-signal arterial (synthetic)",
                "hyd_gachibowli": "Gachibowli, Hyderabad (OSM)", "hyd_secunderabad": "Secunderabad corridor (OSM)",
                "hyd_hitech": "Hitech City, Cyber Towers (OSM)", "hyd_ameerpet": "Ameerpet / Begumpet (OSM)"}
SHORT = {"fixed": "Fixed-time (standard plan)", "webster": "Webster fixed-time", "actuated": "Vehicle-actuated",
         "maxpressure": "Max-pressure", "chowkpilot": "ChowkPilot"}


def _style(ax, th):
    ax.set_facecolor(th["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["grid"])
    ax.tick_params(colors=th["muted"], length=0, labelsize=9)
    ax.grid(axis="x", color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _hbars(ax, th, names, values, cis, ours_mask, fmt):
    y = list(range(len(names)))[::-1]
    colors = [th["ours"] if o else th["other"] for o in ours_mask]
    ax.barh(y, values, height=0.56, color=colors, zorder=2)
    for yi, v, c in zip(y, values, cis):
        if c:
            ax.plot([v - c, v + c], [yi, yi], color=th["text"], linewidth=1.1, alpha=0.6, zorder=3)
    xmax = max(v + (c or 0) for v, c in zip(values, cis))
    xmin = min(0, min(values))
    for yi, v, c, o in zip(y, values, cis, ours_mask):
        ax.text(max(v, 0) + (c or 0) + 0.02 * (xmax - xmin), yi, fmt(v), va="center", fontsize=9,
                color=th["text"] if o else th["muted"], fontweight="bold" if o else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for lab, o in zip(ax.get_yticklabels(), ours_mask):
        lab.set_color(th["text"] if o else th["muted"])
        lab.set_fontweight("bold" if o else "normal")
    ax.set_xlim(xmin * 1.15 if xmin < 0 else 0, xmax * 1.28)
    if xmin < 0:
        ax.axvline(0, color=th["muted"], linewidth=0.8)


def benchmark(b, theme):
    th = THEMES[theme]
    scen = b["test_scenarios"]
    fig, axes = plt.subplots(1, len(scen), figsize=(4.4 * len(scen), 2.9), facecolor=th["bg"])
    for ax, sid in zip(axes, scen):
        _style(ax, th)
        cells = b["cells"][sid]
        ctrls = b["controllers"]
        _hbars(ax, th, [SHORT[c] for c in ctrls], [cells[c]["mean_delay"]["mean"] for c in ctrls],
               [cells[c]["mean_delay"]["ci"] for c in ctrls], [c == "chowkpilot" for c in ctrls],
               lambda v: f"{v:.0f} s")
        ax.set_title(PANEL_TITLES.get(sid, b["titles"][sid]), color=th["text"], fontsize=10.5, loc="left", fontweight="bold")
        ax.set_xlabel("mean delay per vehicle (s)", color=th["muted"], fontsize=9)
    fig.tight_layout()
    return fig


def robustness(b, theme):
    th = THEMES[theme]
    conds = b["robustness"]["conditions"]
    ctrls = b["robustness"]["controllers"]
    fig, axes = plt.subplots(1, len(conds), figsize=(2.9 * len(conds), 2.3), facecolor=th["bg"], sharey=True)
    for i, (ax, cond) in enumerate(zip(axes, conds)):
        _style(ax, th)
        vals = [-(cond["values"][c] or 0.0) for c in ctrls]  # positive = reduction vs fixed-time
        _hbars(ax, th, [SHORT[c] for c in ctrls], vals, [0] * len(ctrls), [c == "chowkpilot" for c in ctrls],
               lambda v: f"{v:+.0f}%".replace("+", "−") if v >= 0 else f"+{-v:.0f}%")
        ax.set_title(cond["label"], color=th["text"], fontsize=10, loc="left", fontweight="bold")
        if i:
            ax.tick_params(labelleft=False)
    axes[len(conds) // 2].set_xlabel("delay change vs standard fixed-time plan (longer bar = larger reduction)",
                                     color=th["muted"], fontsize=9)
    fig.tight_layout()
    return fig


def _by_layout(d):
    """{'sid@seed': v} -> {'sid': mean over seeds} (older logs used plain 'sid' keys)."""
    out = {}
    for k, v in d.items():
        out.setdefault(k.split("@")[0], []).append(v)
    return {k: sum(v) / len(v) for k, v in out.items()}


def validation_points(log, spath):
    """Checkpoints validated during training plus those scored afterwards by select --fill, on the same tasks."""
    keys = set(log["baseline_validation"]["actuated"])
    pts = {v["ckpt"]: v for v in log["validation"] if v.get("delay") and v.get("ckpt")}
    if os.path.isfile(spath):
        for c in json.load(open(spath, encoding="utf-8"))["candidates"]:
            if c.get("updates") is not None and c.get("delay") and c["ckpt"] not in pts:
                pts[c["ckpt"]] = c
    out = []
    for v in pts.values():
        d = {k: x for k, x in v["delay"].items() if k in keys}
        if set(d) == keys:
            out.append({"ckpt": v["ckpt"], "updates": v["updates"], "delay": d})
    return sorted(out, key=lambda v: v["updates"])


def learning_curve(log, theme, selected=None, val=None):
    th = THEMES[theme]
    val = val or [v for v in log["validation"] if v["delay"]]
    if not val:
        return None
    base = {name: _by_layout(vals) for name, vals in log["baseline_validation"].items()}
    tasks = list(_by_layout(val[-1]["delay"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.0), facecolor=th["bg"])
    titles = {s.id: s.title for s in registry.load_all()}
    handles = []
    for ax, task in zip(axes, tasks):
        _style(ax, th)
        ax.grid(axis="y", color=th["grid"], linewidth=0.8)
        pts = [(v["updates"] / 1000, _by_layout(v["delay"]).get(task)) for v in val]
        pts = [(x, y) for x, y in pts if y is not None]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        line, = ax.plot(xs, ys, color=th["ours"], linewidth=2, marker="o", markersize=3.5, zorder=3,
                        label="ChowkPilot checkpoints (greedy, held-out seeds)")
        refs = []
        for name, ls in (("fixed", "-."), ("maxpressure", ":"), ("actuated", "--")):
            refs.append(ax.axhline(base[name][task], color=th["other"], linewidth=1.1, linestyle=ls, zorder=2,
                                   label=SHORT[name]))
        if selected:
            sel = [(x, y) for (x, y), v in zip(pts, val) if v.get("ckpt") in selected]
            if sel:
                star = ax.scatter([x for x, _ in sel], [y for _, y in sel], s=90, marker="*", color=th["text"], zorder=4,
                                  label="selected checkpoint" if len(selected) == 1 else "ensemble members")
                if len(sel) == 1:
                    ax.annotate(f"{sel[0][1]:.0f} s", sel[0], xytext=(6, -14), textcoords="offset points",
                                color=th["text"], fontsize=9, fontweight="bold")
                refs.append(star)
        top = 1.2 * max(base["fixed"][task], *(ys[1:] or [0]))
        ax.set_ylim(0, top)
        if ys and ys[0] > top:
            ax.annotate(f"untrained: {ys[0]:.0f} s", (xs[0], top), xytext=(4, -12), textcoords="offset points",
                        color=th["muted"], fontsize=8)
        ax.set_title(PANEL_TITLES.get(task, titles.get(task, task)), color=th["text"], fontsize=10, loc="left",
                     fontweight="bold")
        ax.set_xlabel("gradient updates (thousands)", color=th["muted"], fontsize=9)
        ax.set_ylabel("mean delay (s)", color=th["muted"], fontsize=9)
        handles = [line] + refs
    leg = fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
                     bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts():
        t.set_color(th["muted"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    made = []
    bpath = os.path.join(registry.ROOT, "results", "benchmark.json")
    lpath = os.path.join(registry.ROOT, "models", "train_log.json")
    b = json.load(open(bpath, encoding="utf-8")) if os.path.isfile(bpath) else None
    log = json.load(open(lpath, encoding="utf-8")) if os.path.isfile(lpath) else None
    spath = os.path.join(registry.ROOT, "models", "selection.json")
    sel = json.load(open(spath, encoding="utf-8"))["selected"] if os.path.isfile(spath) else None
    selected = (sel.get("members") or [sel["ckpt"]]) if sel else None  # checkpoint names to mark on the curve
    for theme in THEMES:
        figs = {}
        if b:
            figs["benchmark"] = benchmark(b, theme)
            if b.get("robustness"):
                figs["robustness"] = robustness(b, theme)
        if log:
            figs["learning_curve"] = learning_curve(log, theme, selected, validation_points(log, spath))
        for name, fig in figs.items():
            if fig is None:
                continue
            path = os.path.join(OUT, f"{name}-{theme}.png")
            fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
            plt.close(fig)
            made.append(path)
    print("\n".join(made))


if __name__ == "__main__":
    main()
