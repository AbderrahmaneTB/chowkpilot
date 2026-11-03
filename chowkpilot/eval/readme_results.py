"""Write the README results section from results/benchmark.json (no hand-typed numbers).

    python -m chowkpilot.eval.readme_results

Replaces the blocks between <!-- RESULTS-HERO --> ... <!-- /RESULTS-HERO --> and
<!-- RESULTS --> ... <!-- /RESULTS --> in README.md.
"""

from __future__ import annotations

import json
import os
import re

from ..scenarios import registry

NAMES = {"fixed": "Fixed-time (standard plan)", "webster": "Fixed-time (Webster-optimised)",
         "actuated": "Vehicle-actuated", "maxpressure": "Max-pressure", "chowkpilot": "**ChowkPilot**"}


def pct(x):
    return f"{'+' if x > 0 else '−'}{abs(x):.1f}%"


def num(v, dp, unit=""):
    return "–" if v is None or v != v else f"{v:.{dp}f}{unit}"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def picture(name, alt):
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="docs/img/{name}-dark.png">'
            f'<img alt="{alt}" src="docs/img/{name}-light.png"></picture>')


def rel(x, what):
    """Signed comparison in words: never reports a loss as a gain."""
    return f"**{abs(x):.0f}% lower** than {what}" if x < 0 else f"{x:.0f}% higher than {what}"


def narrative(b):
    """Per-junction reading of the results, including where a baseline wins."""
    test, paired, titles, ctrls = b["test_scenarios"], b["paired"], b["titles"], b["controllers"]
    rows, wins = [], 0
    for sid in test:
        cells = b["cells"][sid]
        ours = cells["chowkpilot"]["mean_delay"]["mean"]
        best = min((c for c in ctrls if c != "chowkpilot"), key=lambda c: cells[c]["mean_delay"]["mean"])
        pf, pa = paired[sid]["fixed"], paired[sid]["actuated"]
        txt = (f"- **{titles[sid]}:** {pct(pf['delay_change_pct'])} delay vs fixed-time (p={pval(pf['p_value'])}), "
               f"{pct(pa['delay_change_pct'])} vs vehicle-actuated (p={pval(pa['p_value'])}).")
        if ours <= cells[best]["mean_delay"]["mean"]:
            wins += 1
            txt += f" Lowest mean delay of the five controllers ({ours:.1f} s)."
        else:
            txt += f" {NAMES[best]} is best here ({cells[best]['mean_delay']['mean']:.1f} s vs ChowkPilot's {ours:.1f} s)."
        rows.append(txt)
    return "\n".join([f"ChowkPilot has the lowest mean delay on **{wins} of {len(test)}** unseen junctions.", ""] + rows)


def build(b, log):
    s, test, ctrls = b["summary"], b["test_scenarios"], b["controllers"]
    titles = b["titles"]
    paired = b["paired"]
    # the significance claim is only made when every per-junction difference is an improvement with p < 0.01
    better = [paired[sid][c]["delay_change_pct"] < 0 and paired[sid][c]["p_value"] < 0.01
              for sid in test for c in ("fixed", "actuated") if c in paired.get(sid, {})]
    areas = {1: "one real Hyderabad area", 2: "two real Hyderabad areas", 3: "three real Hyderabad areas"}.get(
        len(test), f"{len(test)} real Hyderabad areas")
    side = [f"{label} is {rel(s[key], 'with fixed-time')}" for key, label in (("co2_vs_fixed", "CO₂"),
            ("ambulance_vs_fixed", "ambulance delay")) if s.get(key) is not None and s[key] == s[key]]
    hero = (f"> **Result:** on {areas} it never saw in training, ChowkPilot's mean delay per vehicle is "
            f"{rel(s['delay_vs_fixed'], 'the standard fixed-time plan')}, {rel(s['delay_vs_webster'], 'a Webster-optimised plan')}, "
            f"{rel(s['delay_vs_actuated'], 'vehicle-actuated control')} and {rel(s['delay_vs_maxpressure'], 'max-pressure')}. "
            + (" and ".join(side) + " " if side else "")
            + f"({b['seeds']} paired seeds per junction"
            + ("; every per-junction difference vs fixed-time and vs actuated is an improvement significant at p < 0.01)."
               if better and all(better) else ")."))
    lines = []
    lines.append(f"All numbers below are produced by `python -m chowkpilot.eval.benchmark --seeds {b['seeds']}` and stored per "
                 f"run in [`results/benchmark.json`](results/benchmark.json). Each junction is simulated for a "
                 f"{b['horizon'] // 60}-minute peak (time-of-day profile, 4 ambulances/h) and then run until the network "
                 f"empties. **Delay** = SUMO time loss plus waiting-to-enter time, averaged over every vehicle. Seeds are "
                 f"**paired**: every controller sees exactly the same vehicles.")
    lines.append("")
    lines.append(picture("benchmark", "Mean delay per vehicle by controller on the three unseen junctions"))
    lines.append("")
    lines.append("### Reading the results")
    lines.append("")
    lines.append(narrative(b))
    lines.append("")
    lines.append("### Mean delay per vehicle (seconds, mean ± 95% CI)")
    lines.append("")
    lines.append("| Junction (unseen in training) | " + " | ".join(NAMES[c] for c in ctrls) + " |")
    lines.append("|---|" + "---:|" * len(ctrls))
    for sid in test:
        cells = b["cells"][sid]
        row = []
        best = min(cells[c]["mean_delay"]["mean"] for c in ctrls)
        for c in ctrls:
            m, ci = cells[c]["mean_delay"]["mean"], cells[c]["mean_delay"]["ci"]
            txt = f"{m:.1f} ± {ci:.1f}"
            row.append(f"**{txt}**" if m == best else txt)
        lines.append(f"| {titles[sid]} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("### ChowkPilot vs each baseline (paired by seed)")
    lines.append("")
    lines.append("| Junction | vs " + " | vs ".join(NAMES[c] for c in ctrls[:-1]) + " |")
    lines.append("|---|" + "---:|" * (len(ctrls) - 1))
    for sid in test:
        row = []
        for c in ctrls[:-1]:
            pr = paired[sid].get(c)
            row.append(f"{pct(pr['delay_change_pct'])} (wins {pr['wins']}/{pr['n']}, p={pval(pr['p_value'])})" if pr else "-")
        lines.append(f"| {titles[sid]} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("### Other metrics (mean over seeds)")
    lines.append("")
    metrics = [("p95_delay", "95th-percentile delay (s)", 1, min), ("mean_stops", "Stops per vehicle", 2, min),
               ("co2_kg", "CO₂ in the network (kg)", 1, min), ("fuel_kg", "Fuel (kg)", 1, min),
               ("throughput_horizon", "Vehicles served during the peak", 0, max),
               ("ambulance_delay", "Ambulance delay (s)", 1, min)]
    lines.append("| Junction | Metric | " + " | ".join(NAMES[c] for c in ctrls) + " |")
    lines.append("|---|---|" + "---:|" * len(ctrls))
    for sid in test:
        cells = b["cells"][sid]
        for i, (key, label, dp, better) in enumerate(metrics):
            vals = [cells[c][key]["mean"] for c in ctrls]
            ok = [v for v in vals if v == v]
            if not ok:
                continue
            best = better(ok)
            row = [("**" + num(v, dp) + "**") if v == best else num(v, dp) for v in vals]
            lines.append(f"| {titles[sid] if i == 0 else ''} | {label} | " + " | ".join(row) + " |")
    lines.append("")
    if b.get("impact"):
        lines.append("### What that means on the street")
        lines.append("")
        lines.append(f"Per simulated peak period ({b['horizon'] // 60} minutes of arrivals, run until the network "
                     "empties), compared with the standard fixed-time plan:")
        lines.append("")
        lines.append("| Junction | Vehicles | Vehicle-hours of delay saved | vs actuated | CO₂ saved (kg) | Fuel saved (kg) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for sid in test:
            im = b["impact"][sid]
            lines.append(f"| {titles[sid]} | {im['vehicles']:.0f} | {im['vehicle_hours_saved_vs_fixed']:.1f} | "
                         f"{im['vehicle_hours_saved_vs_actuated']:.1f} | {im['co2_kg_saved_vs_fixed']:.1f} | "
                         f"{im['fuel_kg_saved_vs_fixed']:.1f} |")
        lines.append("")
    if b.get("robustness"):
        R = b["robustness"]
        lines.append("### Robustness: change in mean delay vs the standard fixed-time plan")
        lines.append("")
        lines.append(picture("robustness", "Delay change vs fixed-time under different conditions"))
        lines.append("")
        lines.append("| Condition | " + " | ".join(NAMES[c] for c in R["controllers"]) + " | Fixed-time mean delay |")
        lines.append("|---|" + "---:|" * (len(R["controllers"]) + 1))
        for cond in R["conditions"]:
            vals = [cond["values"][c] for c in R["controllers"]]
            best = min(v for v in vals if v is not None)
            row = [("**" + pct(v) + "**") if v == best else pct(v) for v in vals]
            lines.append(f"| {cond['label']} | " + " | ".join(row) + f" | {cond['mean_delay']['fixed']:.0f} s |")
        lines.append("")
        lines.append(f"Averaged over the three unseen junctions, {R.get('seeds', 5)} seeds each (seeds 101 onwards, "
                     "disjoint from the main benchmark). *Camera noise* adds 20% multiplicative count error and drops 10% "
                     "of vehicles from the camera counts that ChowkPilot and max-pressure see; fixed-time needs no "
                     "detection and SUMO's actuated controller keeps ideal loop detectors, so noise favours those baselines.")
        lines.append("")
    noev = {sid: b["cells"][sid].get("chowkpilot_noev") for sid in test}
    if all(noev.values()):
        lines.append("### Ablation: ambulance green corridor")
        lines.append("")
        lines.append("| Junction | Ambulance delay, ChowkPilot without corridor | with corridor | Mean delay of all traffic, without | with |")
        lines.append("|---|---:|---:|---:|---:|")
        for sid in test:
            n, w = noev[sid], b["cells"][sid]["chowkpilot"]
            lines.append(f"| {titles[sid]} | {num(n['ambulance_delay']['mean'], 1, ' s')} | "
                         f"{num(w['ambulance_delay']['mean'], 1, ' s')} | {num(n['mean_delay']['mean'], 1, ' s')} | "
                         f"{num(w['mean_delay']['mean'], 1, ' s')} |")
        lines.append("")
    if log:
        lines.append("### Learning curve")
        lines.append("")
        lines.append(picture("learning_curve", "Validation delay during training compared with baselines"))
        lines.append("")
        args = log.get("args", {})
        curve = log.get("curve", [])
        last = curve[-1] if curve else {}
        sel_path = os.path.join(registry.ROOT, "models", "selection.json")
        sel = json.load(open(sel_path, encoding="utf-8"))["selected"] if os.path.isfile(sel_path) else None
        lines.append(f"Greedy validation of each checkpoint on training layouts with held-out seeds "
                     f"({args.get('minutes', '?')} minutes of training on a laptop CPU, {args.get('workers', '?')} SUMO "
                     f"workers, {last.get('env_steps', 0):,} decisions collected, {last.get('updates', 0):,} gradient "
                     "updates)."
                     + ((f" The deployed policy is an ensemble of checkpoints {', '.join(f'`{m}`' for m in sel['members'])} "
                         "(it averages their phase scores)" if sel.get("members") else
                         f" The deployed policy is checkpoint `{sel['ckpt']}` ({sel['updates']:,} updates)")
                        + f", the best by validation delay relative to vehicle-actuated control (score {sel['score']:.3f}; "
                        "[`models/selection.json`](models/selection.json))." if sel else "")
                     + " The real test junctions were never used for training, validation or model selection.")
        lines.append("")
    return hero, "\n".join(lines)


def main():
    b = json.load(open(os.path.join(registry.ROOT, "results", "benchmark.json"), encoding="utf-8"))
    lp = os.path.join(registry.ROOT, "models", "train_log.json")
    log = json.load(open(lp, encoding="utf-8")) if os.path.isfile(lp) else None
    hero, body = build(b, log)
    path = os.path.join(registry.ROOT, "README.md")
    readme = open(path, encoding="utf-8").read()
    for tag, content in (("RESULTS-HERO", hero), ("RESULTS", body)):
        block = f"<!-- {tag} -->\n{content}\n<!-- /{tag} -->"
        pattern = re.compile(rf"<!-- {tag} -->.*?<!-- /{tag} -->", re.S)
        if pattern.search(readme):
            readme = pattern.sub(lambda _m: block, readme)
        else:
            readme = readme.replace(f"<!-- {tag} -->", block, 1)
    open(path, "w", encoding="utf-8").write(readme)
    print("README results updated")


if __name__ == "__main__":
    main()
