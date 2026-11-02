"""Benchmark ChowkPilot against the baselines on held-out real junctions.

    python -m chowkpilot.eval.benchmark --seeds 10 --workers 8

Every (scenario, seed) pair uses identical vehicles for every controller, so the
comparisons are paired. Reported: mean and 95% CI per cell, the paired change of
ChowkPilot vs each baseline with a paired t-test, and robustness conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from ..scenarios import registry

TEST = ["hyd_secunderabad", "hyd_hitech", "hyd_ameerpet"]
CONTROLLERS = ["fixed", "webster", "actuated", "maxpressure", "chowkpilot"]
METRICS = ["mean_delay", "p95_delay", "mean_wait", "mean_stops", "mean_travel_time",
           "throughput_horizon", "co2_kg", "fuel_kg", "unfinished", "teleports"]
ROBUSTNESS = [
    ("Off-peak (0.6×)", {"scale": 0.6}, {}),
    ("Peak (1.0×)", {"scale": 1.0}, {}),
    ("Heavy (1.25×)", {"scale": 1.25}, {}),
    ("Sudden surge", {"scale": 1.0, "profile": "surge"}, {}),
    ("Camera noise 20%", {"scale": 1.0}, {"noise_sigma": 0.2, "miss_prob": 0.1}),
]
ROBUST_CONTROLLERS = ["actuated", "maxpressure", "chowkpilot"]


def _task(t):
    from .runner import run_one
    sid, ctrl, seed, horizon, demand, sim, tag = t
    sim = dict(sim)
    if ctrl == "chowkpilot" and tag != "noev":
        sim["ev_priority"] = True
    t0 = time.time()
    try:
        r = run_one(sid, ctrl, seed=seed, horizon=horizon, demand=demand, sim=sim)
    except Exception as ex:  # keep the benchmark going; failures are reported
        return {"scenario": sid, "controller": ctrl, "seed": seed, "tag": tag, "error": repr(ex)}
    out = {k: r.get(k) for k in METRICS}
    amb = r.get("by_type", {}).get("ambulance")
    out["ambulance_delay"] = amb["mean_delay"] if amb else None
    out.update(scenario=sid, controller=ctrl, seed=seed, tag=tag, sec=round(time.time() - t0, 1),
               trips=r.get("trips"))
    return out


def ci95(x):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--robust-seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=1800)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ambulances", type=float, default=4.0)
    ap.add_argument("--out", default=os.path.join(registry.ROOT, "results", "benchmark.json"))
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()

    tasks = []
    base_demand = {"scale": 1.0, "profile": "peak", "ambulances_per_hour": args.ambulances}
    for sid in TEST:
        for seed in range(1, args.seeds + 1):
            for c in CONTROLLERS:
                tasks.append((sid, c, seed, args.horizon, base_demand, {}, "main"))
            tasks.append((sid, "chowkpilot", seed, args.horizon, base_demand, {}, "noev"))
    if not args.skip_robustness:
        for label, dem, sim in ROBUSTNESS:
            for sid in TEST:
                for seed in range(101, 101 + args.robust_seeds):
                    for c in ["fixed"] + ROBUST_CONTROLLERS:
                        tasks.append((sid, c, seed, args.horizon, {"profile": "peak", **dem}, sim, "rob:" + label))
    t0 = time.time()
    print(f"{len(tasks)} runs on {args.workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(_task, tasks, chunksize=1)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
    errors = [r for r in rows if "error" in r]
    for e in errors[:10]:
        print("ERROR", e, flush=True)
    ok = [r for r in rows if "error" not in r]

    titles = {s.id: s.title for s in registry.load_all()}
    cells, paired = {}, {}
    main_rows = [r for r in ok if r["tag"] == "main"]
    for sid in TEST:
        cells[sid] = {}
        for c in CONTROLLERS:
            rs = [r for r in main_rows if r["scenario"] == sid and r["controller"] == c]
            cells[sid][c] = {}
            for m in METRICS + ["ambulance_delay"]:
                mean, ci = ci95([r[m] for r in rs])
                cells[sid][c][m] = {"mean": mean, "ci": ci, "n": len(rs)}
        noev = [r for r in ok if r["tag"] == "noev" and r["scenario"] == sid]
        cells[sid]["chowkpilot_noev"] = {m: dict(zip(("mean", "ci"), ci95([r[m] for r in noev])))
                                         for m in ("mean_delay", "ambulance_delay", "co2_kg")}
        # paired comparison per seed
        paired[sid] = {}
        ours = {r["seed"]: r for r in main_rows if r["scenario"] == sid and r["controller"] == "chowkpilot"}
        for c in CONTROLLERS[:-1]:
            base = {r["seed"]: r for r in main_rows if r["scenario"] == sid and r["controller"] == c}
            common = sorted(set(ours) & set(base))
            if len(common) < 2:
                continue
            a = np.array([ours[s]["mean_delay"] for s in common])
            b = np.array([base[s]["mean_delay"] for s in common])
            change = (a - b) / b * 100
            mean, ci = ci95(change)
            p = float(stats.ttest_rel(a, b).pvalue)
            wins = int((a < b).sum())
            paired[sid][c] = {"delay_change_pct": mean, "ci": ci, "p_value": p, "wins": wins, "n": len(common)}

    def avg_change(metric, base):
        vals = []
        for sid in TEST:
            o, b = cells[sid]["chowkpilot"][metric]["mean"], cells[sid][base][metric]["mean"]
            if o is not None and b and not np.isnan(o) and not np.isnan(b):
                vals.append((o - b) / b * 100)
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "delay_vs_fixed": avg_change("mean_delay", "fixed"),
        "delay_vs_webster": avg_change("mean_delay", "webster"),
        "delay_vs_actuated": avg_change("mean_delay", "actuated"),
        "delay_vs_maxpressure": avg_change("mean_delay", "maxpressure"),
        "stops_vs_fixed": avg_change("mean_stops", "fixed"),
        "co2_vs_fixed": avg_change("co2_kg", "fixed"),
        "fuel_vs_fixed": avg_change("fuel_kg", "fixed"),
        "p95_vs_fixed": avg_change("p95_delay", "fixed"),
        "ambulance_vs_fixed": avg_change("ambulance_delay", "fixed"),
    }

    robustness = None
    if not args.skip_robustness:
        conds = []
        for label, _, _ in ROBUSTNESS:
            rr = [r for r in ok if r["tag"] == "rob:" + label]
            vals = {}
            for c in ROBUST_CONTROLLERS:
                changes = []
                for sid in TEST:
                    for seed in {r["seed"] for r in rr}:
                        o = [r for r in rr if r["scenario"] == sid and r["seed"] == seed and r["controller"] == c]
                        f = [r for r in rr if r["scenario"] == sid and r["seed"] == seed and r["controller"] == "fixed"]
                        if o and f and f[0]["mean_delay"]:
                            changes.append((o[0]["mean_delay"] - f[0]["mean_delay"]) / f[0]["mean_delay"] * 100)
                vals[c] = round(float(np.mean(changes)), 1) if changes else None
            abs_delay = {c: round(float(np.mean([r["mean_delay"] for r in rr if r["controller"] == c])), 1)
                         for c in ["fixed"] + ROBUST_CONTROLLERS if any(r["controller"] == c for r in rr)}
            conds.append({"label": label, "values": vals, "mean_delay": abs_delay})
        robustness = {"controllers": ROBUST_CONTROLLERS, "conditions": conds}

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": args.seeds, "horizon": args.horizon, "test_scenarios": TEST, "titles": titles,
        "controllers": CONTROLLERS, "cells": cells, "paired": paired, "summary": summary,
        "robustness": robustness, "errors": len(errors), "runs": len(ok),
        "note": ("Peak-hour demand calibrated per junction (see scenarios/index.json); 4 ambulances/h. "
                 "ChowkPilot includes the ambulance green-corridor layer; the baselines have none, as deployed. "
                 "Delay = SUMO timeLoss + insertion delay, averaged over all vehicles after the network empties."),
        "rows": ok,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(json.dumps(paired, indent=1))
    print(f"total {time.time() - t0:.0f}s, errors={len(errors)}")


if __name__ == "__main__":
    main()
