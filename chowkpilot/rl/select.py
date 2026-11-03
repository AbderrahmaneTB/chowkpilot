"""Select the deployed policy among training checkpoints, using validation runs only.

    python -m chowkpilot.rl.select [--revalidate 3] [--workers 10]

Each validated checkpoint is scored by its mean delay relative to vehicle-actuated control
on the validation tasks (training layouts, held-out seeds; equal weight per layout,
seeds averaged). One seed per task is noisy, so with --revalidate K the K best
checkpoints are run again on extra held-out seeds of the same layouts (actuated control
runs on the same seeds as the reference) and the final choice uses every seed. The best
one is copied to <models>/chowkpilot.npz. The real test junctions play no part in this choice.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..scenarios import registry

MODELS = os.path.join(registry.ROOT, "models")
EXTRA_SEEDS = (6262, 7272, 8282)  # training layouts only; disjoint from training, validation and test seeds


def score(delays: dict, base: dict) -> float | None:
    if not all(k in delays for k in base):
        return None
    by_layout = defaultdict(list)
    for k, ref in base.items():
        by_layout[k.split("@")[0]].append(delays[k] / ref)
    return float(np.mean([np.mean(v) for v in by_layout.values()]))


def _run(job):
    from ..eval.runner import run_one
    sid, ctrl, seed, demand, policy = job
    return run_one(sid, ctrl, seed=seed, horizon=1200, demand=demand, policy=policy)["mean_delay"]


def revalidate(top: list, workers: int, models: str = MODELS) -> dict:
    """Run the candidates and actuated control on the extra seeds; returns the actuated reference delays."""
    from .train import VALIDATION, task_key
    layouts = {}
    for sid, _, demand in VALIDATION:
        layouts.setdefault(sid, demand)
    tasks = [(sid, seed, demand) for sid, demand in layouts.items() for seed in EXTRA_SEEDS]
    jobs = [(sid, "actuated", seed, demand, None) for sid, seed, demand in tasks]
    for r in top:
        path = os.path.join(models, "checkpoints", r["ckpt"])
        jobs += [(sid, "chowkpilot", seed, demand, path) for sid, seed, demand in tasks]
    with ProcessPoolExecutor(workers) as ex:
        delays = list(ex.map(_run, jobs))
    n = len(tasks)
    base = {task_key(sid, seed): d for (sid, seed, _), d in zip(tasks, delays[:n])}
    for i, r in enumerate(top):
        r["extra_delay"] = {task_key(sid, seed): d for (sid, seed, _), d in zip(tasks, delays[n * (i + 1): n * (i + 2)])}
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revalidate", type=int, default=0, help="re-run the K best checkpoints on extra seeds")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=MODELS, help="training output folder (train_log.json, checkpoints/)")
    args = ap.parse_args()
    models = args.models
    log = json.load(open(os.path.join(models, "train_log.json"), encoding="utf-8"))
    base = json.load(open(os.path.join(models, "baseline_validation.json"), encoding="utf-8"))["actuated"]
    rows = []
    for v in log["validation"]:
        sc = score(v["delay"], base)
        if sc is not None and v.get("ckpt"):
            rows.append({"ckpt": v["ckpt"], "updates": v["updates"], "minutes": v.get("minutes"), "score": sc,
                         "delay": v["delay"]})
    if not rows:
        raise SystemExit("no complete validation rounds found")
    criterion = "mean validation delay / vehicle-actuated delay (training layouts, held-out seeds)"
    extra_base, pool = None, rows
    if args.revalidate:
        pool = sorted(rows, key=lambda r: r["score"])[:args.revalidate]
        extra_base = revalidate(pool, args.workers, models)
        for r in pool:
            r["score_first_pass"] = r["score"]
            r["score"] = score({**r["delay"], **r["extra_delay"]}, {**base, **extra_base})
        criterion += f"; the {len(pool)} best re-run on seeds {', '.join(map(str, EXTRA_SEEDS))} and scored on all seeds"
    best = min(pool, key=lambda r: r["score"])
    shutil.copyfile(os.path.join(models, "checkpoints", best["ckpt"]), os.path.join(models, "chowkpilot.npz"))
    with open(os.path.join(models, "selection.json"), "w", encoding="utf-8") as f:
        json.dump({"selected": {k: best[k] for k in ("ckpt", "updates", "minutes", "score")}, "criterion": criterion,
                   "extra_actuated_delay": extra_base, "candidates": rows}, f, indent=1)
    for r in pool:
        first = f"  first pass={r['score_first_pass']:.3f}" if "score_first_pass" in r else ""
        mark = "  <- selected" if r is best else ""
        print(f"{r['ckpt']}  updates={r['updates']:6d}  score={r['score']:.3f}{first}{mark}")


if __name__ == "__main__":
    main()
