"""Select the deployed policy among training checkpoints, using validation runs only.

    python -m chowkpilot.rl.select [--revalidate 3 [--ensemble]] [--fill N] [--workers 10] [--models models]

Each validated checkpoint is scored by its mean delay relative to vehicle-actuated control
on the validation tasks (training layouts, held-out seeds; equal weight per layout,
seeds averaged). One seed per task is noisy, so with --revalidate K the K best
checkpoints are run again on extra held-out seeds of the same layouts (actuated control
runs on the same seeds as the reference) and the final choice uses every seed. With
--ensemble, the ensemble of those K checkpoints (mean phase score) competes on the same
seeds. The winner is copied to <models>/chowkpilot.npz. The real test junctions play no part in this choice.
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


def extra_tasks() -> list:
    """(scenario, seed, demand) on every validation layout for each extra seed."""
    from .train import VALIDATION
    layouts = {}
    for sid, _, demand in VALIDATION:
        layouts.setdefault(sid, demand)
    return [(sid, seed, demand) for sid, demand in layouts.items() for seed in EXTRA_SEEDS]


def run_tasks(jobs: list, workers: int) -> dict:
    """jobs of (scenario, controller, seed, demand, policy) -> {policy path or controller: {task: mean delay}}."""
    from .train import task_key
    with ProcessPoolExecutor(workers) as ex:
        delays = list(ex.map(_run, jobs))
    out = defaultdict(dict)
    for (sid, ctrl, seed, _, policy), d in zip(jobs, delays):
        out[policy or ctrl][task_key(sid, seed)] = d
    return out


def main():
    from .policy import NumpyPolicy
    from .train import VALIDATION

    ap = argparse.ArgumentParser()
    ap.add_argument("--revalidate", type=int, default=0, help="re-run the K best checkpoints on extra seeds")
    ap.add_argument("--ensemble", action="store_true",
                    help="with --revalidate: also score the ensemble of those K checkpoints on every validation seed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=MODELS, help="training output folder (train_log.json, checkpoints/)")
    ap.add_argument("--fill", type=int, default=0, metavar="N",
                    help="first run the validation tasks for every checkpoint at a multiple of N updates "
                         "that training did not validate (training only validates the latest checkpoint)")
    args = ap.parse_args()
    models = args.models
    ckpt = lambda name: os.path.join(models, "checkpoints", name)  # noqa: E731
    log = json.load(open(os.path.join(models, "train_log.json"), encoding="utf-8"))
    base = json.load(open(os.path.join(models, "baseline_validation.json"), encoding="utf-8"))["actuated"]
    rows = []
    for v in log["validation"]:
        sc = score(v["delay"], base)
        if sc is not None and v.get("ckpt"):
            rows.append({"ckpt": v["ckpt"], "updates": v["updates"], "minutes": v.get("minutes"), "score": sc,
                         "delay": v["delay"]})
    if args.fill:
        done = {r["ckpt"] for r in rows}
        todo = [f for f in sorted(os.listdir(os.path.join(models, "checkpoints")))
                if f.endswith(".npz") and f not in done and int(f[1:7]) % args.fill == 0]
        res = run_tasks([(sid, "chowkpilot", seed, dem, ckpt(f)) for f in todo for sid, seed, dem in VALIDATION],
                        args.workers) if todo else {}
        for f in todo:
            rows.append({"ckpt": f, "updates": int(f[1:7]), "minutes": None, "score": score(res[ckpt(f)], base),
                         "delay": res[ckpt(f)], "validated_after_training": True})
    if not rows:
        raise SystemExit("no complete validation rounds found")
    if args.ensemble and not args.revalidate:
        raise SystemExit("--ensemble needs --revalidate K (the ensemble members)")
    criterion = "mean validation delay / vehicle-actuated delay (training layouts, held-out seeds)"
    extra_base, pool = None, rows
    if args.revalidate:
        pool = sorted(rows, key=lambda r: r["score"])[:args.revalidate]
        tasks = extra_tasks()
        jobs = [(sid, "actuated", seed, dem, None) for sid, seed, dem in tasks]
        jobs += [(sid, "chowkpilot", seed, dem, ckpt(r["ckpt"])) for r in pool for sid, seed, dem in tasks]
        ens_path = os.path.join(models, "ensemble.npz")
        if args.ensemble:
            np.savez(ens_path, **NumpyPolicy.ensemble_params([ckpt(r["ckpt"]) for r in pool]))
            jobs += [(sid, "chowkpilot", seed, dem, ens_path) for sid, seed, dem in list(VALIDATION) + tasks]
        res = run_tasks(jobs, args.workers)
        extra_base = res["actuated"]
        every = {**base, **extra_base}
        for r in pool:
            r["extra_delay"] = res[ckpt(r["ckpt"])]
            r["score_first_pass"] = r["score"]
            r["score"] = score({**r["delay"], **r["extra_delay"]}, every)
        criterion += f"; the {len(pool)} best re-run on seeds {', '.join(map(str, EXTRA_SEEDS))} and scored on all seeds"
        if args.ensemble:
            ens = {"ckpt": "ensemble.npz", "members": [r["ckpt"] for r in pool], "updates": None, "minutes": None,
                   "score": score(res[ens_path], every), "delay": res[ens_path]}
            rows.append(ens)
            pool = pool + [ens]
            criterion += "; their ensemble (mean phase score) is scored on the same seeds"
    best = min(pool, key=lambda r: r["score"])
    src = os.path.join(models, best["ckpt"]) if best.get("members") else ckpt(best["ckpt"])
    shutil.copyfile(src, os.path.join(models, "chowkpilot.npz"))
    with open(os.path.join(models, "selection.json"), "w", encoding="utf-8") as f:
        json.dump({"selected": {k: best[k] for k in ("ckpt", "members", "updates", "minutes", "score") if k in best},
                   "criterion": criterion, "extra_actuated_delay": extra_base, "candidates": rows}, f, indent=1)
    for r in pool:
        first = f"  first pass={r['score_first_pass']:.3f}" if "score_first_pass" in r else ""
        name = r["ckpt"] + (f" ({'+'.join(r['members'])})" if r.get("members") else f"  updates={r['updates']:6d}")
        print(f"{name}  score={r['score']:.3f}{first}{'  <- selected' if r is best else ''}")


if __name__ == "__main__":
    main()
