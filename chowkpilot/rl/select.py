"""Select the deployed policy among training checkpoints, using validation runs only.

    python -m chowkpilot.rl.select

Each validated checkpoint is scored by its mean delay relative to vehicle-actuated control
on the validation tasks (training layouts, held-out seeds; equal weight per layout,
seeds averaged). The best one is copied to models/chowkpilot.npz. The real test
junctions play no part in this choice.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict

import numpy as np

from ..scenarios import registry

MODELS = os.path.join(registry.ROOT, "models")


def score(delays: dict, base: dict) -> float | None:
    if not all(k in delays for k in base):
        return None
    by_layout = defaultdict(list)
    for k, ref in base.items():
        by_layout[k.split("@")[0]].append(delays[k] / ref)
    return float(np.mean([np.mean(v) for v in by_layout.values()]))


def main():
    log = json.load(open(os.path.join(MODELS, "train_log.json"), encoding="utf-8"))
    base = json.load(open(os.path.join(MODELS, "baseline_validation.json"), encoding="utf-8"))["actuated"]
    rows = []
    for v in log["validation"]:
        sc = score(v["delay"], base)
        if sc is not None and v.get("ckpt"):
            rows.append({"ckpt": v["ckpt"], "updates": v["updates"], "minutes": v.get("minutes"), "score": sc})
    if not rows:
        raise SystemExit("no complete validation rounds found")
    best = min(rows, key=lambda r: r["score"])
    shutil.copyfile(os.path.join(MODELS, "checkpoints", best["ckpt"]), os.path.join(MODELS, "chowkpilot.npz"))
    with open(os.path.join(MODELS, "selection.json"), "w", encoding="utf-8") as f:
        json.dump({"selected": best, "criterion": "mean validation delay / vehicle-actuated delay "
                   "(training layouts, held-out seeds)", "candidates": rows}, f, indent=1)
    for r in rows:
        mark = "  <- selected" if r is best else ""
        print(f"{r['ckpt']}  updates={r['updates']:6d}  score={r['score']:.3f}{mark}")


if __name__ == "__main__":
    main()
