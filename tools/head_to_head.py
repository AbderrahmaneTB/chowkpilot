"""Compare two policies on fresh validation seeds (training layouts only; never the test junctions).

    python tools/head_to_head.py models/run4/chowkpilot.npz models/chowkpilot.npz [out.json]

Used on 3 Nov to choose between run 4 and run 5b. The rule was fixed before running: the lower score wins, and a
gap below 0.01 keeps the simpler model. Score = mean over layouts of delay / vehicle-actuated delay.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chowkpilot.rl.select import run_tasks, score  # noqa: E402
from chowkpilot.rl.train import VALIDATION  # noqa: E402

SEEDS = (9191, 10101, 11111)  # used nowhere else: not in training, validation, selection or the benchmark


def main():
    a, b = sys.argv[1], sys.argv[2]
    layouts = {}
    for sid, _, demand in VALIDATION:
        layouts.setdefault(sid, demand)
    tasks = [(sid, seed, demand) for sid, demand in layouts.items() for seed in SEEDS]
    jobs = [(sid, "actuated", seed, demand, None) for sid, seed, demand in tasks]
    jobs += [(sid, "chowkpilot", seed, demand, p) for p in (a, b) for sid, seed, demand in tasks]
    res = run_tasks(jobs, int(os.environ.get("WORKERS", "10")))
    base = res["actuated"]
    out = {"seeds": SEEDS, "rule": "lower score wins; a gap below 0.01 keeps the simpler model", "actuated": base}
    for p in (a, b):
        per = {}
        for k, ref in base.items():
            per.setdefault(k.split("@")[0], []).append(res[p][k] / ref)
        out[p] = {"score": score(res[p], base), "per_layout": {k: sum(v) / len(v) for k, v in per.items()},
                  "delay": res[p]}
        print(p, f"{out[p]['score']:.3f}", {k: round(v, 3) for k, v in out[p]["per_layout"].items()})
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
