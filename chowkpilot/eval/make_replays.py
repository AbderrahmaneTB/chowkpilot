"""Record the replays shown in the web control room.

    python -m chowkpilot.eval.make_replays [--only hyd_hitech] [--controllers fixed,chowkpilot]
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from .replay import REPLAY_DIR, record, save_replay

DEMO = ["hyd_secunderabad", "hyd_hitech", "hyd_ameerpet", "syn_arterial3", "syn_grid3x3"]
CONTROLLERS = ["fixed", "webster", "actuated", "maxpressure", "chowkpilot"]
DEMAND = {"scale": 1.0, "profile": "peak", "ambulances_per_hour": 6}
SEED, DURATION = 3, 900


def _one(args):
    sid, ctrl = args
    t0 = time.time()
    res = record(sid, ctrl, seed=SEED, duration=DURATION, demand=DEMAND,
                 sim={"ev_priority": ctrl == "chowkpilot"})
    res.pop("rows", None)
    path = os.path.join(REPLAY_DIR, f"{sid}__{ctrl}.json.gz")
    save_replay(res, path)
    return f"{sid:18s} {ctrl:12s} delay={res['mean_delay']:6.1f}s frames={len(res['frames'])} " \
           f"size={os.path.getsize(path) / 1e6:.1f}MB {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--controllers", default=",".join(CONTROLLERS))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    scen = [s for s in DEMO if not args.only or s in args.only.split(",")]
    jobs = [(s, c) for s in scen for c in args.controllers.split(",")]
    with ProcessPoolExecutor(args.workers) as ex:
        for line in ex.map(_one, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
