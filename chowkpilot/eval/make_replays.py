"""Record the replays shown in the control room (real SUMO runs, 1 Hz).

    python -m chowkpilot.eval.make_replays [--only hyd_hitech] [--skip-conditions]

Peak-hour replays for every demo junction and controller, plus a conditions
explorer (off-peak, heavy, surge, camera noise) on the unseen Hyderabad junctions.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from .replay import REPLAY_DIR, record, save_replay

TEST = ["hyd_secunderabad", "hyd_hitech", "hyd_ameerpet"]
DEMO = TEST + ["syn_arterial3", "syn_grid3x3"]
PEAK_CONTROLLERS = {sid: ["fixed", "webster", "actuated", "maxpressure", "chowkpilot"] for sid in TEST}
PEAK_CONTROLLERS.update({sid: ["fixed", "actuated", "maxpressure", "chowkpilot"] for sid in DEMO[3:]})
CONDITIONS = {
    "offpeak": ("Off-peak (0.6× demand)", {"scale": 0.6}, {}),
    "heavy": ("Heavy (1.25× demand)", {"scale": 1.25}, {}),
    "surge": ("Sudden surge", {"profile": "surge"}, {}),
    "noise": ("Camera noise (20% count error, 10% missed)", {}, {"noise_sigma": 0.2, "miss_prob": 0.1}),
}
CONDITION_CONTROLLERS = ["fixed", "chowkpilot"]
BASE_DEMAND = {"scale": 1.0, "profile": "peak", "ambulances_per_hour": 6}
SEED = 3


def _one(job):
    sid, ctrl, cond, duration = job
    demand, sim = dict(BASE_DEMAND), {}
    if cond:
        _, d, s = CONDITIONS[cond]
        demand.update(d)
        sim.update(s)
    if ctrl == "chowkpilot":
        sim["ev_priority"] = True
    t0 = time.time()
    res = record(sid, ctrl, seed=SEED, duration=duration, demand=demand, sim=sim)
    res["condition"] = cond or "peak"
    name = f"{sid}__{ctrl}" + (f"__{cond}" if cond else "") + ".json.gz"
    path = os.path.join(REPLAY_DIR, name)
    save_replay(res, path)
    return f"{name:44s} delay={res['mean_delay']:6.1f}s size={os.path.getsize(path) / 1e6:.1f}MB {time.time() - t0:.0f}s"


def jobs(only=None, conditions=True):
    out = []
    for sid in DEMO:
        if only and sid not in only:
            continue
        out += [(sid, c, None, 900) for c in PEAK_CONTROLLERS[sid]]
        if conditions and sid in TEST:
            out += [(sid, c, cond, 600) for cond in CONDITIONS for c in CONDITION_CONTROLLERS]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-conditions", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    todo = jobs(args.only.split(",") if args.only else None, not args.skip_conditions)
    os.makedirs(REPLAY_DIR, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for line in ex.map(_one, todo):
            print(line, flush=True)


if __name__ == "__main__":
    main()
