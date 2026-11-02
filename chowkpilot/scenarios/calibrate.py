"""Per-scenario calibration of nominal demand and of the Webster baseline.

Nominal (peak) demand, scale 1.0, is the level at which competent adaptive control
(SUMO's vehicle-actuated controller) reaches the onset of congestion: its mean delay
is 1.7x its free-flow value. Comparisons are therefore made near capacity, where
signal control matters, instead of in empty or totally gridlocked networks.
Webster's saturation headway is then tuned on a calibration seed that is never
used for testing, so that baseline is the strongest version of itself.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import registry

CAL_SEED = 999          # never used by the benchmark
BASE_RATE = 300.0       # veh/h per entry lane before calibration
SCALES = (0.3, 0.5, 0.7, 0.9, 1.1, 1.35, 1.6)
HEADWAYS = (1.8, 2.2, 2.6, 3.0)
CONGESTION_FACTOR = 1.7
HORIZON = 1200


def _run(sid, controller, demand, sim=None):
    from ..eval.runner import run_one
    return run_one(sid, controller, seed=CAL_SEED, horizon=HORIZON, demand=demand, sim=sim)


def calibrate_demand(sid: str) -> dict:
    points = []
    for s in SCALES:
        r = _run(sid, "actuated", {"per_lane_rate": BASE_RATE, "scale": s})
        frac_unf = r["unfinished"] / max(1, r["trips"])
        points.append((s, r["mean_delay"], frac_unf))
        target = CONGESTION_FACTOR * points[0][1]
        if r["mean_delay"] > 1.3 * target or frac_unf > 0.05:
            break
    xs = np.array([p[0] for p in points])
    ds = np.maximum.accumulate(np.array([p[1] for p in points]))  # monotone for interpolation
    target = CONGESTION_FACTOR * ds[0]
    best = xs[-1] if ds[-1] < target else float(np.interp(target, ds, xs))
    return {"id": sid, "per_lane_rate": round(BASE_RATE * best, 1), "points": points}


def calibrate_webster(sid: str, per_lane_rate: float) -> dict:
    best, scores = None, []
    for h in HEADWAYS:
        r = _run(sid, "webster", {"per_lane_rate": per_lane_rate}, {"webster_headway": h})
        scores.append((h, r["mean_delay"]))
        if best is None or r["mean_delay"] < best[1]:
            best = (h, r["mean_delay"])
    return {"id": sid, "webster_headway": best[0], "scores": scores}


def calibrate_all(workers: int = 6) -> list:
    scenarios = registry.load_all()
    with ProcessPoolExecutor(workers) as ex:
        dem = list(ex.map(calibrate_demand, [s.id for s in scenarios]))
        web = list(ex.map(calibrate_webster, [s.id for s in scenarios], [d["per_lane_rate"] for d in dem]))
    by_id = {d["id"]: d for d in dem}
    for w in web:
        by_id[w["id"]].update(w)
    path = os.path.join(registry.SCENARIO_DIR, "index.json")
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    for entry in index:
        c = by_id.get(entry["id"])
        if c:
            entry["per_lane_rate"] = c["per_lane_rate"]
            entry["webster_headway"] = c["webster_headway"]
            entry["calibration"] = {"reference": "actuated", "delay_by_scale": c["points"],
                                    "webster_scores": c["scores"]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return list(by_id.values())


if __name__ == "__main__":
    for row in calibrate_all():
        print(row["id"], row["per_lane_rate"], row["webster_headway"], flush=True)
