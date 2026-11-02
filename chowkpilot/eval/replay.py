"""Exports for the web control room: network geometry and recorded simulation replays.

Replays are real SUMO runs recorded at 1 Hz (vehicle positions, signal states,
queue series, and ChowkPilot's phase scores at every decision), not animations.
"""

from __future__ import annotations

import gzip
import json
import math
import os

import sumolib

from ..scenarios import registry
from ..scenarios.vtypes import VEHICLE_TYPES
from .runner import network, run_one

SITE_DATA = os.path.join(registry.ROOT, "docs", "data")  # published with GitHub Pages
REPLAY_DIR = os.path.join(SITE_DATA, "replays")


def _r(v: float) -> float:
    return round(float(v), 1)


def _approach_name(net, edge_id: str) -> str:
    edge = net.getEdge(edge_id)
    if edge.getName():
        return edge.getName()
    (x0, y0), (x1, y1) = edge.getShape()[-2], edge.getShape()[-1]
    heading = math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360  # direction of travel, 0 = north
    origin = ["south", "south-west", "west", "north-west", "north", "north-east", "east", "south-east"]
    return "from " + origin[int(((heading + 22.5) % 360) // 45)]


def _phase_labels(net, jm) -> list[str]:
    labels = []
    for p in range(jm.num_phases):
        names = []
        for m, mv in enumerate(jm.movements):
            if jm.mask[p, m]:
                n = _approach_name(net, mv.in_edge)
                if n not in names:
                    names.append(n)
        labels.append(" + ".join(names[:3]) + (f" (+{len(names) - 3})" if len(names) > 3 else ""))
    return labels


def network_geometry(scenario_id: str) -> dict:
    scenario = registry.get(scenario_id)
    nm = network(scenario)
    net = nm.net
    xmin, ymin, xmax, ymax = net.getBoundary()
    lanes = []
    for edge in net.getEdges():
        if edge.getFunction() == "internal":
            continue
        for lane in edge.getLanes():
            lanes.append({"s": [[_r(x), _r(y)] for x, y in lane.getShape()], "w": _r(lane.getWidth())})
    junctions = []
    for node in net.getNodes():
        shape = node.getShape()
        if len(shape) >= 3:
            junctions.append([[_r(x), _r(y)] for x, y in shape])
    signals = []
    for j, jm in enumerate(nm.junctions):
        by_lane = {}
        for mv in jm.movements:
            for lane_id in mv.in_lanes:
                by_lane.setdefault(lane_id, set())
        tls = net.getTLS(jm.id)
        for in_lane, _out, idx in tls.getConnections():
            if in_lane.getID() in by_lane:
                by_lane[in_lane.getID()].add(idx)
        for lane_id, links in by_lane.items():
            shape = net.getLane(lane_id).getShape()
            (x0, y0), (x1, y1) = shape[-2], shape[-1]
            signals.append({"j": j, "links": sorted(links), "x": _r(x1), "y": _r(y1),
                            "a": round(math.atan2(y1 - y0, x1 - x0), 3),
                            "w": _r(net.getLane(lane_id).getWidth())})
    names, seen = [], set()
    for edge in sorted(net.getEdges(), key=lambda e: -e.getLength()):
        name = edge.getName()
        if not name or name in seen or edge.getLength() < 120:
            continue
        seen.add(name)
        shape = edge.getShape()
        mid = shape[len(shape) // 2]
        (ax, ay), (bx, by) = shape[0], shape[-1]
        names.append({"n": name, "x": _r(mid[0]), "y": _r(mid[1]), "a": round(math.atan2(by - ay, bx - ax), 3)})
    junction_info = []
    for j, jm in enumerate(nm.junctions):
        pts = [s for s in signals if s["j"] == j]  # centroid of stop lines (works for joined clusters)
        x = sum(p["x"] for p in pts) / max(1, len(pts))
        y = sum(p["y"] for p in pts) / max(1, len(pts))
        streets = sorted({net.getEdge(mv.in_edge).getName() for mv in jm.movements
                          if net.getEdge(mv.in_edge).getName()})
        junction_info.append({"id": jm.id, "x": _r(x), "y": _r(y), "phases": jm.num_phases,
                              "movements": jm.num_movements, "streets": streets[:4],
                              "phase_labels": _phase_labels(net, jm)})
    return {"id": scenario.id, "title": scenario.title, "bounds": [_r(xmin), _r(ymin), _r(xmax), _r(ymax)],
            "lanes": lanes, "junctions": junctions, "signals": signals, "names": names[:14],
            "tls": junction_info, "attribution": scenario.description,
            "types": [{"name": k, "length": v["length"], "width": v["width"]} for k, v in VEHICLE_TYPES.items()]}


def record(scenario_id: str, controller: str, seed: int = 1, duration: int = 900,
           demand: dict | None = None, sim: dict | None = None, policy: str | None = None) -> dict:
    sim = {"record": True, "record_window": (0, duration), "max_extra_time": 900, **(sim or {})}
    res = run_one(scenario_id, controller, seed=seed, horizon=duration, demand=demand, sim=sim,
                  policy=policy, explain=controller == "chowkpilot", keep_frames=True)
    return res


def save_replay(res: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(res, f, separators=(",", ":"))


def load_replay(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
