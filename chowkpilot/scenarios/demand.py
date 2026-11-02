"""Seeded, explicit traffic demand for a SUMO network.

Every vehicle is written with its own departure time, type and route, so the same
(scenario, seed) gives every controller exactly the same vehicles. That pairing
is what makes the benchmark comparisons fair and low-variance.

Arrivals at each entry road follow a non-homogeneous Poisson process whose rate
follows a time-of-day profile (e.g. a peak in the middle of the hour).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import sumolib

from .vtypes import INDIAN_MIX, vtypes_xml

PROFILES = {
    # multiplier of the nominal rate as a function of normalised time u in [0, 1]
    "flat": lambda u: 1.0,
    "peak": lambda u: 0.65 + 0.45 * math.exp(-((u - 0.5) / 0.22) ** 2),
    "ramp": lambda u: 0.55 + 0.6 * u,
    "surge": lambda u: 1.35 if 0.35 <= u <= 0.7 else 0.8,
    "wave": lambda u: 0.85 + 0.3 * math.sin(2 * math.pi * 2 * u),
}


@dataclass
class DemandSpec:
    horizon: int = 3600              # seconds during which vehicles are generated
    scale: float = 1.0               # global multiplier of the nominal demand
    per_lane_rate: float = 420.0     # nominal veh/h entering per entry lane
    profile: str = "peak"
    imbalance: float = 0.45          # spread of per-entry demand (0 = perfectly balanced)
    major_bias: float = 1.0          # extra weight for high-priority roads (OSM arterials)
    ambulances_per_hour: float = 0.0
    mix: dict = field(default_factory=lambda: dict(INDIAN_MIX))
    seed: int = 0


def _reverse_of(edge, candidates):
    to_node, from_node = edge.getToNode(), edge.getFromNode()
    for c in candidates:
        if c.getFromNode() == to_node and c.getToNode() == from_node:
            return c
    return None


MIN_ENTRY_LENGTH = 25.0  # vehicles cannot be inserted on stub edges
EXCLUDED_TYPES = ("highway.motorway",)  # grade-separated ring road: not part of the junctions


def entry_exit_edges(net):
    """Edges where traffic can enter (no upstream) and leave (no downstream)."""
    edges = [e for e in net.getEdges() if e.getFunction() == "" and e.allows("passenger")]
    sources, sinks = [], []
    for e in edges:
        if e.getLength() < MIN_ENTRY_LENGTH or e.getType() in EXCLUDED_TYPES:
            continue
        rev = _reverse_of(e, edges)
        incoming = [i for i in e.getIncoming() if i is not rev and i.allows("passenger")]
        outgoing = [o for o in e.getOutgoing() if o is not rev and o.allows("passenger")]
        if not incoming and outgoing:
            sources.append(e)
        if not outgoing and incoming:
            sinks.append(e)
    return sources, sinks


def build_routes(net, sources, sinks):
    """Shortest (fastest) route for every reachable entry->exit pair."""
    routes = {}
    for s in sources:
        rev = _reverse_of(s, sinks)
        for t in sinks:
            if t is rev:
                continue  # no U-turn trips back out where they came in
            path, _cost = net.getFastestPath(s, t, vClass="passenger")
            if path:
                routes[(s.getID(), t.getID())] = [e.getID() for e in path]
    return routes


def generate(net_path: str, out_path: str, spec: DemandSpec) -> dict:
    """Write a .rou.xml with explicit vehicles; returns summary statistics."""
    rng = np.random.default_rng(spec.seed)
    net = sumolib.net.readNet(net_path)
    sources, sinks = entry_exit_edges(net)
    routes = build_routes(net, sources, sinks)
    if not routes:
        raise RuntimeError(f"no routable entry/exit pairs in {net_path}")

    prio = np.array([s.getPriority() for s in sources], dtype=float)
    prio_w = 1.0 + spec.major_bias * (prio - prio.min()) / max(1.0, prio.max() - prio.min())
    src_rate = np.array([
        s.getLaneNumber() * spec.per_lane_rate * w
        * float(np.exp(rng.normal(0.0, spec.imbalance)))
        for s, w in zip(sources, prio_w)
    ]) * spec.scale  # veh/h per entry edge
    sink_w = {t.getID(): t.getLaneNumber() * float(np.exp(rng.normal(0.0, 0.35))) for t in sinks}

    profile = PROFILES[spec.profile]
    types = list(spec.mix)
    type_p = np.array([spec.mix[k] for k in types], dtype=float)
    type_p /= type_p.sum()

    vehicles = []  # (depart, type, route_key)
    for s, rate in zip(sources, src_rate):
        dests = [(k, sink_w[k[1]]) for k in routes if k[0] == s.getID()]
        if not dests:
            continue
        keys = [d[0] for d in dests]
        w = np.array([d[1] for d in dests])
        w /= w.sum()
        # thinning: sample a homogeneous process at the peak rate, keep with p = rate(t)/peak
        peak = rate * max(profile(u) for u in np.linspace(0, 1, 101)) / 3600.0
        t = 0.0
        while True:
            t += rng.exponential(1.0 / peak)
            if t >= spec.horizon:
                break
            if rng.random() <= rate * profile(t / spec.horizon) / 3600.0 / peak:
                k = keys[rng.choice(len(keys), p=w)]
                vt = types[rng.choice(len(types), p=type_p)]
                vehicles.append((t, vt, k))

    if spec.ambulances_per_hour > 0:
        n_amb = rng.poisson(spec.ambulances_per_hour * spec.horizon / 3600.0)
        keys = list(routes)
        for _ in range(n_amb):
            # keep ambulances away from the warm-up and the very end of the horizon
            t = rng.uniform(0.1 * spec.horizon, 0.85 * spec.horizon)
            vehicles.append((t, "ambulance", keys[rng.integers(len(keys))]))

    vehicles.sort(key=lambda v: v[0])
    route_ids = {k: f"r{i}" for i, k in enumerate(routes)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("<routes>\n")
        f.write(vtypes_xml(spec.mix) + "\n")
        for k, rid in route_ids.items():
            f.write(f'    <route id="{rid}" edges="{" ".join(routes[k])}"/>\n')
        for i, (t, vt, k) in enumerate(vehicles):
            f.write(f'    <vehicle id="{vt[:2]}{i}" type="{vt}" route="{route_ids[k]}" '
                    f'depart="{t:.1f}" departLane="best" departSpeed="max" departPosLat="random"/>\n')
        f.write("</routes>\n")

    counts = {}
    for _, vt, _ in vehicles:
        counts[vt] = counts.get(vt, 0) + 1
    return {"vehicles": len(vehicles), "by_type": counts, "entries": len(sources),
            "exits": len(sinks), "routes": len(routes),
            "veh_per_hour": len(vehicles) * 3600.0 / spec.horizon}
