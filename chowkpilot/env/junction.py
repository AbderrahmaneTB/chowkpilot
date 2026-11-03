"""Topology-agnostic description of a signalised junction.

Any SUMO traffic light, from a simple 4-arm crossing to a joined OSM cluster with
dozens of links, is reduced to:

* movements: (incoming road -> outgoing road) pairs with their lanes and signal links,
* green phases: the candidate actions, each a set of movements that may proceed,
* mask[p, m]: whether movement m has green in phase p.

Controllers only ever see per-movement features and this mask, which is why a
single learned policy can drive junctions it has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GREEN_CHARS = ("G", "g")
NEAR_RANGE = 50.0  # metres before the stop line counted as "arriving soon"


@dataclass
class Movement:
    in_edge: str
    out_edge: str
    in_lanes: list
    out_lanes: list
    links: list


def is_green_phase(state: str) -> bool:
    return "y" not in state and "Y" not in state and any(c in GREEN_CHARS for c in state)


class JunctionModel:
    def __init__(self, tls, program_states: list[str], detect_range: float = 150.0):
        """tls: sumolib TLS object; program_states: phase strings of its static program."""
        self.id = tls.getID()
        self.detect_range = detect_range
        conns = tls.getConnections()  # [inLane, outLane, linkIndex]
        self.num_links = max(c[2] for c in conns) + 1

        greens = []
        for s in program_states:
            if is_green_phase(s) and s not in greens:
                greens.append(s)
        if not greens:
            raise ValueError(f"traffic light {self.id} has no green phase")
        self.green_states = greens

        groups: dict[tuple, Movement] = {}
        for in_lane, out_lane, idx in conns:
            key = (in_lane.getEdge().getID(), out_lane.getEdge().getID())
            mv = groups.setdefault(key, Movement(key[0], key[1], [], [], []))
            if in_lane.getID() not in mv.in_lanes:
                mv.in_lanes.append(in_lane.getID())
            if out_lane.getID() not in mv.out_lanes:
                mv.out_lanes.append(out_lane.getID())
            mv.links.append(idx)
        self.movements = list(groups.values())

        self.mask = np.zeros((len(greens), len(self.movements)), dtype=bool)
        for p, s in enumerate(greens):
            for m, mv in enumerate(self.movements):
                self.mask[p, m] = any(s[i] in GREEN_CHARS for i in mv.links)
        # drop phases that release nothing (can appear in odd OSM programs)
        keep = self.mask.any(axis=1)
        self.green_states = [s for s, k in zip(greens, keep) if k]
        self.mask = self.mask[keep]

        self.in_lanes = sorted({l for mv in self.movements for l in mv.in_lanes})
        self.out_lanes = sorted({l for mv in self.movements for l in mv.out_lanes})
        self.lane_index = {l: i for i, l in enumerate(self.in_lanes)}
        self.out_index = {l: i for i, l in enumerate(self.out_lanes)}
        # movement -> lane index lists, for vectorised feature aggregation
        self.mv_in = [[self.lane_index[l] for l in mv.in_lanes] for mv in self.movements]
        self.mv_out = [[self.out_index[l] for l in mv.out_lanes] for mv in self.movements]

        self.lanes_geo = {}  # in-lane -> (length, chain of lanes covered by the detector)
        self._lane_objs = {c[0].getID(): c[0] for c in conns}

    @property
    def num_phases(self) -> int:
        return len(self.green_states)

    @property
    def num_movements(self) -> int:
        return len(self.movements)

    def transition_states(self, cur: int, nxt: int) -> tuple[str, str]:
        """(yellow, all-red) signal strings for switching from phase cur to nxt."""
        a, b = self.green_states[cur], self.green_states[nxt]
        yellow, red = [], []
        for ca, cb in zip(a, b):
            ga, gb = ca in GREEN_CHARS, cb in GREEN_CHARS
            if ga and gb:
                yellow.append(ca)
                red.append(ca)
            elif ga:
                yellow.append("y")
                red.append("r")
            else:
                yellow.append("r")
                red.append("r")
        return "".join(yellow), "".join(red)

    def detector_xml(self, net) -> list[str]:
        """Lane-area (E2) detectors covering the last `detect_range` metres before each stop line.

        If the approach lane is shorter than the range, the zone is extended upstream along the
        straight-ahead predecessor lanes, mimicking a camera that sees the whole queue.
        """
        out = []
        for lane_id in self.in_lanes:
            lane = net.getLane(lane_id)
            chain, covered = [lane], lane.getLength()
            cur = lane
            while covered < self.detect_range:
                preds = [c for c in cur.getIncoming() if c.getEdge().getFunction() == ""]
                if not preds:
                    break
                straight = [p for p in preds if _direction(p, cur) == "s"]
                prev = (straight or preds)[0]
                if prev in chain:
                    break
                chain.insert(0, prev)
                covered += prev.getLength()
                cur = prev
            det_id = f"e2_{lane_id}"
            if len(chain) == 1:
                start = max(0.0, lane.getLength() - self.detect_range)
                out.append(f'    <laneAreaDetector id="{det_id}" lane="{lane_id}" pos="{start:.2f}" '
                           f'endPos="{lane.getLength():.2f}" friendlyPos="true" freq="100000" file="NUL"/>')
            else:
                excess = max(0.0, covered - self.detect_range)
                lanes = " ".join(l.getID() for l in chain)
                out.append(f'    <laneAreaDetector id="{det_id}" lanes="{lanes}" pos="{excess:.2f}" '
                           f'endPos="{lane.getLength():.2f}" friendlyPos="true" freq="100000" file="NUL"/>')
            # the same camera's view of the stop line (gap since the last crossing) and of the
            # last NEAR_RANGE metres (vehicles about to arrive): what actuated control reacts to
            near = max(0.0, lane.getLength() - NEAR_RANGE)
            out.append(f'    <inductionLoop id="e1_{lane_id}" lane="{lane_id}" pos="{max(0.0, lane.getLength() - 0.5):.2f}" '
                       f'friendlyPos="true" freq="100000" file="NUL"/>')
            out.append(f'    <laneAreaDetector id="e2n_{lane_id}" lane="{lane_id}" pos="{near:.2f}" '
                       f'endPos="{lane.getLength():.2f}" friendlyPos="true" freq="100000" file="NUL"/>')
            self.lanes_geo[lane_id] = (lane.getLength(), [l.getID() for l in chain])
        return out


def _direction(from_lane, to_lane) -> str:
    for c in from_lane.getOutgoing():
        if c.getToLane() is to_lane:
            return c.getDirection()
    return "?"
