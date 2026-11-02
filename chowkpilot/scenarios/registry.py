"""Scenario catalogue: synthetic training layouts and real Hyderabad digital twins.

Real areas used for testing are never seen during training, so results on them
measure zero-shot generalisation to new junctions.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass

from ..sumo_paths import binary

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENARIO_DIR = os.path.join(ROOT, "scenarios")


@dataclass
class Scenario:
    id: str
    title: str
    split: str                 # "train" or "test"
    kind: str                  # "synthetic" or "osm"
    net: str                   # path relative to SCENARIO_DIR
    per_lane_rate: float = 420.0
    major_bias: float = 1.0
    description: str = ""
    bbox: tuple | None = None  # (south, west, north, east) for OSM areas
    webster_headway: float = 2.2
    calibration: dict | None = None

    @property
    def net_path(self) -> str:
        return os.path.join(SCENARIO_DIR, self.net)


SYNTHETIC = [
    # id, generator kind, options, per-lane demand
    ("syn_single_1lane", "grid", dict(x=1, y=1, lanes=1, turn=0, layout="incoming"), 380),
    ("syn_single_2lane_opp", "grid", dict(x=1, y=1, lanes=2, turn=1, layout="opposites"), 420),
    ("syn_single_3lane", "grid", dict(x=1, y=1, lanes=3, turn=0, layout="incoming"), 400),
    ("syn_single_2lane_inc", "grid", dict(x=1, y=1, lanes=2, turn=1, layout="incoming"), 380),
    ("syn_arterial3", "grid", dict(x=3, y=1, lanes=2, turn=1, layout="opposites", length=260), 380),
    ("syn_arterial4_inc", "grid", dict(x=4, y=1, lanes=2, turn=0, layout="incoming", length=320), 340),
    ("syn_grid2x2", "grid", dict(x=2, y=2, lanes=2, turn=0, layout="opposites", length=300), 330),
    ("syn_grid3x3", "grid", dict(x=3, y=3, lanes=1, turn=0, layout="opposites", length=240), 300),
    ("syn_spider5", "spider", dict(arms=5, circles=1, radius=260, lanes=2, layout="incoming"), 300),
    ("syn_random_a", "rand", dict(seed=3, iterations=14, lanes=2, layout="opposites"), 300),
    ("syn_random_b", "rand", dict(seed=11, iterations=18, lanes=1, layout="incoming"), 300),
]

# Real Hyderabad areas (OpenStreetMap). bbox = (south, west, north, east)
OSM_AREAS = [
    ("hyd_gachibowli", "Gachibowli / Financial District", "train", (17.4120, 78.3340, 17.4290, 78.3600), 360),
    ("hyd_jubilee", "Jubilee Hills", "train", (17.4150, 78.4100, 17.4280, 78.4300), 360),
    ("hyd_abids", "Abids / Nampally", "train", (17.3900, 78.4700, 17.4050, 78.4900), 360),
    ("hyd_secunderabad", "Secunderabad: S.D. Road, Patny & YMCA", "test", (17.4355, 78.4925, 17.4465, 78.5085), 380),
    ("hyd_hitech", "Hitech City: Cyber Towers junction", "test", (17.4470, 78.3760, 17.4560, 78.3870), 420),
    ("hyd_ameerpet", "Ameerpet / Begumpet", "test", (17.4330, 78.4380, 17.4450, 78.4500), 380),
]


def netgenerate(out_path: str, kind: str, o: dict) -> str:
    cmd = [binary("netgenerate"), "-o", out_path, "--lefthand", "true", "--no-turnarounds", "true",
           "--default-junction-type", "traffic_light", "--tls.layout", o.get("layout", "opposites"),
           "--default.speed", "13.89", "--default.lanenumber", str(o.get("lanes", 2)),
           "--junctions.corner-detail", "5", "--no-warnings", "true"]
    if o.get("turn"):
        cmd += ["--turn-lanes", str(o["turn"]), "--turn-lanes.length", "60"]
    if kind == "grid":
        length = o.get("length", 300)
        cmd += ["--grid", "--grid.x-number", str(o["x"]), "--grid.y-number", str(o["y"]),
                "--grid.x-length", str(length), "--grid.y-length", str(length),
                "--grid.attach-length", str(o.get("attach", 250))]
    elif kind == "spider":
        cmd += ["--spider", "--spider.arm-number", str(o["arms"]), "--spider.circle-number", str(o["circles"]),
                "--spider.space-radius", str(o["radius"]), "--spider.attach-length", "220"]
    elif kind == "rand":
        cmd += ["--rand", "--seed", str(o["seed"]), "--rand.iterations", str(o["iterations"]),
                "--rand.min-distance", "180", "--rand.max-distance", "320", "--rand.min-angle", "50",
                "--rand.connectivity", "0.9", "--rand.neighbor-dist1", "0", "--rand.neighbor-dist2", "0"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


def build_all(include_osm: bool = True, refresh_osm: bool = False) -> list[Scenario]:
    from . import osm

    os.makedirs(SCENARIO_DIR, exist_ok=True)
    scenarios = []
    for sid, kind, opts, rate in SYNTHETIC:
        rel = f"synthetic/{sid}.net.xml"
        path = os.path.join(SCENARIO_DIR, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        netgenerate(path, kind, opts)
        scenarios.append(Scenario(sid, sid.replace("_", " "), "train", "synthetic", rel, per_lane_rate=rate))
    if include_osm:
        import time
        for sid, title, split, bbox, rate in OSM_AREAS:
            raw = os.path.join(SCENARIO_DIR, "osm", f"{sid}.osm.xml")
            rel = f"osm/{sid}.net.xml"
            if refresh_osm or not os.path.exists(raw):
                for attempt in range(4):
                    try:
                        osm.download(bbox, raw)
                        break
                    except Exception:
                        time.sleep(10 * (attempt + 1))
                else:
                    print(f"skipping {sid}: download failed")
                    continue
            osm.convert(raw, os.path.join(SCENARIO_DIR, rel))
            scenarios.append(Scenario(sid, title, split, "osm", rel, per_lane_rate=rate, major_bias=1.5,
                                      bbox=bbox, description="© OpenStreetMap contributors (ODbL)"))
    with open(os.path.join(SCENARIO_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in scenarios], f, indent=2)
    return scenarios


def load_all() -> list[Scenario]:
    with open(os.path.join(SCENARIO_DIR, "index.json"), encoding="utf-8") as f:
        return [Scenario(**{**d, "bbox": tuple(d["bbox"]) if d.get("bbox") else None}) for d in json.load(f)]


def get(sid: str) -> Scenario:
    for s in load_all():
        if s.id == sid:
            return s
    raise KeyError(sid)
