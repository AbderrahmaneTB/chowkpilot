"""Build a SUMO digital twin of a real junction cluster from OpenStreetMap.

Data © OpenStreetMap contributors, available under the Open Database License (ODbL).
"""

from __future__ import annotations

import os
import ssl
import subprocess
import urllib.parse
import urllib.request

from ..sumo_paths import binary, data

OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "ChowkPilot/1.0 (NeuralStorm 2026 hackathon research prototype)"
ROAD_CLASSES = ("motorway|trunk|primary|secondary|tertiary|unclassified|"
                "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")


def download(bbox: tuple[float, float, float, float], out_path: str, timeout: int = 120) -> str:
    """bbox = (south, west, north, east). Saves the OSM XML of the main road network."""
    s, w, n, e = bbox
    query = (f'[out:xml][timeout:{timeout}];'
             f'(way["highway"~"^({ROAD_CLASSES})$"]({s},{w},{n},{e}););(._;>;);out body;')
    req = urllib.request.Request(
        OVERPASS, data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": USER_AGENT, "Accept": "application/xml"})
    try:  # certifi's bundle avoids stale Windows/OpenSSL chains
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout + 30, context=ctx) as resp:
        payload = resp.read()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(payload)
    return out_path


def convert(osm_path: str, net_path: str, tls_layout: str = "incoming", extra: list[str] | None = None) -> str:
    """OSM -> SUMO network with Indian (left-hand) traffic and guessed signal plans."""
    cmd = [
        binary("netconvert"), "--osm-files", osm_path, "-o", net_path,
        "--type-files", data(os.path.join("typemap", "osmNetconvert.typ.xml")),
        "--lefthand", "true",
        "--geometry.remove", "true", "--roundabouts.guess", "true", "--ramps.guess", "true",
        "--junctions.join", "true", "--junctions.corner-detail", "5",
        "--edges.join", "true",
        "--tls.guess-signals", "true", "--tls.guess-signals.dist", "40",
        "--tls.discard-simple", "true", "--tls.join", "true",
        "--tls.default-type", "static", "--tls.layout", tls_layout,
        "--keep-edges.by-vclass", "passenger", "--remove-edges.isolated", "true",
        "--no-turnarounds", "true", "--osm.turn-lanes", "true",
        "--output.street-names", "true", "--output.original-names", "false",
        "--no-warnings", "true",
    ] + (extra or [])
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return net_path
