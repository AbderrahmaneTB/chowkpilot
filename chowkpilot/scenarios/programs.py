"""Signal programs for the native-SUMO baselines (vehicle-actuated and Webster)."""

from __future__ import annotations

import os
import subprocess
import sys

import sumolib

from ..env.junction import is_green_phase
from ..sumo_paths import tool


def write_actuated(net_path: str, out_path: str, min_green: int = 10, max_green: int = 60) -> str:
    """Gap-based actuated version of each junction's own phase sequence (SUMO 'actuated' type)."""
    net = sumolib.net.readNet(net_path, withPrograms=True)
    lines = ["<additional>"]
    for tls in net.getTrafficLights():
        progs = tls.getPrograms()
        if not progs:
            continue
        prog = progs.get("0") or next(iter(progs.values()))
        lines.append(f'    <tlLogic id="{tls.getID()}" type="actuated" programID="actuated" offset="0">')
        lines.append('        <param key="max-gap" value="3.0"/>')
        lines.append('        <param key="detector-gap" value="2.0"/>')
        lines.append('        <param key="passing-time" value="1.9"/>')
        for ph in prog.getPhases():
            if is_green_phase(ph.state):
                lines.append(f'        <phase duration="{max(min_green, int(ph.duration))}" minDur="{min_green}" '
                             f'maxDur="{max_green}" state="{ph.state}"/>')
            else:
                lines.append(f'        <phase duration="{int(ph.duration)}" state="{ph.state}"/>')
        lines.append("    </tlLogic>")
    lines.append("</additional>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def write_webster(net_path: str, route_path: str, out_path: str, sat_headway: float = 1.6,
                  min_cycle: int = 40, max_cycle: int = 180) -> str:
    """Webster-optimised cycle and splits from the actual route file (SUMO tlsCycleAdaptation).

    The saturation headway of 1.6 s (2250 veh/h/lane) reflects a two-wheeler-heavy
    stream discharging side by side; this baseline is given perfect demand knowledge.
    """
    cmd = [sys.executable, tool("tlsCycleAdaptation.py"), "-n", net_path, "-r", route_path,
           "-o", out_path, "-p", "webster", "-b", "0", "-y", "4", "-a", "0", "-g", "10",
           "--min-cycle", str(min_cycle), "--max-cycle", str(max_cycle), "-H", str(sat_headway),
           "--sorted"]
    subprocess.run(cmd, check=True, capture_output=True, text=True,
                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return out_path
