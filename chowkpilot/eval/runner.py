"""Run one (scenario, controller, seed) episode end to end and return its metrics."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import replace

from ..control.controllers import BASELINES, ChowkPilot
from ..env.sim import NetworkModel, SimConfig, Simulation
from ..scenarios import programs, registry
from ..scenarios.demand import DemandSpec, generate

ROOT = registry.ROOT
DEFAULT_POLICY = os.path.join(ROOT, "models", "chowkpilot.npz")
_NETWORKS: dict[str, NetworkModel] = {}


def network(scenario) -> NetworkModel:
    if scenario.id not in _NETWORKS:
        _NETWORKS[scenario.id] = NetworkModel(scenario.net_path)
    return _NETWORKS[scenario.id]


def make_controller(name: str, policy: str | None = None, explain: bool = False):
    if name == "chowkpilot":
        return ChowkPilot(policy or DEFAULT_POLICY, explain=explain)
    return BASELINES[name]()


def demand_for(scenario, seed: int, horizon: int, **overrides) -> DemandSpec:
    spec = DemandSpec(horizon=horizon, per_lane_rate=scenario.per_lane_rate,
                      major_bias=scenario.major_bias, seed=seed)
    return replace(spec, **overrides)


def run_one(scenario_id: str, controller: str, seed: int = 0, horizon: int = 3600,
            demand: dict | None = None, sim: dict | None = None, policy: str | None = None,
            explain: bool = False, keep_frames: bool = False) -> dict:
    scenario = registry.get(scenario_id)
    nm = network(scenario)
    workdir = tempfile.mkdtemp(prefix="chowk_run_")
    try:
        spec = demand_for(scenario, seed, horizon, **(demand or {}))
        routes = os.path.join(workdir, "routes.rou.xml")
        dstats = generate(scenario.net_path, routes, spec)
        sim = dict(sim or {})
        headway = sim.pop("webster_headway", scenario.webster_headway)
        cfg = replace(SimConfig(), **sim)
        additional = []
        if controller == "actuated":
            additional.append(programs.write_actuated(scenario.net_path, os.path.join(workdir, "act.add.xml"),
                                                      cfg.min_green, cfg.max_green))
        elif controller == "webster":
            additional.append(programs.write_webster(scenario.net_path, routes,
                                                     os.path.join(workdir, "webster.add.xml"),
                                                     sat_headway=headway))
        ctrl = make_controller(controller, policy, explain)
        s = Simulation(nm, routes, ctrl, cfg, seed=seed, horizon=horizon,
                       additional=additional, workdir=workdir)
        res = s.run()
        res.update(scenario=scenario_id, controller=controller, seed=seed, horizon=horizon,
                   demand=dstats, ev_priority=cfg.ev_priority)
        if keep_frames:
            res["frames"] = s.frames
            res["explain"] = s.explain
        return res
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
