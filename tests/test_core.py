"""Core tests: policy equivalence, deterministic demand, junction model and safety invariants.

    python -m pytest -q
"""

import os
import tempfile

import numpy as np
import pytest

from chowkpilot.env.sim import MOVEMENT_FEATURES, PHASE_FEATURES, NetworkModel, SimConfig, Simulation
from chowkpilot.rl.policy import NumpyPolicy
from chowkpilot.scenarios import registry
from chowkpilot.scenarios.demand import DemandSpec, generate

SCEN = "syn_single_2lane_inc"


def test_numpy_policy_matches_torch():
    torch = pytest.importorskip("torch")
    from chowkpilot.rl.policy import export_numpy, torch_model

    torch.manual_seed(0)
    model = torch_model(MOVEMENT_FEATURES, PHASE_FEATURES)
    npol = NumpyPolicy(export_numpy(model))
    rng = np.random.default_rng(0)
    M, P, Mpad, Ppad = 7, 3, 12, 5
    mv = rng.random((M, MOVEMENT_FEATURES)).astype(np.float32)
    ph = rng.random((P, PHASE_FEATURES)).astype(np.float32)
    mask = rng.random((P, M)) > 0.5
    mask[:, 0] = True
    q_np = npol.q_values(mv, ph, mask)
    # padded batch through the torch model must give the same scores
    tm = torch.zeros(1, Mpad, MOVEMENT_FEATURES)
    tm[0, :M] = torch.from_numpy(mv)
    tp = torch.zeros(1, Ppad, PHASE_FEATURES)
    tp[0, :P] = torch.from_numpy(ph)
    tk = torch.zeros(1, Ppad, Mpad)
    tk[0, :P, :M] = torch.from_numpy(mask.astype(np.float32))
    valid = torch.zeros(1, Mpad)
    valid[0, :M] = 1
    with torch.no_grad():
        q_t = model(tm, tp, tk, valid)[0, :P].numpy()
    np.testing.assert_allclose(q_np, q_t, rtol=1e-4, atol=1e-5)


def test_ensemble_scores_are_member_means():
    torch = pytest.importorskip("torch")
    from chowkpilot.rl.policy import export_numpy, torch_model

    rng = np.random.default_rng(1)
    mv = rng.random((6, MOVEMENT_FEATURES)).astype(np.float32)
    ph = rng.random((3, PHASE_FEATURES)).astype(np.float32)
    mask = rng.random((3, 6)) > 0.5
    with tempfile.TemporaryDirectory() as d:
        paths, member_q = [], []
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            params = export_numpy(torch_model(MOVEMENT_FEATURES, PHASE_FEATURES))
            paths.append(os.path.join(d, f"m{seed}.npz"))
            np.savez(paths[-1], **params)
            member_q.append(NumpyPolicy(params).q_values(mv, ph, mask))
        ens = os.path.join(d, "ensemble.npz")
        np.savez(ens, **NumpyPolicy.ensemble_params(paths))
        q = NumpyPolicy.load(ens).q_values(mv, ph, mask)
    np.testing.assert_allclose(q, np.mean(member_q, axis=0), rtol=1e-5, atol=1e-6)


def test_demand_is_deterministic():
    s = registry.get(SCEN)
    with tempfile.TemporaryDirectory() as d:
        a, b, c = (os.path.join(d, f"{k}.rou.xml") for k in "abc")
        generate(s.net_path, a, DemandSpec(horizon=600, seed=5))
        generate(s.net_path, b, DemandSpec(horizon=600, seed=5))
        generate(s.net_path, c, DemandSpec(horizon=600, seed=6))
        assert open(a).read() == open(b).read()
        assert open(a).read() != open(c).read()


def test_junction_model_structure():
    nm = NetworkModel(registry.get(SCEN).net_path)
    jm = nm.junctions[0]
    assert jm.num_movements == 12 and jm.num_phases == 4
    assert jm.mask.any(axis=1).all(), "every phase releases something"
    assert jm.mask.any(axis=0).all(), "every movement is served by some phase"
    yellow, red = jm.transition_states(0, 1)
    g0, g1 = jm.green_states[0], jm.green_states[1]
    for c0, c1, y, r in zip(g0, g1, yellow, red):
        if c0 in "Gg" and c1 not in "Gg":
            assert y == "y" and r == "r"
        if c0 not in "Gg":
            assert y == "r"


class RandomController:
    """Adversarial controller: requests a random phase at every decision."""
    name, adaptive, program = "random", True, None

    def __init__(self):
        self.rng = np.random.default_rng(1)

    def act(self, sim, ready):
        return {j: int(self.rng.integers(sim.junctions[j].num_phases)) for j in ready}


def test_safety_layer_invariants():
    s = registry.get(SCEN)
    nm = NetworkModel(s.net_path)
    cfg = SimConfig(lateral_resolution=None, max_extra_time=0)
    with tempfile.TemporaryDirectory() as d:
        routes = os.path.join(d, "r.rou.xml")
        generate(s.net_path, routes, DemandSpec(horizon=900, seed=2, per_lane_rate=s.per_lane_rate))
        states, worst_red = [], [0.0]
        from chowkpilot.env.sim import traci

        def on_step(sim):
            jm = nm.junctions[0]
            states.append(traci.trafficlight.getRedYellowGreenState(jm.id))
            t = traci.simulation.getTime()
            halt = [traci.lanearea.getLastStepHaltingNumber("e2_" + l) for l in jm.in_lanes]
            for m in range(jm.num_movements):
                if sum(halt[i] for i in jm.mv_in[m]) > 0:
                    worst_red[0] = max(worst_red[0], t - sim.mv_last_green[0][m])

        sim = Simulation(nm, routes, RandomController(), cfg, seed=2, horizon=900, workdir=d)
        res = sim.run(on_step=on_step)
    assert res["trips"] > 100
    # max red: a movement with stopped vehicles is released within max_red plus one decision cycle
    bound = cfg.max_red + cfg.min_green + cfg.delta + cfg.yellow + cfg.all_red + 1
    assert worst_red[0] <= bound, f"a movement waited {worst_red[0]:.0f}s for green (bound {bound}s)"
    # segment the signal history into constant-state runs
    runs, cur, n = [], states[0], 0
    for st in states:
        if st == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur, n = st, 1
    greens = set(nm.junctions[0].green_states)
    for i, (st, n) in enumerate(runs[1:-1], start=1):
        if st in greens:
            assert n >= cfg.min_green, f"green held only {n}s"
            assert n <= cfg.max_green + cfg.delta, f"green held {n}s > max green"
        elif "y" in st:
            assert n == cfg.yellow, f"yellow lasted {n}s"
            nxt = runs[i + 1][0]
            assert nxt not in greens or cfg.all_red == 0, "all-red clearance skipped"


def test_gap_features_and_older_policies():
    """The stop-line gap and near-arrival features are in range, and an 8-feature policy still runs."""
    s = registry.get(SCEN)
    nm = NetworkModel(s.net_path)
    with tempfile.TemporaryDirectory() as d:
        routes = os.path.join(d, "r.rou.xml")
        generate(s.net_path, routes, DemandSpec(horizon=300, seed=3, per_lane_rate=s.per_lane_rate))
        seen = []
        sim = Simulation(nm, routes, RandomController(), SimConfig(lateral_resolution=None, max_extra_time=0),
                         seed=3, horizon=300, workdir=d)
        sim.run(on_step=lambda sim: seen.append(sim.observe(0)[0]) if len(seen) < 300 else None)
    mv = np.stack(seen)
    assert mv.shape[-1] == MOVEMENT_FEATURES == 10
    assert ((0 <= mv[..., 8]) & (mv[..., 8] <= 1)).all() and (mv[..., 9] >= 0).all()
    assert (mv[..., 8] < 1).any() and (mv[..., 9] > 0).any(), "vehicles cross and arrive during the run"
    old = os.path.join(registry.ROOT, "models", "chowkpilot.npz")
    q = NumpyPolicy.load(old).q_values(mv[-1], np.zeros((4, PHASE_FEATURES), np.float32), nm.junctions[0].mask)
    assert q.shape == (4,) and np.isfinite(q).all()
