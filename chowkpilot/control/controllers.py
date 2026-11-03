"""Signal controllers: the baselines and ChowkPilot.

Fixed-time, Webster and actuated run as native SUMO programs. Max-pressure and
ChowkPilot are adaptive controllers driven through the shared safety state machine
in `chowkpilot.env.sim` (same min/max green, yellow and all-red for both).
"""

from __future__ import annotations

import numpy as np

from ..rl.policy import NumpyPolicy


class FixedTime:
    name, label, adaptive, program = "fixed", "Fixed-time (standard plan)", False, "0"


class Webster:
    """Fixed-time plan re-optimised with Webster's method using the true demand."""
    name, label, adaptive, program = "webster", "Fixed-time (Webster-optimised)", False, "webster"


class Actuated:
    """SUMO gap-based vehicle-actuated control (extends green while vehicles keep arriving)."""
    name, label, adaptive, program = "actuated", "Vehicle-actuated", False, "actuated"


def _allowed(sim, j):
    """Phases a controller may pick (max-green and max-red safety constraints, see Simulation)."""
    return sim.allowed_phases(j)


class MaxPressure:
    """Max-pressure control (Varaiya, Transportation Research Part C, 2013)."""
    name, label, adaptive, program = "maxpressure", "Max-pressure", True, None

    def act(self, sim, ready):
        out = {}
        for j in ready:
            p = sim.pressure(j)
            p[~_allowed(sim, j)] = -np.inf
            out[j] = int(np.argmax(p))
        return out

    def fallback(self, sim, j):
        p = sim.pressure(j)
        p[sim.phase[j]] = -np.inf
        return int(np.argmax(p))


class ChowkPilot:
    """Learned phase scorer (see chowkpilot.rl.policy); greedy at deployment."""
    name, label, adaptive, program = "chowkpilot", "ChowkPilot (ours)", True, None

    def __init__(self, policy: NumpyPolicy | str, explain: bool = False):
        self.policy = NumpyPolicy.load(policy) if isinstance(policy, str) else policy
        self.explain = explain

    def scores(self, sim, j):
        mv, ph = sim.observe(j)
        return self.policy.q_values(mv, ph, sim.junctions[j].mask)

    def act(self, sim, ready):
        out = {}
        for j in ready:
            q = self.scores(sim, j)
            q_masked = np.where(_allowed(sim, j), q, -np.inf)
            out[j] = int(np.argmax(q_masked))
            if self.explain:
                sim.explain.append({"t": int(sim_time(sim)), "j": j, "q": [round(float(x), 3) for x in q],
                                    "a": out[j], "cur": int(sim.phase[j])})
        return out

    def fallback(self, sim, j):
        q = self.scores(sim, j)
        q[sim.phase[j]] = -np.inf
        return int(np.argmax(q))


def sim_time(sim):
    from ..env.sim import traci
    return traci.simulation.getTime()


BASELINES = {c.name: c for c in (FixedTime, Webster, Actuated, MaxPressure)}
