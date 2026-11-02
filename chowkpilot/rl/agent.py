"""Exploring controller used inside training workers.

Decisions form a semi-Markov process: a decision lasts until the junction's next
decision point (5 s for an extension; yellow + all-red + min green for a switch).
Rewards are therefore accumulated per second with discounting, and each transition
stores its own bootstrap discount gamma**elapsed_seconds.
"""

from __future__ import annotations

import numpy as np

from ..control.controllers import _allowed
from .policy import NumpyPolicy


class LearningAgent:
    name, label, adaptive, program = "chowkpilot-train", "ChowkPilot (training)", True, None

    def __init__(self, policy: NumpyPolicy | None, eps: float, mp_share: float = 0.5,
                 gamma: float = 0.985, reward_scale: float = 50.0, seed: int = 0):
        self.policy = policy
        self.eps = eps
        self.mp_share = mp_share
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.rng = np.random.default_rng(seed)
        self.transitions: list = []
        self.episode_return = 0.0

    def reset(self, sim):
        J = len(sim.junctions)
        self.acc = np.zeros(J)
        self.disc = np.ones(J)
        self.pending: dict[int, tuple] = {}

    def on_step(self, sim):
        for j in self.pending:
            r = -sim.delay_rate(j) / self.reward_scale
            self.acc[j] += self.disc[j] * r
            self.disc[j] *= self.gamma
            self.episode_return += r

    def act(self, sim, ready):
        out = {}
        for j in ready:
            jm = sim.junctions[j]
            mv, ph = sim.observe(j)
            if j in self.pending:
                pmv, pph, pa = self.pending[j]
                self.transitions.append((pmv, pph, jm.mask, pa, self.acc[j], self.disc[j], mv, ph))
            allowed = _allowed(sim, j)
            choices = np.flatnonzero(allowed)
            if self.policy is None or self.rng.random() < self.eps:
                if self.rng.random() < self.mp_share:
                    p = sim.pressure(j)
                    p[~allowed] = -np.inf
                    a = int(np.argmax(p))
                else:
                    a = int(self.rng.choice(choices))
            else:
                q = self.policy.q_values(mv, ph, jm.mask)
                q[~allowed] = -np.inf
                a = int(np.argmax(q))
            self.pending[j] = (mv, ph, a)
            self.acc[j] = 0.0
            self.disc[j] = 1.0
            out[j] = a
        return out
