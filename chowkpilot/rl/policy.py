"""ChowkPilot phase-scoring network.

Max-pressure scores a phase by summing a hand-written per-movement quantity
(vehicles in - vehicles out) over the movements the phase releases. ChowkPilot keeps
that structure and learns the per-movement embedding and the scoring head with
reinforcement learning:

    e_m   = phi(movement features m)                       shared MLP, per movement
    g_p   = [mean_{m in G_p} e_m, sum_{m in G_p} e_m / 8,  pooled over released movements
             mean_{m not in G_p} e_m]                       and over the ones kept waiting
    Q(p)  = psi([g_p, phase features p])                   shared MLP, per phase

Because it only ever sees movements and a phase/movement mask, the same weights
run on any junction layout (3-arm, 4-arm, joined OSM clusters, any number of phases).

`NumpyPolicy` is the deployment form: 38k parameters, pure NumPy, no PyTorch.
"""

from __future__ import annotations

import numpy as np

HIDDEN = 64
HEAD = 128


def _relu(x):
    return np.maximum(x, 0.0)


class NumpyPolicy:
    def __init__(self, params: dict):
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}

    @classmethod
    def load(cls, path: str) -> "NumpyPolicy":
        with np.load(path) as z:
            return cls(dict(z))

    def q_values(self, mv: np.ndarray, ph: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """mv [M, Fm], ph [P, Fp], mask [P, M] bool -> Q [P]."""
        p = self.p
        e = _relu(mv @ p["phi1.weight"].T + p["phi1.bias"])
        e = _relu(e @ p["phi2.weight"].T + p["phi2.bias"])          # [M, H]
        g = mask.astype(np.float32)
        r = 1.0 - g
        g_cnt = np.maximum(g.sum(1, keepdims=True), 1.0)
        r_cnt = np.maximum(r.sum(1, keepdims=True), 1.0)
        g_sum = g @ e
        feats = np.concatenate([g_sum / g_cnt, g_sum / 8.0, (r @ e) / r_cnt, ph], axis=1)
        h = _relu(feats @ p["psi1.weight"].T + p["psi1.bias"])
        h = _relu(h @ p["psi2.weight"].T + p["psi2.bias"])
        return (h @ p["psi3.weight"].T + p["psi3.bias"])[:, 0]


def torch_model(mv_features: int, ph_features: int):
    """The trainable PyTorch twin of NumpyPolicy (imported lazily: inference needs no torch)."""
    import torch
    from torch import nn

    class PhaseScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi1 = nn.Linear(mv_features, HIDDEN)
            self.phi2 = nn.Linear(HIDDEN, HIDDEN)
            self.psi1 = nn.Linear(3 * HIDDEN + ph_features, HEAD)
            self.psi2 = nn.Linear(HEAD, HIDDEN)
            self.psi3 = nn.Linear(HIDDEN, 1)

        def forward(self, mv, ph, mask, mv_valid):
            """mv [B,M,Fm], ph [B,P,Fp], mask [B,P,M] (0/1, zero on padding), mv_valid [B,M]."""
            e = torch.relu(self.phi2(torch.relu(self.phi1(mv)))) * mv_valid.unsqueeze(-1)
            g = mask
            r = (1.0 - mask) * mv_valid.unsqueeze(1)
            g_cnt = g.sum(-1, keepdim=True).clamp(min=1.0)
            r_cnt = r.sum(-1, keepdim=True).clamp(min=1.0)
            g_sum = torch.bmm(g, e)
            feats = torch.cat([g_sum / g_cnt, g_sum / 8.0, torch.bmm(r, e) / r_cnt, ph], dim=-1)
            h = torch.relu(self.psi2(torch.relu(self.psi1(feats))))
            return self.psi3(h).squeeze(-1)  # [B, P]

    return PhaseScorer()


def export_numpy(model) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
