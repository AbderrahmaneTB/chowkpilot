"""ChowkPilot phase-scoring network.

Max-pressure scores a phase by summing a hand-written per-movement quantity
(vehicles in - vehicles out) over the movements the phase releases. ChowkPilot keeps
that structure, receives the max-pressure score itself as an input, and learns the
per-movement embedding and the scoring head with reinforcement learning:

    e_m   = phi(movement features m)                       shared MLP, per movement
    g_p   = [sum_{m in G_p} e_m / 8, sum_{m not in G_p} e_m / 8]
                                                            released vs kept waiting
    Q(p)  = psi([g_p, phase features p incl. max-pressure])  shared MLP, per phase

Sum pooling (not mean pooling) matters: with a mean, a phase that releases two very
congested turn bays looks as valuable as one that releases six busy through
movements, and a learned policy starves the arterial (we observed exactly that).

Because it only ever sees movements and a phase/movement mask, the same weights
run on any junction layout (3-arm, 4-arm, joined OSM clusters, any number of phases).

`NumpyPolicy` is the deployment form: pure NumPy, no PyTorch.
"""

from __future__ import annotations

import numpy as np

HIDDEN = 64
HEAD = 128


def _relu(x):
    return np.maximum(x, 0.0)


class NumpyPolicy:
    def __init__(self, params: dict):
        # an ensemble is stored as m0/<name>, m1/<name>, ...; it scores a phase by the mean of its members' scores
        groups = sorted({k.split("/", 1)[0] for k in params if "/" in k})
        self.members = [NumpyPolicy({k.split("/", 1)[1]: v for k, v in params.items() if k.startswith(g + "/")})
                        for g in groups]
        if self.members:
            self.delta = self.members[0].delta
            return
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        self.delta = int(self.p["__delta"][0]) if "__delta" in self.p else None  # decision interval, s (None: default)
        # architecture from the head's input width: 2H + Fp (sum pooling) or 3H + 4 (legacy mean pooling)
        self.legacy = self.p["psi1.weight"].shape[1] == 3 * HIDDEN + 4

    @classmethod
    def load(cls, path: str) -> "NumpyPolicy":
        with np.load(path) as z:
            return cls(dict(z))

    @staticmethod
    def ensemble_params(paths: list[str]) -> dict:
        """Parameters of an ensemble of saved policies, ready for np.savez."""
        out = {}
        for i, path in enumerate(paths):
            with np.load(path) as z:
                out.update({f"m{i}/{k}": z[k] for k in z.files})
        return out

    def q_values(self, mv: np.ndarray, ph: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """mv [M, Fm], ph [P, Fp], mask [P, M] bool -> Q [P]."""
        if self.members:
            return np.mean([m.q_values(mv, ph, mask) for m in self.members], axis=0)
        p = self.p
        mv = mv[:, :p["phi1.weight"].shape[1]]  # features are only ever appended: older policies read the first ones
        e = _relu(mv @ p["phi1.weight"].T + p["phi1.bias"])
        e = _relu(e @ p["phi2.weight"].T + p["phi2.bias"])          # [M, H]
        g = mask.astype(np.float32)
        r = 1.0 - g
        g_sum, r_sum = g @ e, r @ e
        if self.legacy:
            g_cnt = np.maximum(g.sum(1, keepdims=True), 1.0)
            r_cnt = np.maximum(r.sum(1, keepdims=True), 1.0)
            feats = np.concatenate([g_sum / g_cnt, g_sum / 8.0, r_sum / r_cnt, ph[:, :4]], axis=1)
        else:
            feats = np.concatenate([g_sum / 8.0, r_sum / 8.0, ph], axis=1)
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
            self.psi1 = nn.Linear(2 * HIDDEN + ph_features, HEAD)
            self.psi2 = nn.Linear(HEAD, HIDDEN)
            self.psi3 = nn.Linear(HIDDEN, 1)

        def forward(self, mv, ph, mask, mv_valid):
            """mv [B,M,Fm], ph [B,P,Fp], mask [B,P,M] (0/1, zero on padding), mv_valid [B,M]."""
            e = torch.relu(self.phi2(torch.relu(self.phi1(mv)))) * mv_valid.unsqueeze(-1)
            g = mask
            r = (1.0 - mask) * mv_valid.unsqueeze(1)
            feats = torch.cat([torch.bmm(g, e) / 8.0, torch.bmm(r, e) / 8.0, ph], dim=-1)
            h = torch.relu(self.psi2(torch.relu(self.psi1(feats))))
            return self.psi3(h).squeeze(-1)  # [B, P]

    return PhaseScorer()


def export_numpy(model) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
