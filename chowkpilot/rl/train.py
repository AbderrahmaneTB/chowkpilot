"""Distributed Double-DQN training of the ChowkPilot phase scorer.

    python -m chowkpilot.rl.train --minutes 150 --workers 5

Worker processes each run their own SUMO (libsumo) episodes on randomly drawn training
layouts, demand levels, time-of-day profiles, ambulance rates and camera-noise levels,
and stream semi-Markov transitions to a single learner. Exploration mixes uniform
random phases with max-pressure actions, so early data covers sensible traffic states.

Worker 0 periodically runs greedy validation episodes on training networks with seeds
that are never used for testing; the curve is saved to models/train_log.json. The
held-out Hyderabad test areas are never touched during training or model selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import time
import traceback

import numpy as np

M_MAX, P_MAX = 32, 8
FM, FP = 10, 5  # = MOVEMENT_FEATURES, PHASE_FEATURES in chowkpilot.env.sim
VALIDATION = [  # (scenario, seed, demand overrides) on training networks only, seeds never used for testing
    ("syn_single_2lane_inc", 4242, {"scale": 1.0, "profile": "peak"}),
    ("syn_single_2lane_inc", 5252, {"scale": 1.0, "profile": "peak"}),
    ("syn_arterial3", 4243, {"scale": 1.0, "profile": "peak"}),
    ("syn_arterial3", 5253, {"scale": 1.0, "profile": "peak"}),
    ("hyd_gachibowli", 4244, {"scale": 1.0, "profile": "peak"}),
]


def task_key(sid, seed):
    return f"{sid}@{seed}"


def pack(transitions):
    n = len(transitions)
    mv = np.zeros((n, M_MAX, FM), np.float16)
    ph = np.zeros((n, P_MAX, FP), np.float16)
    mv2 = np.zeros((n, M_MAX, FM), np.float16)
    ph2 = np.zeros((n, P_MAX, FP), np.float16)
    mask = np.zeros((n, P_MAX, M_MAX), bool)
    mvalid = np.zeros((n, M_MAX), bool)
    pvalid = np.zeros((n, P_MAX), bool)
    a = np.zeros(n, np.int64)
    r = np.zeros(n, np.float32)
    d = np.zeros(n, np.float32)
    for i, (m1, p1, mk, ai, ri, di, m2, p2) in enumerate(transitions):
        M, P = m1.shape[0], p1.shape[0]
        mv[i, :M], ph[i, :P], mv2[i, :M], ph2[i, :P] = m1, p1, m2, p2
        mask[i, :P, :M] = mk
        mvalid[i, :M] = True
        pvalid[i, :P] = True
        a[i], r[i], d[i] = ai, ri, di
    return dict(mv=mv, ph=ph, mv2=mv2, ph2=ph2, mask=mask, mvalid=mvalid, pvalid=pvalid, a=a, r=r, d=d)


class Replay:
    def __init__(self, capacity: int):
        self.cap, self.size, self.ptr = capacity, 0, 0
        self.data = {
            "mv": np.zeros((capacity, M_MAX, FM), np.float16), "ph": np.zeros((capacity, P_MAX, FP), np.float16),
            "mv2": np.zeros((capacity, M_MAX, FM), np.float16), "ph2": np.zeros((capacity, P_MAX, FP), np.float16),
            "mask": np.zeros((capacity, P_MAX, M_MAX), bool), "mvalid": np.zeros((capacity, M_MAX), bool),
            "pvalid": np.zeros((capacity, P_MAX), bool), "a": np.zeros(capacity, np.int64),
            "r": np.zeros(capacity, np.float32), "d": np.zeros(capacity, np.float32),
        }

    def add(self, batch: dict):
        n = len(batch["a"])
        idx = (self.ptr + np.arange(n)) % self.cap
        for k, v in batch.items():
            self.data[k][idx] = v
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.cap, self.size + n)

    def sample(self, n: int, rng):
        idx = rng.integers(0, self.size, n)
        b = {k: v[idx] for k, v in self.data.items()}
        # trim padding to the largest junction in this batch (most junctions have <= 12 movements)
        m = int(np.flatnonzero(b["mvalid"].any(0)).max()) + 1
        p = int(np.flatnonzero(b["pvalid"].any(0)).max()) + 1
        for k in ("mv", "mv2"):
            b[k] = b[k][:, :m]
        for k in ("ph", "ph2", "pvalid"):
            b[k] = b[k][:, :p]
        b["mask"] = b["mask"][:, :p, :m]
        b["mvalid"] = b["mvalid"][:, :m]
        return b


def sample_task(rng, ids, weights):
    from chowkpilot.scenarios.demand import PROFILES
    sid = str(rng.choice(ids, p=weights))
    demand = dict(scale=float(rng.uniform(0.4, 1.35)), profile=str(rng.choice(list(PROFILES))),
                  imbalance=float(rng.uniform(0.2, 0.7)), ambulances_per_hour=float(rng.choice([0, 2, 4])))
    sim = dict(lateral_resolution=0.8, max_extra_time=0)
    if rng.random() < 0.3:
        sim.update(noise_sigma=float(rng.uniform(0.0, 0.2)), miss_prob=float(rng.uniform(0.0, 0.15)))
    return sid, demand, sim, int(rng.choice([1200, 1500, 1800]))


def worker_main(wid, weights_path, out_q, stop_ev, eps_val, seed_base, validate_every, out_weight=0.0):
    from chowkpilot.env.sim import SimConfig, Simulation
    from chowkpilot.eval.runner import demand_for, network, run_one
    from chowkpilot.rl.agent import LearningAgent
    from chowkpilot.rl.policy import NumpyPolicy
    from chowkpilot.scenarios import registry
    from chowkpilot.scenarios.demand import generate

    rng = np.random.default_rng(seed_base + 1000 * wid)
    train = [s for s in registry.load_all() if s.split == "train"]
    by_id = {s.id: s for s in train}
    ids = [s.id for s in train]
    w = np.array([1.0 + 0.15 * len(network(s).junctions) for s in train])
    w /= w.sum()
    policy, mtime, episodes = None, None, 0
    while not stop_ev.is_set():
        try:
            if os.path.exists(weights_path) and os.path.getmtime(weights_path) != mtime:
                mtime = os.path.getmtime(weights_path)
                policy = NumpyPolicy.load(weights_path)
        except Exception:
            pass
        if wid == 0 and validate_every and episodes % validate_every == 0:
            ckdir = os.path.join(os.path.dirname(weights_path), "checkpoints")
            ckpts = sorted(f for f in os.listdir(ckdir) if f.endswith(".npz")) if os.path.isdir(ckdir) else []
            if ckpts and ckpts[-1] != getattr(worker_main, "_last_ckpt", None):
                worker_main._last_ckpt = ckpts[-1]
                path = os.path.join(ckdir, ckpts[-1])
                res = {}
                for sid, seed, dem in VALIDATION:
                    try:
                        r = run_one(sid, "chowkpilot", seed=seed, horizon=1200, demand=dem, policy=path)
                        res[task_key(sid, seed)] = r["mean_delay"]
                    except Exception:
                        traceback.print_exc()
                out_q.put(("val", wid, {"ckpt": ckpts[-1], "updates": int(ckpts[-1][1:7]), "delay": res}))
        sid, demand, simo, horizon = sample_task(rng, ids, w)
        scen = by_id[sid]
        seed = int(rng.integers(1, 2_000_000))
        workdir = tempfile.mkdtemp(prefix=f"chowk_w{wid}_")
        try:
            routes = os.path.join(workdir, "r.rou.xml")
            generate(scen.net_path, routes, demand_for(scen, seed, horizon, **demand))
            agent = LearningAgent(policy, eps=eps_val.value, seed=seed, out_weight=out_weight)
            sim = Simulation(network(scen), routes, agent, SimConfig(**simo), seed=seed,
                             horizon=horizon, workdir=workdir)
            t0 = time.time()
            res = sim.run(on_step=agent.on_step)
            if agent.transitions:
                out_q.put(("data", wid, pack(agent.transitions),
                           {"sid": sid, "ret": agent.episode_return, "n": len(agent.transitions),
                            "sec": time.time() - t0, "delay": res.get("mean_delay", float("nan"))}))
        except Exception:
            traceback.print_exc()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        episodes += 1


def baseline_validation():
    from chowkpilot.eval.runner import run_one
    out = {}
    for c in ("fixed", "actuated", "maxpressure"):
        out[c] = {task_key(sid, seed): run_one(sid, c, seed=seed, horizon=1200, demand=dem)["mean_delay"]
                  for sid, seed, dem in VALIDATION}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--capacity", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.985)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--replay-ratio", type=float, default=0.25, help="gradient updates per new transition")
    ap.add_argument("--eps-start", type=float, default=0.7)
    ap.add_argument("--eps-end", type=float, default=0.1)
    ap.add_argument("--eps-decay-frac", type=float, default=0.6)
    ap.add_argument("--validate-every", type=int, default=6)
    ap.add_argument("--ckpt-every", type=int, default=2000, help="save a numbered checkpoint every N updates")
    ap.add_argument("--lr-min", type=float, default=3e-5, help="cosine-decayed learning rate at the end")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="learner device, e.g. cuda (SUMO workers always run on CPU)")
    ap.add_argument("--out-weight", type=float, default=0.0,
                    help="weight of exit-road delay in the reward (neighbourhood coupling)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines-only", action="store_true", help="compute baseline validation scores and exit")
    args = ap.parse_args()
    if args.baselines_only:
        os.makedirs(args.out, exist_ok=True)
        scores = baseline_validation()
        path = os.path.join(args.out, "baseline_validation.json")
        with open(path + ".tmp", "w") as f:
            json.dump(scores, f, indent=1)
        os.replace(path + ".tmp", path)
        return

    import torch
    from chowkpilot.rl.policy import export_numpy, torch_model

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    weights_path = os.path.abspath(os.path.join(args.out, "chowkpilot_live.npz"))
    final_path = os.path.abspath(os.path.join(args.out, "chowkpilot.npz"))

    device = torch.device(args.device)
    model = torch_model(FM, FP).to(device)
    target = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    replay = Replay(args.capacity)

    def save(path, updates):
        params = export_numpy(model)
        params["__updates"] = np.array([updates])
        tmp = path + ".tmp.npz"
        np.savez(tmp, **params)
        os.replace(tmp, path)

    cache = os.path.join(args.out, "baseline_validation.json")
    base_val = None  # computed separately (python -m chowkpilot.rl.train --baselines-only) so training starts at once
    def read_cache():
        try:
            with open(cache) as f:
                return json.load(f)
        except (OSError, ValueError):  # missing, or still being written by --baselines-only
            return None

    base_val = read_cache()

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=64)
    stop_ev = ctx.Event()
    eps_val = ctx.Value("d", args.eps_start)
    save(weights_path, 0)
    procs = [ctx.Process(target=worker_main, args=(i, weights_path, out_q, stop_ev, eps_val, args.seed + 17,
                                                  args.validate_every, args.out_weight), daemon=True)
             for i in range(args.workers)]
    for p in procs:
        p.start()

    t_start = time.time()
    deadline = t_start + 60 * args.minutes
    env_steps, updates, budget = 0, 0, 0.0
    log = {"args": vars(args), "baseline_validation": base_val, "curve": [], "validation": []}
    recent = []
    losses, qs = [], []
    last_log, last_save = time.time(), time.time()
    planned_steps = None
    try:
        while time.time() < deadline:
            got = 0
            while True:
                try:
                    kind, wid, payload, *rest = out_q.get_nowait()
                except queue.Empty:
                    break
                if kind == "data":
                    replay.add(payload)
                    n = len(payload["a"])
                    env_steps += n
                    budget += n * args.replay_ratio
                    recent.append(rest[0])
                    got += 1
                elif kind == "val":
                    payload["minutes"] = round((time.time() - t_start) / 60, 1)
                    log["validation"].append(payload)
                    print(f"[val] {payload}", flush=True)
            # epsilon decays with wall-clock progress (throughput varies with CPU load)
            frac = (time.time() - t_start) / (deadline - t_start)
            eps_val.value = args.eps_end + (args.eps_start - args.eps_end) * max(0.0, 1 - frac / args.eps_decay_frac)
            lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + np.cos(np.pi * min(1.0, frac)))
            for g in opt.param_groups:
                g["lr"] = lr
            if replay.size < args.warmup or budget < 1:
                time.sleep(0.05 if got else 0.25)
            while replay.size >= args.warmup and budget >= 1 and time.time() < deadline:
                b = replay.sample(args.batch, rng)
                t = {k: torch.from_numpy(np.asarray(v, dtype=np.float32 if v.dtype in (np.float16, bool) else v.dtype))
                     .to(device) for k, v in b.items()}
                q = model(t["mv"], t["ph"], t["mask"], t["mvalid"])
                q_sa = q.gather(1, t["a"].unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = model(t["mv2"], t["ph2"], t["mask"], t["mvalid"]).masked_fill(t["pvalid"] < 0.5, -1e9)
                    a2 = q_next.argmax(1, keepdim=True)
                    q_t = target(t["mv2"], t["ph2"], t["mask"], t["mvalid"]).gather(1, a2).squeeze(1)
                    y = t["r"] + t["d"] * q_t
                loss = torch.nn.functional.smooth_l1_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                with torch.no_grad():
                    for p_t, p in zip(target.parameters(), model.parameters()):
                        p_t.mul_(1 - args.tau).add_(p, alpha=args.tau)
                updates += 1
                budget -= 1
                if updates % args.ckpt_every == 0:
                    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
                    save(os.path.join(args.out, "checkpoints", f"u{updates:06d}.npz"), updates)
                losses.append(float(loss))
                qs.append(float(q_sa.mean()))
                if updates % 50 == 0:
                    break  # go back and drain the queue
            if time.time() - last_save > 30:
                save(weights_path, updates)
                last_save = time.time()
            if time.time() - last_log > 60:
                eps_ret = float(np.mean([r["ret"] / max(1, r["n"]) for r in recent])) if recent else float("nan")
                row = {"minutes": round((time.time() - t_start) / 60, 1), "env_steps": env_steps,
                       "updates": updates, "eps": round(eps_val.value, 3), "replay": replay.size,
                       "loss": float(np.mean(losses)) if losses else None, "q": float(np.mean(qs)) if qs else None,
                       "ret_per_decision": eps_ret, "episodes": len(recent),
                       "sim_sec_per_episode": float(np.mean([r["sec"] for r in recent])) if recent else None}
                log["curve"].append(row)
                print(json.dumps(row), flush=True)
                recent, losses, qs = [], [], []
                last_log = time.time()
                with open(os.path.join(args.out, "train_log.json"), "w") as f:
                    json.dump(log, f, indent=1)
    finally:
        stop_ev.set()
        save(weights_path, updates)
        save(final_path, updates)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(args.out, "chowkpilot.pt"))
        log["baseline_validation"] = read_cache() or log["baseline_validation"]
        with open(os.path.join(args.out, "train_log.json"), "w") as f:
            json.dump(log, f, indent=1)
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
        print(f"done: env_steps={env_steps} updates={updates}", flush=True)


if __name__ == "__main__":
    main()
