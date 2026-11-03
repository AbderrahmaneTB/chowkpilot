"""SUMO simulation driver shared by training, benchmarking and the web demo.

Adaptive controllers (max-pressure, ChowkPilot) act through a safety state machine
that every adaptive controller shares:

    GREEN (>= min_green, re-decided every `delta` s, forced change after max_green)
      -> YELLOW (yellow s) -> ALL-RED (all_red s) -> GREEN of the chosen phase

Observations come only from camera-like lane-area detectors covering the last
`detect_range` metres of each approach. The reward (training only) uses simulator
ground truth: the delay rate of every vehicle on the approach lanes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import sumolib
import traci.constants as tc

try:
    import libsumo as traci
    LIBSUMO = True
except ImportError:  # pragma: no cover - fallback for platforms without libsumo
    import traci
    LIBSUMO = False

from ..scenarios.vtypes import TYPE_INDEX
from ..sumo_paths import binary
from .junction import JunctionModel

MOVEMENT_FEATURES = 8
PHASE_FEATURES = 5
TYPE_BY_PREFIX = {name[:2]: idx for name, idx in TYPE_INDEX.items()}
AMBULANCE_PREFIX = "am"


@dataclass
class SimConfig:
    yellow: int = 3
    all_red: int = 1
    min_green: int = 10
    max_green: int = 60
    max_red: int = 120              # a movement with stopped vehicles must get green within this
    delta: int = 5
    detect_range: float = 150.0
    lateral_resolution: float | None = 0.8
    teleport: int = 300
    max_extra_time: int = 1200      # after the demand horizon, run until empty or this cap
    noise_sigma: float = 0.0        # multiplicative detector count noise (robustness tests)
    miss_prob: float = 0.0          # probability that the camera misses a vehicle
    ev_priority: bool = False       # emergency-vehicle green-corridor layer
    ev_min_green: int = 5           # green that must elapse before pre-empting for an ambulance
    record: bool = False            # store vehicle positions for the web replay
    record_window: tuple = (0, 900)
    record_every: int = 1
    collect_reward: bool = False


class NetworkModel:
    """Static, cached information about one SUMO network."""

    def __init__(self, net_path: str, detect_range: float = 150.0):
        self.net_path = os.path.abspath(net_path)
        self.net = sumolib.net.readNet(self.net_path, withPrograms=True)
        self.junctions: list[JunctionModel] = []
        for tls in sorted(self.net.getTrafficLights(), key=lambda t: t.getID()):
            progs = tls.getPrograms()
            if not progs:
                continue
            prog = progs.get("0") or next(iter(progs.values()))
            try:
                jm = JunctionModel(tls, [p.state for p in prog.getPhases()], detect_range)
            except ValueError:
                continue
            if jm.num_phases >= 2:  # single-phase signals have nothing to decide
                self.junctions.append(jm)
        self._det_lines = []
        for jm in self.junctions:
            self._det_lines += jm.detector_xml(self.net)

    def write_detectors(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write("<additional>\n" + "\n".join(self._det_lines).replace('file="NUL"', f'file="{os.devnull}"')
                    + "\n</additional>\n")
        return path


class Simulation:
    def __init__(self, network: NetworkModel, route_file: str, controller, cfg: SimConfig,
                 seed: int = 0, horizon: int = 3600, additional: list[str] | None = None,
                 workdir: str | None = None):
        self.nm = network
        self.junctions = network.junctions
        self.route_file = route_file
        self.controller = controller
        self.cfg = cfg
        self.seed = seed
        self.horizon = horizon
        self.additional = additional or []
        self._own_workdir = workdir is None
        self.workdir = workdir or tempfile.mkdtemp(prefix="chowk_")
        self.tripinfo = os.path.join(self.workdir, f"trips_{os.getpid()}_{seed}.xml")
        self.rng = np.random.default_rng(seed + 7919)
        J = len(self.junctions)
        self.phase = np.zeros(J, dtype=int)
        self.mode = ["G"] * J
        self.timer = np.zeros(J, dtype=int)
        self.next_phase = np.zeros(J, dtype=int)
        self.last_served = [np.zeros(jm.num_phases) for jm in self.junctions]
        self.mv_last_green = [np.zeros(jm.num_movements) for jm in self.junctions]
        self.decisions = 0
        self.frames: list = []
        self.vid_map: dict[str, int] = {}
        self.series = {"t": [], "halting": [], "vehicles": []}
        self.explain: list = []
        self.reward_acc = np.zeros(J)
        self.teleports = 0
        self.ev_active: set[str] = set()

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        det_file = self.nm.write_detectors(os.path.join(self.workdir, f"det_{os.getpid()}.add.xml"))
        adds = [det_file] + self.additional
        cmd = [binary("sumo"), "-n", self.nm.net_path, "-r", self.route_file, "-a", ",".join(adds),
               "--seed", str(self.seed), "--step-length", "1",
               "--no-step-log", "true", "--no-warnings", "true", "--duration-log.disable", "true",
               "--time-to-teleport", str(self.cfg.teleport),
               "--tripinfo-output", self.tripinfo, "--tripinfo-output.write-unfinished", "true",
               "--device.emissions.probability", "1", "--waiting-time-memory", "100000",
               "--collision.action", "warn", "--collision.check-junctions", "false",
               "--ignore-route-errors", "true", "--time-to-impatience", "30",
               "--ignore-junction-blocker", "20"]
        if self.cfg.lateral_resolution:
            cmd += ["--lateral-resolution", str(self.cfg.lateral_resolution)]
        traci.start(cmd)
        self.adaptive = getattr(self.controller, "adaptive", True)
        program = getattr(self.controller, "program", None)
        for j, jm in enumerate(self.junctions):
            if self.adaptive:
                self.phase[j] = 0
                traci.trafficlight.setRedYellowGreenState(jm.id, jm.green_states[0])
            elif program:
                traci.trafficlight.setProgram(jm.id, program)
        self.lane_speed = {}
        for jm in self.junctions:
            for l in jm.in_lanes + jm.out_lanes:
                self.lane_speed[l] = max(1.0, traci.lane.getMaxSpeed(l))
        if hasattr(self.controller, "reset"):
            self.controller.reset(self)

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    def cleanup(self):
        if self._own_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)

    # ------------------------------------------------------------------ observations
    def lane_counts(self, jm: JunctionModel):
        """Camera-zone counts per in-lane: (halting, vehicles, occupancy, ambulance flag)."""
        n = len(jm.in_lanes)
        halt, veh, occ, ev = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for i, lane in enumerate(jm.in_lanes):
            d = "e2_" + lane
            halt[i] = traci.lanearea.getLastStepHaltingNumber(d)
            veh[i] = traci.lanearea.getLastStepVehicleNumber(d)
            occ[i] = traci.lanearea.getLastStepOccupancy(d) / 100.0
            if self.ev_active:
                ev[i] = any(v.startswith(AMBULANCE_PREFIX) for v in traci.lanearea.getLastStepVehicleIDs(d))
        if self.cfg.noise_sigma > 0 or self.cfg.miss_prob > 0:
            keep = 1.0 - self.cfg.miss_prob
            veh = self.rng.binomial(veh.astype(int), keep).astype(float)
            halt = np.minimum(self.rng.binomial(halt.astype(int), keep).astype(float), veh)
            if self.cfg.noise_sigma > 0:
                f = np.clip(1.0 + self.rng.normal(0, self.cfg.noise_sigma, n), 0.0, None)
                veh, halt = np.round(veh * f), np.round(halt * f)
        return halt, veh, occ, ev

    def observe(self, j: int):
        """Per-movement features [M, 8] and per-phase features [P, 5] for junction j."""
        jm = self.junctions[j]
        halt, veh, occ, ev = self.lane_counts(jm)
        wait = np.array([traci.lane.getWaitingTime(l) for l in jm.in_lanes])
        out_occ = np.array([traci.lane.getLastStepOccupancy(l) / 100.0 for l in jm.out_lanes])
        out_veh = np.array([traci.lane.getLastStepVehicleNumber(l) for l in jm.out_lanes], dtype=float)
        M = jm.num_movements
        mv = np.zeros((M, MOVEMENT_FEATURES), dtype=np.float32)
        cur_mask = jm.mask[self.phase[j]] if self.mode[j] == "G" else jm.mask[self.next_phase[j]]
        for m in range(M):
            li, lo = jm.mv_in[m], jm.mv_out[m]
            h, v = halt[li].sum(), veh[li].sum()
            mv[m, 0] = h / 10.0
            mv[m, 1] = (v - h) / 10.0
            mv[m, 2] = occ[li].mean()
            mv[m, 3] = out_occ[lo].mean() if lo else 0.0
            mv[m, 4] = min(3.0, wait[li].sum() / max(1.0, h) / 60.0)
            mv[m, 5] = ev[li].max()
            mv[m, 6] = len(li) / 3.0
            mv[m, 7] = float(cur_mask[m])
        P = jm.num_phases
        t = traci.simulation.getTime()
        ph = np.zeros((P, PHASE_FEATURES), dtype=np.float32)
        ph[:, 3] = jm.mask.sum(axis=1) / max(1, M)
        if self.mode[j] == "G":
            ph[self.phase[j], 0] = 1.0
            ph[self.phase[j], 1] = self.timer[j] / self.cfg.max_green
        ph[:, 2] = np.minimum(2.0, (t - self.last_served[j]) / 120.0)
        # the max-pressure score of each phase: the classical prior the network learns to correct
        mp = np.array([veh[jm.mv_in[m]].sum() - (out_veh[jm.mv_out[m]].mean() if jm.mv_out[m] else 0.0)
                       for m in range(M)])
        ph[:, 4] = (jm.mask.astype(float) @ mp) / 20.0
        return mv, ph

    def pressure(self, j: int) -> np.ndarray:
        """Max-pressure score of every phase (Varaiya 2013), from camera-zone counts."""
        jm = self.junctions[j]
        _, veh, _, _ = self.lane_counts(jm)
        out = np.array([traci.lane.getLastStepVehicleNumber(l) for l in jm.out_lanes], dtype=float)
        mp = np.array([veh[jm.mv_in[m]].sum() - (out[jm.mv_out[m]].mean() if jm.mv_out[m] else 0.0)
                       for m in range(jm.num_movements)])
        return jm.mask.astype(float) @ mp

    def delay_rate(self, j: int, out_weight: float = 0.0) -> float:
        """Ground-truth delay accumulated per second on junction j's approaches (training reward).

        With out_weight > 0 the junction also pays for delay on its exit roads, so it cannot
        push platoons into a jam downstream for free (the coupling max-pressure gets from '- out').
        """
        jm = self.junctions[j]
        lanes = [(l, 1.0) for l in jm.in_lanes]
        if out_weight > 0:
            lanes += [(l, out_weight) for l in jm.out_lanes]
        total = 0.0
        for l, w in lanes:
            n = traci.lane.getLastStepVehicleNumber(l)
            if n:
                total += w * n * max(0.0, 1.0 - traci.lane.getLastStepMeanSpeed(l) / self.lane_speed[l])
        return total

    # ------------------------------------------------------------------ control
    def allowed_phases(self, j: int) -> np.ndarray:
        """Safety constraints on the next phase, applied to every adaptive controller.

        * max green: the current phase cannot be extended beyond max_green;
        * max red: if a movement with stopped vehicles has waited longer than max_red,
          only the phases that release the longest-waiting such movement are allowed.
        """
        jm, cfg = self.junctions[j], self.cfg
        ok = np.ones(jm.num_phases, dtype=bool)
        if self.timer[j] >= cfg.max_green:
            ok[self.phase[j]] = False
        t = traci.simulation.getTime()
        waited = t - self.mv_last_green[j]
        if waited.max() >= cfg.max_red:
            halt = np.array([traci.lanearea.getLastStepHaltingNumber("e2_" + l) for l in jm.in_lanes])
            starving = [m for m in np.argsort(-waited)
                        if waited[m] >= cfg.max_red and halt[jm.mv_in[m]].sum() > 0]
            if starving:
                serve = jm.mask[:, starving[0]] & (np.arange(jm.num_phases) != self.phase[j])                     if not jm.mask[self.phase[j], starving[0]] else jm.mask[:, starving[0]]
                if serve.any():
                    ok = serve
        return ok

    def ready_junctions(self) -> list[int]:
        cfg = self.cfg
        return [j for j in range(len(self.junctions))
                if self.mode[j] == "G" and self.timer[j] >= cfg.min_green
                and (self.timer[j] - cfg.min_green) % cfg.delta == 0]

    def ev_requests(self) -> dict[int, int]:
        """Junctions that must serve an approaching ambulance -> phase to give it."""
        req = {}
        for vid in list(self.ev_active):
            try:
                lane = traci.vehicle.getLaneID(vid)
                route, idx = traci.vehicle.getRoute(vid), traci.vehicle.getRouteIndex(vid)
            except Exception:
                continue
            for j, jm in enumerate(self.junctions):
                if lane not in jm.lane_index:
                    continue
                length = jm.lanes_geo[lane][0]
                if length - traci.vehicle.getLanePosition(vid) > self.cfg.detect_range:
                    continue
                if idx + 1 >= len(route):
                    continue
                key = (route[idx], route[idx + 1])
                ms = [m for m, mv in enumerate(jm.movements) if (mv.in_edge, mv.out_edge) == key]
                if not ms:
                    continue
                phases = np.where(jm.mask[:, ms[0]])[0]
                if len(phases):
                    req[j] = int(phases[0]) if self.phase[j] not in phases else int(self.phase[j])
        return req

    def _switch(self, j: int, a: int):
        jm = self.junctions[j]
        if a == self.phase[j]:
            return
        self.next_phase[j] = a
        self.mode[j] = "Y"
        self.timer[j] = 0
        yellow, _ = jm.transition_states(self.phase[j], a)
        traci.trafficlight.setRedYellowGreenState(jm.id, yellow)

    def _advance(self):
        cfg, t = self.cfg, traci.simulation.getTime()
        for j, jm in enumerate(self.junctions):
            if self.mode[j] == "Y" and self.timer[j] >= cfg.yellow:
                if cfg.all_red > 0:
                    self.mode[j] = "R"
                    self.timer[j] = 0
                    _, red = jm.transition_states(self.phase[j], self.next_phase[j])
                    traci.trafficlight.setRedYellowGreenState(jm.id, red)
                else:
                    self._enter_green(j)
            elif self.mode[j] == "R" and self.timer[j] >= cfg.all_red:
                self._enter_green(j)
            if self.mode[j] == "G":
                self.last_served[j][self.phase[j]] = t
                self.mv_last_green[j][jm.mask[self.phase[j]]] = t

    def _enter_green(self, j: int):
        jm = self.junctions[j]
        self.phase[j] = self.next_phase[j]
        self.mode[j] = "G"
        self.timer[j] = 0
        traci.trafficlight.setRedYellowGreenState(jm.id, jm.green_states[self.phase[j]])

    def control_step(self):
        """Ask the controller for decisions at every junction that is due one."""
        cfg = self.cfg
        ready = self.ready_junctions()
        forced = {}
        if cfg.ev_priority and self.ev_active:
            for j, ph in self.ev_requests().items():
                if self.mode[j] != "G":
                    continue
                if ph == self.phase[j]:
                    forced[j] = ph  # hold green while the ambulance approaches
                elif self.timer[j] >= cfg.ev_min_green:
                    forced[j] = ph
        ready = [j for j in ready if j not in forced]
        if ready:
            actions = self.controller.act(self, ready)
            for j in ready:
                a = int(actions[j])
                ok = self.allowed_phases(j)  # enforced here, whatever the controller requested
                if not ok[a]:
                    p = self.pressure(j)
                    p[~ok] = -np.inf
                    a = int(np.argmax(p))
                self._switch(j, a)
            self.decisions += len(ready)
        for j, a in forced.items():
            self._switch(j, a)

    # ------------------------------------------------------------------ main loop
    def run(self, on_step=None) -> dict:
        self.start()
        cfg = self.cfg
        cap = self.horizon + cfg.max_extra_time
        try:
            while True:
                t = traci.simulation.getTime()
                if t >= cap or (t >= self.horizon and traci.simulation.getMinExpectedNumber() == 0):
                    break
                if self.adaptive:
                    self._advance()
                    self.control_step()
                traci.simulationStep()
                self.timer += 1
                self._track_vehicles()
                if cfg.collect_reward and self.adaptive:
                    for j in range(len(self.junctions)):
                        self.reward_acc[j] += self.delay_rate(j)
                if on_step is not None:
                    on_step(self)
                t1 = t + 1
                if cfg.record and cfg.record_window[0] <= t1 <= cfg.record_window[1] \
                        and int(t1) % cfg.record_every == 0:
                    self._record_frame(t1)
                if int(t1) % 10 == 0:
                    self._record_series(t1)
        finally:
            self.close()
        result = parse_tripinfo(self.tripinfo, self.horizon)
        result["teleports"] = self.teleports
        result["decisions"] = self.decisions
        result["series"] = self.series
        return result

    def _track_vehicles(self):
        for vid in traci.simulation.getDepartedIDList():
            if vid.startswith(AMBULANCE_PREFIX):
                self.ev_active.add(vid)
            if self.cfg.record:
                traci.vehicle.subscribe(vid, [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_SPEED])
        if self.ev_active:
            for vid in traci.simulation.getArrivedIDList():
                self.ev_active.discard(vid)
        self.teleports += traci.simulation.getStartingTeleportNumber()

    def _record_series(self, t):
        halting = 0
        for jm in self.junctions:
            for l in jm.in_lanes:
                halting += traci.lane.getLastStepHaltingNumber(l)
        self.series["t"].append(int(t))
        self.series["halting"].append(int(halting))
        self.series["vehicles"].append(int(traci.vehicle.getIDCount()))

    def _record_frame(self, t):
        res = traci.vehicle.getAllSubscriptionResults()
        flat = []
        for vid, r in res.items():
            if tc.VAR_POSITION not in r:
                continue
            k = self.vid_map.setdefault(vid, len(self.vid_map))
            x, y = r[tc.VAR_POSITION]
            ty = TYPE_BY_PREFIX.get(vid[:2], 1)
            halted = 1 if r[tc.VAR_SPEED] < 0.1 else 0
            flat += [k, int(round(x * 10)), int(round(y * 10)), int(round(r[tc.VAR_ANGLE])) % 360, ty + 10 * halted]
        signals = [traci.trafficlight.getRedYellowGreenState(jm.id) for jm in self.junctions]
        self.frames.append({"t": int(t), "v": flat, "s": signals})


def parse_tripinfo(path: str, horizon: int) -> dict:
    """Aggregate SUMO tripinfo output into benchmark metrics."""
    rows = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "tripinfo":
            continue
        em = el.find("emissions")
        rows.append((
            el.get("vType"),
            float(el.get("duration")),
            float(el.get("waitingTime")),
            float(el.get("timeLoss")) + max(0.0, float(el.get("departDelay"))),
            float(el.get("waitingCount")),
            float(el.get("arrival")),
            float(em.get("CO2_abs")) if em is not None else 0.0,
            float(em.get("fuel_abs")) if em is not None else 0.0,
        ))
        el.clear()
    if not rows:
        return {"trips": 0}
    vt = np.array([r[0] for r in rows])
    dur, wait, delay, stops, arr, co2, fuel = (np.array([r[i] for r in rows]) for i in range(1, 8))
    finished = arr >= 0
    out = {
        "trips": int(len(rows)),
        "unfinished": int((~finished).sum()),
        "throughput_horizon": int(((arr >= 0) & (arr <= horizon)).sum()),
        "mean_travel_time": float(dur.mean()),
        "mean_wait": float(wait.mean()),
        "mean_delay": float(delay.mean()),
        "p95_delay": float(np.percentile(delay, 95)),
        "max_wait": float(wait.max()),
        "mean_stops": float(stops.mean()),
        "co2_kg": float(co2.sum() / 1e6),          # mg -> kg
        "fuel_kg": float(fuel.sum() / 1e6),        # mg -> kg
        "by_type": {},
    }
    for name in np.unique(vt):
        sel = vt == name
        out["by_type"][str(name)] = {"n": int(sel.sum()), "mean_delay": float(delay[sel].mean()),
                                     "mean_travel_time": float(dur[sel].mean())}
    return out
