# ChowkPilot 🚦 One AI signal brain for every *chowk*

**NeuralStorm 2026 · Track 1 Future Tech · Problem statement FT-01: Smart Traffic Signal Optimisation**
**Team Blackwater:** Abderrahmane ([@AbderrahmaneTB](https://github.com/AbderrahmaneTB)) · Wael ([@WaelYTz](https://github.com/WaelYTz))

**[▶ Live demo (digital twin control room)](https://abderrahmanetb.github.io/chowkpilot/)** · [How it works](#how-it-works) · [Results](#results) · [Run it yourself](#run-it-yourself)

<!-- RESULTS-HERO -->

ChowkPilot is a reinforcement-learning traffic-signal controller built for Indian junctions. Every few
seconds it decides which approach gets green, using only what a roadside camera can count. One small
network (38k parameters) drives **any** junction layout: 3-arm, 4-arm, 5-arm, or a joined
OpenStreetMap cluster with any number of phases. It was trained on synthetic junctions and two real
Hyderabad areas, then evaluated **zero-shot on three real Hyderabad areas it never saw**, in a
simulator configured for Indian traffic: lane-less two-wheelers, autos, buses, left-hand driving and
assertive merging.

---

## The problem

- Indian cities are among the most congested in the world. In the 2026 TomTom Traffic Index (2025 data),
  **Bengaluru ranks #2 worldwide** (36 min 9 s per 10 km, 168 hours lost per year in rush hour).
  **Hyderabad** drivers need 32 min 37 s per 10 km and lose **123 hours a year** in rush hour.
  ([TomTom Traffic Index](https://www.tomtom.com/traffic-index/city/hyderabad/))
- Congestion costs Delhi, Mumbai, Bengaluru and Kolkata up to **USD 22 billion a year**
  ([BCG, *Unlocking Cities*, 2018](https://web-assets.bcg.com/img-src/BCG-Unlocking-Cities-Ridesharing-India_tcm9-185213.pdf)).
- Traffic is **mixed and lane-less**: **73.4%** of Telangana's registered vehicles are motorcycles
  ([Telangana Transport Department, 31 Mar 2025](https://transport.telangana.gov.in/html/statistics_vehicles.html)).
  Controllers tuned on lane-disciplined Western car traffic don't match this stream.
- Most signals still run **fixed-time plans**. Hyderabad's adaptive programme covers 122 junctions alongside
  213 older HTRIMS signals ([Telangana Today, 2022](https://telanganatoday.com/new-traffic-signal-system-in-hyderabad)).
  Commercial adaptive systems typically need dedicated detectors, licences and per-junction tuning.
- Emergency vehicles lose minutes at red lights: Delhi's average ambulance response time has risen above
  17 minutes ([The Hindu, 2025](https://www.thehindu.com/sci-tech/health/reclaiming-the-golden-hour-how-technology-can-help-beat-urban-traffic-and-save-lives/article69678684.ece)).

**What existing approaches miss**

| Approach | Limitation | ChowkPilot |
|---|---|---|
| Fixed-time plans (most Indian signals) | Can't react to demand; plans go stale | Re-decides every 5 s from live counts |
| Vehicle-actuated / gap-based | Built for loop detectors and gaps between vehicles; dense two-wheeler streams rarely leave clean gaps | Uses queue, flow, waiting time and spillback from camera zones |
| Max-pressure (Varaiya, 2013) | Hand-written score, ignores waiting time, approaching platoons and switching cost | Keeps max-pressure's structure and *learns* the score |
| RL controllers (PressLight, CoLight, …) | Usually trained and evaluated on the same synthetic grid or city map, with lane-disciplined traffic | Shares weights across movements and phases (in the spirit of FRAP and MPLight) for *any* phase set, and is tested zero-shot on unseen real junctions with Indian mixed traffic |
| Timing-advice systems (e.g. Google Green Light) | Offline plan suggestions; emissions modelled with a single vehicle type | Real-time control, evaluated on an Indian mixed-traffic stream |

## How it works

```
 OpenStreetMap ──► SUMO digital twin ──► camera zones (last 150 m per lane)
 (real junction)    left-hand, sublane       │  vehicles, stopped vehicles, occupancy,
                    mixed traffic            │  waiting time, exit-road spillback, ambulance flag
                                             ▼
            per-movement features ──► φ (shared MLP) ──► sum-pool over the movements each phase
                                                          releases / keeps waiting
                                                                    │   + max-pressure score (prior)
                                                                    ▼
                         safety layer ◄── argmax ◄── ψ (shared MLP): score Q(phase)
   min green 10 s · max green 60 s · max red 120 s · yellow 3 s · all-red 1 s · ambulance green corridor
```

### 1. Digital twins of real Hyderabad junctions
- `chowkpilot/scenarios/osm.py` downloads the main road network for a bounding box from OpenStreetMap
  (Overpass API) and converts it with SUMO `netconvert`, with **left-hand traffic**, signal guessing,
  joined junction clusters and street names kept.
- Test areas, never used in training: **Secunderabad** (S.D. Road, Rashtrapati Road, Sardar Patel Road; 5 signals),
  **Hitech City** (Cyber Towers / HITEC City Flyover junction) and **Ameerpet / Begumpet** (5 signals).
  Training areas: Gachibowli / Financial District and Abids / Nampally, plus 11 synthetic layouts
  (single junctions with 1 to 3 lanes, arterials, grids, a 5-arm spider network and random networks).
- **Indian mixed traffic** (`chowkpilot/scenarios/vtypes.py`): two-wheelers 52%, cars 27%, autos 12%,
  light trucks 5%, buses 4%, simulated with SUMO's **sublane model**. Two-wheelers and autos filter between
  cars and queue side by side at the stop line. Drivers merge assertively at unsignalised junctions
  (impatience, short minor-road gaps), as Indian traffic does.
- **Paired, seeded demand** (`chowkpilot/scenarios/demand.py`): every vehicle has its own departure time,
  type and route, so each controller faces *exactly the same vehicles*. Arrivals follow a
  time-of-day profile. Nominal (peak) demand is calibrated per junction to the onset of congestion under
  vehicle-actuated control (`chowkpilot/scenarios/calibrate.py`).

### 2. A junction abstraction that works for any layout
`chowkpilot/env/junction.py` reduces any SUMO signal to **movements** (incoming road → outgoing road),
**green phases** (the candidate actions) and a mask of which movements each phase releases. Controllers only
ever see per-movement features and this mask, which is why one set of weights can run junctions it has
never seen.

### 3. The learned max-pressure policy
Max-pressure gives each phase the score `Σ (vehicles in − vehicles out)` over the movements it releases, and
picks the highest. It comes with a proof of throughput optimality, but its score is hand-written. ChowkPilot keeps
that structure, takes the max-pressure score as an input, and **learns a correction to it**
(`chowkpilot/rl/policy.py`):

- `e_m = φ(features of movement m)`: 8 camera-zone features (stopped and moving vehicles, occupancy, mean
  waiting time, exit-road occupancy for spillback, ambulance flag, lane count, currently green).
- For each phase *p*: **sum**-pool `e_m` over the movements it releases and over the movements it keeps waiting,
  append 5 phase features (is current, time in green, time since last served, share of movements, max-pressure
  score), then `Q(p) = ψ(·)`. The network has 38k parameters and the same weights serve every junction.
- **Training** (`chowkpilot/rl/train.py`): Double DQN with a semi-Markov formulation. Each decision lasts until the
  junction's next decision point, and rewards are discounted per second. The reward is the ground-truth
  **delay rate** (Σ vehicles × (1 − v / v_max)) on the junction's approaches, plus half the delay on its exit
  roads, so a junction pays for the jams it pushes downstream. Eight SUMO worker processes stream experience to one
  learner. Every episode randomises the layout, demand level, time-of-day profile, OD imbalance, ambulance rate and
  **camera noise**. Exploration mixes random phases with max-pressure actions. A numbered checkpoint is saved every
  2,000 updates and validated on training layouts with held-out seeds; the deployed policy is the best checkpoint by
  delay relative to actuated control (`chowkpilot/rl/select.py`, `models/selection.json`).
- **Deployment form**: pure NumPy inference (`NumpyPolicy`), no PyTorch: about 0.15 ms per decision on a laptop CPU (measured).

**Two failures we found and fixed (both visible in the git history):**
1. *Mean pooling starves arterials.* Our first scorer averaged over a phase's released movements. On a 3-signal
   arterial it learned to hold protected right-turn phases (2 congested turn bays looked as valuable as 6 busy
   through movements), and traffic gridlocked: 244 vehicles were still stuck 20 minutes after demand ended. Max-pressure
   sums, so we switched to sum pooling and gave the network the max-pressure score as a prior.
2. *Max-green is not enough for safety.* A policy can alternate between two phases forever and starve a third. We
   added a max-red guarantee (below) that holds for every controller.

### 4. Safety layer and ambulance green corridor
Every adaptive controller, learned or not, acts through the same state machine in `chowkpilot/env/sim.py`, which
enforces the constraints whatever the controller requests:

- minimum green 10 s and maximum green 60 s; yellow 3 s, then all-red 1 s;
- **max red**: a movement with stopped vehicles gets green within 120 s;
- **ambulance green corridor**: an ambulance detected within 150 m of a stop line gets its phase after 5 s of green
  on the current phase (or keeps its green).

The unit tests attack this layer with an adversarial random controller and check every invariant
(`tests/test_core.py`).

## Results

<!-- RESULTS -->

## Live demo

**https://abderrahmanetb.github.io/chowkpilot/** is a static control room that replays recorded SUMO
simulations side by side (same vehicles, baseline vs ChowkPilot):

- **Digital twin:** real road geometry, vehicles coloured by type, signal state at every stop line, synchronised
  zoom and pan, and live counters for stopped vehicles and vehicle-minutes lost.
- **Why this phase?** ChowkPilot's learned score for every phase at each decision, labelled with real street
  names, so you can see its reasoning.
- **Benchmark:** the full results on the unseen junctions, with confidence intervals.
- **Stress-test:** recorded runs under off-peak, heavy, surge and camera-noise conditions.

To run *new* simulations from the browser, start the local server (below) and the same page shows a live
simulation form.

<!-- SCREENSHOTS -->

## Run it yourself

Requirements: Python 3.12 (Windows, Linux or macOS). SUMO comes as a pip wheel (`eclipse-sumo`), so there is
nothing else to install.

```bash
git clone https://github.com/AbderrahmaneTB/chowkpilot.git
cd chowkpilot
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # runtime (simulation + app)
pip install -r requirements-train.txt                   # optional: training and tests

python -m pytest -q                                     # tests, including the safety invariants
uvicorn app.server:app --port 7860                      # control room + live simulations at http://localhost:7860
```

Reproduce the whole pipeline (each step writes into the repo; these are the exact commands behind the results):

```bash
python -c "from chowkpilot.scenarios.registry import build_all; build_all()"   # digital twins (OSM → SUMO)
python -m chowkpilot.scenarios.calibrate                       # per-junction peak demand + Webster tuning
python -m chowkpilot.rl.train --baselines-only                 # baseline scores for the validation tasks
python -m chowkpilot.rl.train --minutes 75 --workers 8 --out-weight 0.5   # train (CPU only, laptop)
python -m chowkpilot.rl.select                                 # pick the best checkpoint on validation
python -m chowkpilot.eval.benchmark --seeds 10                 # benchmark on the unseen junctions
python -m chowkpilot.eval.make_replays                         # record the demo replays
python -m chowkpilot.eval.export_site                          # data for the static site
python -m chowkpilot.eval.figures && python -m chowkpilot.eval.readme_results   # figures + README tables
python -m chowkpilot.eval.make_gif && python -m chowkpilot.eval.make_video      # demo GIF + video
```

Quick single run from Python:

```python
from chowkpilot.eval.runner import run_one
r = run_one("hyd_hitech", "chowkpilot", seed=1, horizon=1800, sim={"ev_priority": True})
print(r["mean_delay"], r["co2_kg"], r["by_type"]["two_wheeler"])
```

Docker (live backend, port 7860): `docker build -t chowkpilot . && docker run -p 7860:7860 chowkpilot`.

## Project structure

```
chowkpilot/
  scenarios/   OSM download + netconvert, synthetic layouts, vehicle types, demand, calibration, signal programs
  env/         junction abstraction (movements × phases) and the SUMO driver with the safety layer
  control/     fixed-time, Webster, actuated, max-pressure and the ChowkPilot controller
  rl/          phase-scoring network (PyTorch for training, NumPy for inference), agent, distributed DQN trainer,
               checkpoint selection
  eval/        runner, paired benchmark with CIs, replays, static-site export, figures, README tables, GIF/video
app/server.py  FastAPI backend: serves docs/ and runs live simulations
docs/          the control room (static site, published with GitHub Pages) and its data
scenarios/     generated SUMO networks, raw OSM extracts, calibration (index.json)
models/        deployed policy (chowkpilot.npz), training log, checkpoint selection, baseline validation scores
results/       benchmark.json (all runs, per seed)
tests/         unit tests
```

## Tech stack

| Layer | Tools |
|---|---|
| Simulation | Eclipse SUMO 1.28 via `libsumo` (in-process), sublane model, HBEFA4 emissions |
| Maps | OpenStreetMap (Overpass API) → SUMO `netconvert` |
| Learning | PyTorch 2 (training), NumPy (inference), Double DQN, deep-sets phase scorer |
| Evaluation | NumPy, SciPy (paired t-tests, 95% CIs), multiprocessing |
| App | FastAPI + Uvicorn (live backend); vanilla JS + Canvas 2D (static control room); GitHub Pages |

## Model card

| | |
|---|---|
| **Task** | Choose the next green phase of a signalised junction every 5 s (after the minimum green) |
| **Inputs** | Per movement: stopped and moving vehicles, occupancy and mean waiting time in the last 150 m of each lane, exit-road occupancy, ambulance flag, lane count, currently green. Per phase: is current, time in green, time since last served, share of movements, max-pressure score |
| **Output** | A score per candidate phase; the safety layer applies the highest-scoring *allowed* phase |
| **Model** | Deep-sets phase scorer, 38k parameters, NumPy inference (about 0.15 ms per decision) |
| **Training data** | Simulated only: 11 synthetic layouts and 2 real Hyderabad areas (Gachibowli, Abids) in SUMO, randomised demand, time-of-day profiles, ambulances and camera noise |
| **Evaluation** | 3 unseen real Hyderabad areas, 10 paired seeds, 4 baselines, robustness conditions (see Results) |
| **Intended use** | Decision support and pilots for traffic engineers: shadow mode first, then supervised control at camera-equipped junctions |
| **Out of scope** | Unsupervised deployment without local validation; junctions with pedestrian-only phases or rail crossings; enforcement or tracking of individuals |
| **Data and privacy** | Aggregate per-lane counts only; no images, number plates or personal data are stored or needed |

## Limitations and responsible use

- **Simulation, not the street.** Results come from SUMO digital twins. Real deployment needs validation against
  local counts, a hardware-in-the-loop trial and traffic-police approval. Demand is synthetic, calibrated to
  near-capacity peak conditions; it is not measured traffic.
- **OSM data quality.** Lane counts, turn restrictions and signal positions come from OpenStreetMap and may not
  match the street exactly. Signal phase plans are guessed by `netconvert`.
- **Pedestrians are not modelled.** The minimum green (10 s) and all-red clearance are conservative placeholders
  for pedestrian phases, which a deployment would add.
- **Camera perception is assumed, not built.** We model camera counting errors (noise and missed vehicles) and
  test robustness to them, but do not ship a vision model.
- **Fallback.** On sensor failure, a deployment should revert to the junction's fixed-time plan; the safety layer
  enforces minimum/maximum greens and clearance whatever the policy outputs.
- **No personal data.** The controller uses aggregate counts per lane; no number plates or identities.

## AI development tools disclosure

As the NeuralStorm rules require, we disclose that this project was developed with the help of an AI coding
assistant, **Claude Code (Anthropic)**, used for research, code generation, debugging and documentation under
the team's direction. All design decisions, experiments and results were run and checked in this repository;
every number in this README is produced by the scripts above.

## Acknowledgements and licences

- [Eclipse SUMO](https://eclipse.dev/sumo/) (EPL-2.0), used as an installed dependency (`eclipse-sumo`, `libsumo`, `sumolib`, `traci`).
  The Webster baseline uses SUMO's `tlsCycleAdaptation.py`.
- Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under the Open Database License (ODbL).
- Methods: P. Varaiya, *Max pressure control of a network of signalized intersections*, Transportation Research Part C, 2013;
  H. Wei et al., *PressLight*, KDD 2019; C. Chen et al., *Toward A Thousand Lights* (MPLight), AAAI 2020;
  H. Wei et al., *CoLight*, CIKM 2019; G. Zheng et al., *FRAP*, CIKM 2019; H. van Hasselt et al., *Double Q-learning*, AAAI 2016.
- Our code is released under the [MIT licence](LICENSE).
