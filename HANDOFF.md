# ChowkPilot handoff (Team Blackwater)

Written 3 Nov 2026, 02:50 Algeria time, at the end of session 1. **Delete this file before the final submission.**

- **Deadline:** Wed 4 Nov 2026, **8:00 PM IST** = 15:30 Algeria (UTC+1) = 14:30 UTC. Aim to submit by **12:00 Algeria**.
- **Repo:** https://github.com/AbderrahmaneTB/chowkpilot (public, required by the rules)
- **Live demo:** https://abderrahmanetb.github.io/chowkpilot/ (GitHub Pages from `docs/`; its data is **not published yet**, see P0)
- **Track:** Track 1 Future Tech, **FT-01 Smart Traffic Signal Optimisation**.
- **Judging:** Technical 35%, Impact 25%, README 20%, Demo 20%.
- **Team:** Abderrahmane Tayeb Boudjema (@AbderrahmaneTB, repo owner) and Ouail Imad Eddine Zitouni, known as Wael (@WaelYTz, collaborator with push access).

## Session 2 update (3 Nov, afternoon, AbderrahmaneTB's laptop)

- The overnight run of `finalize.sh` died at benchmark run 60/480 (the laptop was switched off). None of its outputs
  were used.
- **Deployed model: run 5b's 3-network ensemble** (Wael's GPU run, branch `run5-stable-gpu`, merged into main).
  - Chosen by `tools/head_to_head.py` on 3 fresh validation seeds (9191, 10101, 11111) that nothing else used:
    run 5b 0.853 vs run 4 0.898.
  - The rule was fixed before running: lower wins, and a gap under 0.01 keeps run 4.
  - The choice was committed (`a413671`) before any benchmark run; the test areas played no part.
  - Run 4's model and records are in `models/run4/`. The result is in `models/head_to_head.json`.
- README and site updated for the ensemble: 10 movement inputs, 91,011 parameters, run 5b's training and selection,
  and isolated junctions as a stated limitation.
- Site design: Barlow (signage) typography, a fixed architecture diagram, focus states. CI is green.
- `SKIP_SELECT=1 bash tools/finalize.sh 10` ran from 15:14 to 16:12 on the ensemble: 480 runs, 0 errors.
  - Results, replays, figures, GIF, video (48 s) and screenshots are published.
  - The live site was checked in headless Chrome: all 52 data files load, no console errors, no sideways scroll at
    390 px.
  - The README headline and per-junction reading are generated from `results/benchmark.json`.
- **Results in one line:** −59% delay vs fixed-time (significant at all 3 junctions) and −21% vs max-pressure. Level
  with actuated control (−2%, not significant). The exact-demand Webster plan is best at 2 of 3 areas.
- **Still to do (people):**
  1. Full names: done in the README, licence and site footer; use the same on Devpost.
  2. Upload `docs/img/chowkpilot_walkthrough.mp4` (2.5 min, narrated, made by `tools/make_walkthrough.py`) to
     YouTube and add the link to the README and Devpost. `chowkpilot_demo.mp4` (48 s, no sound) is a short teaser.
  3. Create the Devpost entry from the text below; submit only after both of you agree.
  4. Delete this file in the final commit.

## Rules we must keep (from the team brief)

- Commit with your own git identity (AbderrahmaneTB or WaelYTz). **No `Co-authored-by: Claude` trailers, no AI
  contributor, collaborator or team-member entries, no "Built by Claude" branding.** The two humans are the authors.
- The AI-tools disclosure stays in its own README section, worded truthfully. The rules require it.
- **Never invent results.** Every number in the README, the site, the video and Devpost comes from a script in this repo.
  No mock demos presented as real.
- No paid services without the team's approval. HF Docker Spaces need PRO, which is why we use GitHub Pages.
- Logins, Devpost submission and the YouTube upload are done by a team member, not by the assistant.
- **Do not submit the final Devpost entry until the team explicitly approves it.** Ask first, show what will be
  submitted, and wait for a clear yes.

## Where things stand

Done and pushed:
- SUMO digital twins: 11 synthetic layouts plus 5 real Hyderabad OSM areas. Train: Gachibowli, Abids.
  Test, never used for training or selection: Secunderabad, Hitech City, Ameerpet.
- Indian mixed traffic: two-wheeler 52%, car 27%, auto 12%, light truck 5%, bus 4%; sublane model, left-hand driving,
  assertive merging.
- Seeded, paired demand; per-junction calibration.
- Baselines: fixed-time (standard plan), Webster, SUMO actuated, max-pressure.
- ChowkPilot: a deep-sets phase scorer (30,209 parameters, NumPy inference) trained with Double DQN (semi-Markov),
  8 SUMO workers and domain randomisation.
- Safety layer for every adaptive controller: min green 10 s, max green 60 s, **max red 120 s**, yellow 3 s,
  all-red 1 s, ambulance green corridor. Unit tests attack it with an adversarial controller.
- Benchmark tooling: paired seeds, 95% CIs, paired t-tests, robustness conditions, impact (vehicle-hours, CO₂).
- Static control room in `docs/`:
  - side-by-side digital twin, "why this phase", benchmark, stress tests;
  - colour-blind-safe palette, works on phones;
  - deep links: `#scenario=hyd_hitech&baseline=actuated&t=600&junction=1&view=twin`.
- Optional FastAPI live backend (`app/server.py`), Dockerfile, README with method and model card, MIT licence.

Training history (validation = training layouts with held-out seeds; score = delay / actuated delay, lower is better):

| Run | Change | Outcome |
|---|---|---|
| 1 | mean pooling | arterial gridlock in validation (max wait 1,732 s, 294 vehicles unfinished) |
| 2 | + max-red safety rule, neighbourhood reward | still gridlocked; a diagnostic re-run showed 244 vehicles stuck 20 min after demand ended (the figure the README cites). Cause: mean pooling favoured 2-movement turn phases |
| 3 | **sum pooling + max-pressure score as an input** | fixed; best score about 0.93, then drifted after about 25 min |
| 4 | checkpoints every 2,000 updates, cosine LR, ε floor 0.1, 400k buffer | 75 min, the current model (scores below) |

Run 4: every validated checkpoint (74.4 min of training, 15 validation rounds):

| Checkpoint | Score | single@4242 | single@5252 | arterial@4243 | arterial@5253 | gachibowli@4244 |
|---|---:|---:|---:|---:|---:|---:|
| u002000 (selected, first pass) | **0.906** | 31.9 | 75.9 | 120.4 | 50.2 | 180.3 |
| u006000 | 1.007 | 28.4 | 88.6 | 136.4 | 58.0 | 212.1 |
| u008000 | 0.980 | 31.0 | 84.3 | 131.0 | 58.0 | 197.6 |
| u012000 | 0.977 | 30.0 | 100.3 | 108.8 | 57.2 | 196.9 |
| u016000 | 0.949 | 29.1 | 100.6 | 104.3 | 57.4 | 185.6 |
| u018000 | 0.982 | 29.7 | 96.3 | 123.9 | 51.1 | 202.0 |
| u022000 | 0.986 | 28.3 | 102.3 | 99.3 | 55.4 | 216.2 |
| u024000 | 0.986 | 28.8 | 112.2 | 95.7 | 54.4 | 204.6 |
| u028000 | 0.977 | 29.1 | 108.2 | 98.2 | 52.0 | 204.9 |
| u030000 | 1.032 | 28.2 | 109.6 | 102.1 | 61.2 | 224.8 |
| u036000 | 1.068 | 30.1 | 113.6 | 108.1 | 62.6 | 228.3 |
| u038000 | 1.023 | 29.0 | 112.2 | 102.6 | 55.0 | 220.7 |
| u040000 | 1.022 | 31.3 | 112.0 | 108.9 | 54.1 | 208.2 |
| u042000 | 0.996 | 29.4 | 102.1 | 109.6 | 53.6 | 212.5 |
| u044000 | 1.037 | 31.3 | 106.7 | 115.5 | 58.3 | 213.5 |
| *actuated (reference)* | 1.000 | 31.1 | 73.7 | 120.0 | 76.4 | 209.1 |
| *fixed-time* | 1.641 | 48.1 | 88.4 | 224.3 | 88.8 | 426.1 |
| *max-pressure* | 1.091 | 32.1 | 110.8 | 130.9 | 52.7 | 233.6 |

What this tells us:
- The best checkpoint beats fixed-time on every validation task (14–58% less delay) and beats max-pressure on average.
- It is roughly level with SUMO's vehicle-actuated control, our strongest baseline, and that control uses ideal loop
  detectors.
- Checkpoints u012000–u022000 beat actuated on the arterial (99–109 s vs 120 s on seed 4243) and on single@4242.
  They consistently lose on the isolated 2-lane junction (96–102 s vs 74 s on seed 5252). That is a systematic weakness
  on isolated junctions, not noise; see P1 item 2.
- One seed per task is noisy, so use `select --revalidate`.
- **The run is best early and then drifts.** u002000 (11 min, ε still 0.7) scores 0.906. After about 30k updates
  scores rise to 0.98–1.07, and single@5252 degrades steadily (76 → 113 s). Training longer as-is will not help.

Full log: `models/train_log.json`. All checkpoints: `models/checkpoints/`.

## P0: finish a complete, honest submission first (≈ 3 h, safety net)

**First check `results/finalize.log`.** Session 1 started `bash tools/finalize.sh` (all the steps below) on the
first laptop at about 02:52 on 3 Nov.
- If the log ends with `finalize done`, every output exists: review them (numbers, GIF, video, README tables) and go
  to the "Then" list.
- If it stopped part-way (laptop asleep or window closed), re-run the remaining steps, or the whole script:
  `PYTHON=<venv python> bash tools/finalize.sh 10`.

Run in the repo root. Prefix commands with `PYTHONIOENCODING=utf-8` on Windows. The selection step takes about 8 min.
To check the pipeline end to end before the full benchmark, run it once with `--seeds 2 --skip-robustness`
(about 10 min).

```bash
python -m chowkpilot.rl.select --revalidate 3 --ensemble --workers 10   # robust pick (validation only) -> models/chowkpilot.npz
python -m chowkpilot.eval.benchmark --seeds 10 --workers 10   # 3 unseen junctions -> results/benchmark.json (~1-1.5 h, ~480 runs)
python -m chowkpilot.eval.make_replays --workers 10           # demo replays -> docs/data/replays (~15 min)
python -m chowkpilot.eval.export_site                         # docs/data/scenarios.json, network/, benchmark.json
python -m chowkpilot.eval.figures                             # docs/img/*-light.png / *-dark.png
python -m chowkpilot.eval.readme_results                      # fills <!-- RESULTS-HERO --> and <!-- RESULTS -->
python -m chowkpilot.eval.make_gif --scenario hyd_hitech --baseline fixed   # docs/img/demo_hyd_hitech.gif
python -m chowkpilot.eval.make_video                          # docs/img/chowkpilot_demo.mp4 (for YouTube)
python -m http.server 8090 --directory docs                   # in another terminal, then:
python tools/screenshots.py                                   # docs/img/screen_*.png via headless Chrome
```

Then:
1. Remove `docs/data/replays/` from `.gitignore`.
   - The local replays and `docs/img/demo_hyd_hitech.gif` / `learning_curve-*.png` are **stale, from run 1** (before both fixes): in them
     ChowkPilot is far worse than fixed-time. Regenerate them all before committing; never publish the old ones.
2. README:
   - Put the GIF right under the intro and fill `<!-- SCREENSHOTS -->` with `screen_twin.png`, plus `screen_mobile.png`
     at a small width.
   - Write a short **"Reading the results"** paragraph per junction, honest about any junction where actuated wins.
   - In the training bullet, mention the re-validation (top 3 re-run on seeds 6262/7272/8282).
3. Check the hero claim generated by `readme_results.py` against `results/benchmark.json`. If ChowkPilot does not beat a
   baseline on average, the hero sentence must not say it does. Adjust the generator, never the numbers.
4. Commit and push. Open the live site on desktop and phone: replays load, KPIs are filled, no console errors.
5. Ask the team for full names (README line 4 and Devpost).

## P1: improve the model (decide on validation only; run the test benchmark once per release)

The test junctions must stay clean: choose between models with `select --revalidate` scores, never with
`results/benchmark.json`. Ranked by expected gain per hour:

1. **Checkpoint ensemble: measured, no gain for run 4.** Overnight, `select --revalidate 3 --ensemble` scored:
   - u002000: **0.922** (first pass 0.906), still deployed;
   - u016000: 1.004 and u012000: 1.005, level with actuated;
   - their mean-Q ensemble: 0.951.

   Re-try the ensemble on any new run's checkpoints; the code is ready (`NumpyPolicy` loads `m0/…`, `m1/…`).
2. **Give the policy the information actuated control uses.** Actuated wins on the isolated junction because it reacts
   to gaps in arrivals. Add per-movement features:
   - time since a vehicle last crossed the stop line (a gap measure);
   - vehicles arriving within the next 50 m (`MOVEMENT_FEATURES` in `chowkpilot/env/sim.py`, `FM` in
     `chowkpilot/rl/train.py`).
   - A finer decision interval: `SimConfig.delta` from 5 s to 2–3 s. Actuated control checks gaps every second;
     ChowkPilot can only extend a green in 5 s steps.
   - This needs a new training run. Update the tests in `tests/test_core.py`.
3. **A more stable training run on Wael's machine (NVIDIA GPU).** Run 4 drifts after its first ~16k updates, so the
   aim is not just more minutes. Try a lower learning rate and slower target (`--lr 1e-4 --tau 0.002`),
   `--ckpt-every 500` so the good early region gets more checkpoints, and the item-2 features. SUMO runs on the CPU,
   so the number of CPU cores sets the speed; the GPU frees those cores by running the learner:

   ```bash
   python -m chowkpilot.rl.train --baselines-only --out models_w           # validation reference for this folder
   python -m chowkpilot.rl.train --minutes 90 --workers <cores - 2> --threads 2 --device cuda --lr 1e-4 --tau 0.002 \
          --ckpt-every 500 --replay-ratio 0.5 --out-weight 0.5 --out models_w
   python -m chowkpilot.rl.select --models models_w --revalidate 3 --ensemble --workers <cores>
   ```

   - Compare its re-validated score with run 4's.
   - If it wins, copy `models_w/{chowkpilot.npz, train_log.json, selection.json, baseline_validation.json,
     checkpoints/}` into `models/` and re-run P0.
   - Google Colab is not worth it: the free tier has 2 CPU cores, slower than either laptop.
4. **A real-junction validation split** (more rigour for the judges):
   - Download 2 more Hyderabad areas (e.g. Mehdipatnam, Kukatpally/KPHB, LB Nagar, Punjagutta) with `scenarios/osm.py`.
   - Add them to `OSM_AREAS` with a new `split="val"`: excluded from training in `worker_main`, used in `VALIDATION`
     and `select`. Calibrate with `python -m chowkpilot.scenarios.calibrate`.
5. Optional: a small switching penalty in the reward, behaviour-cloning warm start from max-pressure, other seeds.

## P2: tests and engineering quality

- Done: GitHub Actions CI (`.github/workflows/tests.yml`; badge in the README). The first run passed: 1 min to
  install, 7 s of tests on Linux.
- Done: 9 tests covering
  - policy NumPy/PyTorch equivalence, ensembles, deterministic demand, the junction model, safety invariants;
  - selection scoring, 95% CIs against SciPy, the replay round trip, every scenario being signalised.
- Done: `tests/test_api.py` (FastAPI `TestClient`, health check plus a short paired `/api/simulate`).
  - The simulate test needs `models/chowkpilot.npz`.
- Done: `paired_stats` (per-seed % change, paired t-test, wins) has its own test against SciPy.
- End-to-end smoke test before the release: `python -m chowkpilot.eval.benchmark --seeds 1 --skip-robustness --horizon 300`.
- Site checks:
  - every replay listed in `docs/data/scenarios.json` loads;
  - no console errors;
  - README links resolve;
  - Lighthouse accessibility score is acceptable.

## P3: design polish (README 20% + demo 20%)

- README: the tests badge is done (add licence and live-demo badges if wanted), GIF at the top, screenshots, maybe an SVG version of the ASCII
  architecture diagram, and a results narrative a non-expert can follow in 30 seconds.
- Site: the link-preview tags are in place and point to `docs/img/screen_hero.png`; `tools/screenshots.py` generates
  that image, so commit it. Optionally add a "copy link to this view" button (deep links already work).
- Video (about 75 s, `docs/img/chowkpilot_demo.mp4`): a team member uploads it to YouTube (unlisted is fine). Its link
  goes in Devpost and the README.
- Devpost gallery: use the `screen_*.png` images and the GIF.

## Devpost text (final numbers from `results/benchmark.json`, generated 3 Nov 16:09)

Copy each part into the matching Devpost field. Re-check the numbers if the benchmark is ever re-run.

**Project name:** ChowkPilot

**Elevator pitch (under 200 characters):** One reinforcement-learning signal brain for any Indian junction. It runs on
camera counts and cut delay by 59% vs fixed-time plans on real Hyderabad junctions it never saw.

**Links:**
- Repo: https://github.com/AbderrahmaneTB/chowkpilot
- Live demo: https://abderrahmanetb.github.io/chowkpilot/
- Video: (YouTube link, after the upload)

**About the project**

## Inspiration
Hyderabad drivers lose 123 hours a year to rush-hour congestion (TomTom Traffic Index 2026). Most signals still run
fixed-time plans that cannot react to the traffic in front of them. The traffic itself is mixed and lane-less: 73%
of Telangana's vehicles are two-wheelers. Controllers built for lane-disciplined car traffic and loop detectors fit
it badly, and ambulances lose minutes at red lights.

## What it does
- Every 5 seconds ChowkPilot decides which approach of a junction gets green.
- It uses only what a roadside camera can measure in the last 150 m of each lane.
- One small learned policy runs any junction: 3-arm, 4-arm, 5-arm, or a joined cluster of signals.
- A fixed safety layer enforces minimum and maximum green, a 120 s max-red guarantee and yellow plus all-red
  clearance, whatever the policy outputs. An approaching ambulance gets a green corridor.
- The live control room replays real SUMO runs side by side (same vehicles, baseline vs ChowkPilot). It explains
  every phase decision with street names, and shows the full benchmark and stress tests.

## How we built it
- **Digital twins:** real Hyderabad areas from OpenStreetMap, converted with SUMO for left-hand traffic. SUMO's
  sublane model lets two-wheelers and autos filter between cars. Indian vehicle mix and assertive merging.
- **The policy:** a learned version of max-pressure control. A shared network embeds each movement and sums the
  embeddings over the movements each phase releases. It receives the max-pressure score as a prior.
- **Training:** Double DQN with semi-Markov rewards on true vehicle delay, using 10 SUMO workers and a GPU learner,
  with randomised layouts, demand, time-of-day profiles, ambulances and camera noise.
- **Selection, on validation only:** dense early checkpoints, re-validation on extra seeds, and an ensemble of the 3
  best (91,011 parameters in total). A final check against our previous model ran on fresh seeds.
- **Evaluation:** 3 unseen real Hyderabad areas, 10 paired seeds (every controller sees the same vehicles), 95% CIs,
  paired t-tests and 5 stress conditions.
- **Product:** the control room is a static site on GitHub Pages; an optional FastAPI backend runs new simulations.
  CI runs 14 tests on every push.

## Results (unseen junctions, averaged over the 3 areas)
- **59% less delay** than the standard fixed-time plan, significant at all 3 junctions, winning 29 of 30 seeds.
  That is 555 vehicle-hours and 1.4 t of CO₂ saved per simulated peak across the three areas.
- **21% less delay** than max-pressure control.
- **Level with vehicle-actuated control:** 2% lower on average, not significant at any junction. That baseline
  gets ideal loop detectors; ChowkPilot sees camera counts.
- A Webster plan re-timed with each run's exact demand, which no city has in advance, is best at 2 of the 3 areas.
- **24% less CO₂** and **67% less ambulance delay** than fixed-time.
- Best in 4 of 5 stress tests (off-peak, peak, sudden surge, 20% camera noise) and level with actuated in heavy
  demand.

## Challenges we ran into
- Our first network averaged over the movements a phase releases. It learned to favour 2-lane turn bays and
  gridlocked an arterial. Summing, as max-pressure does, fixed it.
- A policy could starve a third phase, so we added a max-red guarantee for every controller.
- Every training run peaked early and then drifted, so we keep dense early checkpoints and choose on validation.
- Isolated junctions remain the weak spot. Vehicle-actuated control still wins there, which the README states.

## Accomplishments that we're proud of
- Zero-shot results on real junctions it never saw, with Indian mixed traffic.
- An honest benchmark: paired seeds, significance tests, and the model chosen before any test run.
- Explanations a traffic engineer can read.
- A fully reproducible pipeline from OpenStreetMap to the website.

## What we learned
Sharing weights across movements and phases makes one policy portable across junction layouts. Most of the gain over
fixed plans comes from reacting to live queues. Beating well-tuned actuated control needs more than that.

## What's next
- Shadow-mode trials on real camera counts at a few Hyderabad junctions.
- Pedestrian phases.
- Better isolated-junction control.
- Coordination along corridors.

**Built with:** python, eclipse-sumo, openstreetmap, pytorch, numpy, scipy, fastapi, javascript, html5-canvas,
github-pages, github-actions

**AI-tools disclosure:** use the README's wording. Claude Code (Anthropic) was used for research, code generation,
debugging and documentation under the team's direction. All experiments were run, and all results produced, by the
scripts in the repository.

## Machine notes

- Session 1 machine: Windows 11, 12 threads; venv at `C:\Users\HP\.venvs\pravah` (outside OneDrive on purpose).
- New machine:

  ```bash
  git clone https://github.com/AbderrahmaneTB/chowkpilot.git && cd chowkpilot
  python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
  pip install -r requirements-train.txt
  python -m pytest -q
  ```

  For a CUDA learner, install the CUDA build of torch 2.15 that matches the driver (pytorch.org, "Get started").
- Pitfalls:
  - On Windows, set `PYTHONIOENCODING=utf-8` before printing results.
  - If a training run is killed, end the leftover worker processes too.
  - `libsumo` sets `SUMO_HOME`; tool paths come from `chowkpilot/sumo_paths.py`.
  - The Overpass API needs a User-Agent header (already set).

## Prompt for the next Claude Code session

Paste this into Claude Code, opened in the repo root:

```text
You are continuing ChowkPilot, Team Blackwater's entry to the NeuralStorm 2026 hackathon (Track 1, problem FT-01
Smart Traffic Signal Optimisation). The goal is first place overall. Read HANDOFF.md in the repo root first and
follow it: it has the state of the project, the rules, the commands and the prioritised plan.

Deadline: Wed 4 Nov 2026, 8:00 PM IST (15:30 Algeria, UTC+1). Aim to submit by 12:00 Algeria.

Rules:
- Commit with this machine's git identity.
- Never add Co-authored-by: Claude or any AI co-author trailer, AI contributor, collaborator or team member,
  or "Built by Claude" branding.
- Keep the README's AI-tools disclosure truthful.
- Never invent or hand-type results; every number comes from the scripts.
- No paid services without asking.
- Logins, the Devpost submission and the YouTube upload are done by the team.
- Never submit the final entry without asking the team first and getting an explicit yes.

Who does what: run `git config user.name` and `git pull` first.
- AbderrahmaneTB's laptop owns P0 and main-branch releases. Start by checking results/finalize.log: an overnight
  run of tools/finalize.sh may already have produced every result.
- WaelYTz's machine (NVIDIA GPU) does P1 item 3 first: a training experiment in models_w/ that it reports back.
  Push only to a branch, and merge only after the team agrees.

Work autonomously, in this order:
P0: finish a complete, honest submission with the current model (select --revalidate --ensemble, benchmark,
    replays, site export, figures, README tables, GIF, video, screenshots), push it and verify the live site.
P1: improve the model, judging every change on validation only; run the test benchmark once per release.
P2: the remaining tests and site checks (CI already runs on every push).
P3: README and site design polish, then the Devpost text with the final numbers.

Report progress briefly. Ask only for things that need a person: full names, logins, the video upload, the
final Submit click.
```
