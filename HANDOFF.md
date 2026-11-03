# ChowkPilot handoff (Team Blackwater)

Written 3 Nov 2026, 02:50 Algeria time, at the end of session 1. **Delete this file before the final submission.**

- **Deadline:** Wed 4 Nov 2026, **8:00 PM IST** = 15:30 Algeria (UTC+1) = 14:30 UTC. Aim to submit by **12:00 Algeria**.
- **Repo:** https://github.com/AbderrahmaneTB/chowkpilot (public, required by the rules)
- **Live demo:** https://abderrahmanetb.github.io/chowkpilot/ (GitHub Pages from `docs/`; its data is **not published yet**, see P0)
- **Track:** Track 1 Future Tech, **FT-01 Smart Traffic Signal Optimisation**.
- **Judging:** Technical 35%, Impact 25%, README 20%, Demo 20%.
- **Team:** Abderrahmane (@AbderrahmaneTB, repo owner), Wael (@WaelYTz, invited as a collaborator; he must accept the invite before he can push).

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

1. **Checkpoint ensemble (implemented, not yet measured).** `select --revalidate 3 --ensemble` scores the mean-Q
   ensemble of the 3 best checkpoints on the same validation seeds. It deploys the ensemble only if it beats the best
   single checkpoint. `NumpyPolicy` loads ensembles saved as `m0/…`, `m1/…`, and `tests/test_core.py` covers it.
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

## Devpost text (fill numbers from `results/benchmark.json`)

> **ChowkPilot: one AI signal brain for every chowk.** A reinforcement-learning traffic-signal controller for Indian
> junctions. Every few seconds it decides which approach gets green, using only what a roadside camera can count. One
> 30k-parameter network runs any junction layout. It was trained in SUMO digital twins of synthetic junctions and two
> real Hyderabad areas, with lane-less two-wheelers, autos and buses, then tested zero-shot on three real Hyderabad areas
> it never saw.
>
> **Results** (10 paired seeds per junction): X% less delay than fixed-time, Y% vs vehicle-actuated, Z% less CO₂,
> W% less ambulance delay. Safety rules are enforced for every controller.
>
> **Built with:** Python, Eclipse SUMO (libsumo), OpenStreetMap, PyTorch, NumPy, SciPy, FastAPI, JavaScript (Canvas),
> GitHub Pages.
>
> Links: repo, live demo, video. AI-tools disclosure: same wording as the README section.

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

Work autonomously, in this order:
P0: finish a complete, honest submission with the current model (select --revalidate, benchmark, replays,
    site export, figures, README tables, GIF, video, screenshots), push it and verify the live site.
P1: improve the model, judging every change on validation only; run the test benchmark once per release.
P2: tests + GitHub Actions CI.
P3: README and site design polish, then the Devpost text with the final numbers.

Report progress briefly. Ask only for things that need a person: full names, logins, the video upload, the
final Submit click.
```
