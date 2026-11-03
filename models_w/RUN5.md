# Run 5: training experiment on the GPU machine (WaelYTz, 3 Nov)

HANDOFF P1 item 3. Everything below is validation only: training layouts with held-out seeds. The test areas
(Secunderabad, Hitech City, Ameerpet) were not run. Score = delay / vehicle-actuated delay, lower is better.

## Code changes on this branch

- **Gap features** (P1 item 2), two extra camera inputs per movement:
  - seconds since a vehicle last crossed the stop line (the gap that actuated control reacts to);
  - moving vehicles in the last 50 m.

  Inputs go from 8 to 10. Older models read only the first 8, so the current `models/chowkpilot.npz` gives the
  same results to the last digit, and the baselines are unchanged.
- **`train.py --balance 0.5`**: each episode adds one transition per junction, so with the default layout
  weights the two 9-junction grids fill about 45% of the replay buffer and the four isolated junctions about 5%.
  With `--balance 0.5` (layout weight = junctions^-0.5) the isolated junctions get about 17%. The default is
  unchanged.
- **`select.py --fill N [--fill-max U]`**: training validates only the latest checkpoint, so most checkpoints
  are never scored. These options score them on the same validation tasks before ranking.

## Runs (10 SUMO workers, learner on the GPU)

Common settings: `--lr 1e-4 --tau 0.002 --ckpt-every 1000 --replay-ratio 0.25 --out-weight 0.5`, gap features.

- The GPU learner did about 4,600 updates/min against 660 for run 4. Updates per transition were about 0.20,
  against 0.05 for run 4, whose CPU learner could not keep up with its nominal 0.25.
- Run 4's drift came back in every run, only stretched over more updates by the lower learning rate. The best
  checkpoints come in the first 3k–12k updates (under 4 minutes).

Runs 5a, 5b and 5c were stopped by hand once validation had clearly degraded (the training schedule was set to
55 minutes). Validation during training:

| Run | Layout weights | Seed | Validated checkpoints: score (isolated junction seed 5252, s; actuated 73.7) |
|---|---|---|---|
| 5a | default | 0 | u005000 at 2.1 min: 0.917 (76), u014000 at 4.2 min: 0.951 (90), u022000 at 5.7 min: 0.966 (100) |
| 5b | --balance 0.5 | 0 | u003000 at 1.7 min: 0.892 (82), u011000 at 3.3 min: 0.914 (81), u019000 at 4.7 min: 0.995 (99), u026000 at 6.4 min: 1.011 (109), u035000 at 8.4 min: 1.024 (98) |
| 5c | --balance 0.5 | 1 | u006000 at 2.5 min: 0.962 (85), u016000 at 4.7 min: 1.000 (108), u025000 at 6.6 min: 0.996 (109) |
| 5d | --balance 0.5, --delta 3 | 0 | u003000 at 1.9 min: 1.162 (97), u012000 at 3.8 min: 0.932 (84) |

Run 5d also decides every 3 s instead of 5 s (`train.py --delta 3`, stored in the policy file).

Logs: run 5a `models_w/run5a_train_log.json`, run 5b `models_w/train_log.json`, run 5c `models_w_s1/train_log.json`,
run 5d `models_w_d3/train_log.json`. Each log ends at its last 60-second write before the stop. Run 5b's
checkpoints are committed; those of 5c and 5d stay on the GPU laptop.

## Selection (`select --fill 1000 --fill-max 20000 --revalidate 3 --ensemble`)

- "All 4 seeds" is the selection criterion: the first-pass seed plus seeds 6262, 7272 and 8282.
- "Extra seeds only" leaves out the seed the candidates were picked on, so it is the fair comparison between runs.
- The per-layout columns are averaged over all 4 seeds.

| Run | Candidate | All 4 seeds | First pass | Extra seeds only | Isolated | Arterial | Gachibowli |
|---|---|---:|---:|---:|---:|---:|---:|
| run 4 (`models`) | **u002000** (deployed now) | 0.922 | 0.906 | 0.926 | 1.066 | 0.777 | 0.923 |
| run 4 | u012000 | 1.005 | 0.977 | 1.024 | 1.302 | 0.767 | 0.948 |
| run 4 | u016000 | 1.004 | 0.949 | 1.035 | 1.325 | 0.749 | 0.939 |
| run 4 | ensemble (u002000 + u016000 + u012000) | 0.951 |  | 0.968 | 1.151 | 0.800 | 0.902 |
| 5b (`models_w`) | u003000 | 0.899 | 0.892 | 0.899 | 1.200 | 0.712 | 0.787 |
| 5b | u005000 | 0.919 | 0.893 | 0.936 | 1.136 | 0.765 | 0.856 |
| 5b | u009000 | 0.906 | 0.904 | 0.908 | 1.052 | 0.715 | 0.950 |
| 5b | **ensemble (u003000 + u005000 + u009000)** (selected) | 0.896 |  | 0.911 | 1.078 | 0.761 | 0.848 |
| 5c (`models_w_s1`) | u005000 | 0.940 | 0.895 | 0.961 | 1.095 | 0.751 | 0.974 |
| 5c | u007000 | 0.975 | 0.936 | 0.996 | 1.169 | 0.733 | 1.025 |
| 5c | u011000 | 0.986 | 0.947 | 1.009 | 1.175 | 0.805 | 0.979 |
| 5c | **ensemble (u005000 + u007000 + u011000)** (selected) | 0.921 |  | 0.925 | 1.072 | 0.739 | 0.952 |
| 5d (`models_w_d3`) | u008000 | 0.960 | 0.911 | 0.987 | 1.228 | 0.716 | 0.937 |
| 5d | u009000 | 0.968 | 0.913 | 0.998 | 1.191 | 0.753 | 0.962 |
| 5d | **u013000** (selected) | 0.909 | 0.905 | 0.910 | 1.108 | 0.724 | 0.895 |
| 5d | ensemble (u013000 + u008000 + u009000) | 0.929 |  | 0.932 | 1.068 | 0.776 | 0.944 |

Generated from the four `selection.json` files.

## Reading

- **Run 5b's ensemble is the best-validated model so far.** It scores 0.896 on all seeds and 0.911 on the extra
  seeds, against 0.926 for the deployed run 4 model.
  - The gain is on the multi-junction layouts (Gachibowli 0.848 vs 0.923, arterial 0.761 vs 0.777).
  - The isolated junction is unchanged (1.078 vs 1.066), still behind actuated.
- **The gain is not reliable.** Run 5c uses the same recipe with another seed and lands level with run 4
  (0.921 / 0.925). Run 5b's margin is within seed-to-seed variance.
- **The 3 s decision interval (5d) lands in the same place as 5b**: 0.910 on the extra seeds, with the isolated
  junction at 1.108.
- **The isolated-junction weakness is not fixed.** Neither the gap features, the layout balancing nor the finer
  decision interval removed it.
  Hitech City, one of the three test areas, is a single junction, so expect ChowkPilot to be level with or behind
  actuated there.
- **The drift is driven by updates, not by exploration.** It started while ε was still about 0.6. A lower learning
  rate only spreads it over more updates. Picking an early checkpoint with dense validation (`--fill`) is
  therefore what makes the difference.

## Options for the team

- **Keep run 4** (`models/`, deployed now): no rerun needed.
- **Adopt run 5b's ensemble.** By the validation-only rule this is a legitimate choice, but the expected gain on
  the test areas is small and mostly on the multi-junction ones (Secunderabad, Ameerpet). It needs:
  1. merging this branch, since the model uses the 10-input code;
  2. copying `models_w/{chowkpilot.npz, train_log.json, selection.json, baseline_validation.json, checkpoints/}`
     into `models/`;
  3. re-running all of P0 (benchmark about 1–1.5 h);
  4. updating the parameter count in the README: 3 × 30,337 = 91,011 for the ensemble (30,337 for one network).

Run 5d's u013000 scores about the same (0.910) and is a single network. It is not a better choice: it was picked
among more candidates, and it is worse on the isolated junction.

## Environment

- Training ran in Python 3.11.9 with numpy 2.4.6 and torch 2.11.0+cu128, which were already installed on this
  machine. The 2 GB CUDA torch 2.15 wheel kept stalling on this connection.
- Selection ran in the pinned environment: Python 3.13, numpy 2.5.4, SUMO 1.28.0.
- The baseline validation recomputed here matches `models/baseline_validation.json` exactly.
