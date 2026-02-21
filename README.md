# RL_sac (forest vehicle planning + tracking)

English | [简体中文](README.zh-CN.md)

This repo focuses on a forest-scene kinematic vehicle (Ackermann/bicycle) environment:
`forest_a`, `forest_b`, `forest_c`, `forest_d`.

Default conda environment: `ros2py310`.

## Repository migration note (2026-02-21)

- New GitHub repository: `git@github.com:unstun/RL_SAC.git`
- Historical stable baseline remains `v7p1` (`CNN-DDQN`) for rollback/comparison.
- SAC migration target is `v8` (`SAC-GlobalCNN`), with design/plan in:
  - `docs/plans/2026-02-20-sac-globalcnn-design.md`
  - `docs/plans/2026-02-20-sac-globalcnn-plan.md`
- This round updates documentation/repo routing only; no core train/infer algorithm code is changed.

## AI TL;DR (contract, 2026-02-21)

```text
STABLE_PROFILE=v7p1 (historical DQN baseline)
TRANSITION_TARGET=v8 (SAC-GlobalCNN)
CLAIM_REGIME=shielded/hybrid (v7p1 baseline)
SMOKE_GATE=train episodes=150; infer runs=3 (screening only; no final claims)
FINAL_GATE=short runs=20 + long runs=20; pass iff:
  - success_rate(SAC-GlobalCNN) >= success_rate(Hybrid A*-MPC)
  - avg_path_length(SAC-GlobalCNN) < avg_path_length(Hybrid A*-MPC)
  - path_time_s(SAC-GlobalCNN) < path_time_s(Hybrid A*-MPC)
REPORT_ARTIFACTS:
  - table2_kpis_mean_raw.csv (mean KPIs; use for gates/tables)
  - table2_kpis_raw.csv (per-run/per-pair diagnostics; includes failure_reason)
```

## Stages (terminology)

- `self-check`: imports/device sanity check.
- `micro-smoke`: optional ultra-fast loop (e.g. `episodes=40`) for sanity only; not comparable.
- `smoke`: standard screening gate (`episodes=150`, `runs=3`); go/no-go only.
- `full`: final gate (short/long suites with `runs=20` each).

## Repo map (configs + runs)

- Config selection index: `configs/INDEX.md`
- Config usage details: `configs/README.md`
- Run artifact map: `docs/runs/README.md`
- Candidate cleanup list: `docs/runs/CANDIDATES_TO_ARCHIVE_20260221.md`

## Research objective and current status (2026-02-21)

- This repo is under active `vibe coding` iteration: small deltas, quick validation loops, and strict rollback/archive discipline.
- Current stable rollback baseline is `v7p1` (`configs/v7p1.json`).
- SAC migration target version is `v8` (`SAC-GlobalCNN`), currently in planning/implementation stage.
- `v7p2` and `v7p2p1` are archived failed attempts (Markov-observation fix branch), and are not the current mainline claim baseline.
- Final objective: make RL planning (`SAC-GlobalCNN`) outperform classical planning (`Hybrid A*-MPC`) under fair and reproducible evaluation.
- Core optimization targets: shorter paths (`avg_path_length`), shorter path time (`path_time_s`), and smoother trajectories (`avg_curvature_1_m`, lower is better).

## Quickstart (Ubuntu/bash)

All commands below assume you run from the `RL_sac/` folder so outputs go to `runs/` by default:

```bash
cd /home/sun/phdproject/dqn/RL_sac
```

Two ways to run commands:

- Recommended (reproducible/CI-friendly): keep `conda run -n ros2py310 ...`.
- Optional (interactive shell): run `conda activate ros2py310` once, then use `python ...` directly.

Self-check (fast sanity check of imports/device):

```bash
bash scripts/self_check.sh

# (equivalent explicit commands)
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
conda run -n ros2py310 python game.py --self-check
```

Optional extras (for live training window):

```bash
conda run -n ros2py310 python -m pip install -r requirements-optional.txt
```

## Breaking changes (2026-02-06)

- Forest internal cost-to-go chain is removed; environment now uses distance-based shaping/sampling.
- Forest training expert supports `hybrid_astar` and `astar_mpc` (`--forest-expert {auto,hybrid_astar,astar_mpc}`, where `auto -> hybrid_astar`).
- Forest bicycle observation changed from `11 + 2*N*N` to `10 + N*N` (occupancy map only).
- Random-pair flags were renamed from `*-cost-m` to `*-dist-m` (old names removed).
- Infer supports suite ratio thresholds:
  - `--rand-short-min-dist-ratio`, `--rand-short-max-dist-ratio`
  - `--rand-long-min-dist-ratio`, `--rand-long-max-dist-ratio`

Old checkpoints/configs with the previous forest observation/flag schema are not backward compatible.

## Version note (2026-02-21)

- As of 2026-02-21, the stable mainline remains `v7p1`.
- `v7p2/v7p2p1` attempted a Markov-observation fix (adding `prev_a_n`) but did not show stable gains and was archived as a failed branch.
- Mainline was rolled back to `v7p1` behavior (forest bicycle observation `10 + N*N`) for continued iteration.
- Failure archive: `docs/versions/v7p2p1/`.

## Train / infer (recommended: config profiles)

Profiles live under `configs/*.json` and are loaded via `--profile <name>`:

```bash
conda run -n ros2py310 python train.py --profile forest_a_all6_300_cuda
conda run -n ros2py310 python infer.py --profile forest_a_all6_300_cuda
```

Equivalent commands in an activated shell (`conda activate ros2py310`):

```bash
python train.py --profile forest_a_all6_300_cuda
python infer.py --profile forest_a_all6_300_cuda
```

### Latest train/infer commands (keep updated)

Last updated: 2026-02-21  
Current recommended train profile: `v7p1`

```bash
conda run -n ros2py310 python train.py --profile v7p1
conda run -n ros2py310 python infer.py --profile v7p1
```

Note: training now saves process logs to `<run_dir>/train_flow.log` by default; disable via `--no-save-train-log`.

Primary claim regime (docs/paper, current executable baseline): `CNN-DDQN (shielded/hybrid inference)`.
- For v6p2p3/v7p1 and onward mainline reporting, use `shielded/hybrid` naming.
- Do not present these mainline results as `strict-argmax`.
- SAC (`v8`) claim regime will be updated after SAC train/infer pipeline is merged.

### Remote-first execution (`ubuntu-zt`)

Default policy: run train/infer on `ssh ubuntu-zt` first (including smoke/full).  
Fallback to local only when remote is unavailable.

```bash
# 1) Sync local repo -> remote repo (local is source of truth)
rsync -avz --delete \
  --exclude '.git/' \
  --exclude 'runs/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  /home/sun/phdproject/dqn/RL_sac/ \
  ubuntu-zt:/home/sun/phdproject/dqn/RL_sac/

# 2) Run on remote (example: smoke train / micro-smoke, episodes=40; sanity only)
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/RL_sac && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v6p2p3 --episodes 40 --out v6p2p3_smoke --device cuda --progress"

# 3) Sync remote results -> local runs/
rsync -avz \
  ubuntu-zt:/home/sun/phdproject/dqn/RL_sac/runs/v6p2p3_smoke/ \
  /home/sun/phdproject/dqn/RL_sac/runs/v6p2p3_smoke/
```

Micro-smoke (optional quick loop; not the standard smoke gate):

```bash
conda run -n ros2py310 python train.py --profile v6p2p3 --episodes 40 --out v6p2p3_smoke
conda run -n ros2py310 python infer.py --profile v6p2p3 --models v6p2p3_smoke --runs 3 --out v6p2p3_smoke
```

Fixed mid (14-42m) infer commands (strict vs hybrid, runs=20, diagnostic ablation):

```bash
conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_mid_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_mid_pairs20_v1
```

v6p1 timeout-tuned fixed pairs infer (hybrid/shielded, runs=20; checkpoint pinned in profile):

```bash
# NOTE: v6p1 long/mid gating regresses short; keep v6 for short.
conda run -n ros2py310 python infer.py --profile repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1
```

### Train live view (pygame, RL stage only)

Default is off. Enable it explicitly during training:

```bash
conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --live-view --live-view-fps 0 --live-view-window-size 900 --live-view-trail-len 300 --live-view-skip-steps 1
```

- Viewer is attached only to RL training episodes (`env+algo`); demo collect/pretrain are not shown.
- Closing the pygame window does not stop training; it auto-falls back to no-view mode.
- If `pygame` is not installed, training continues and prints an install hint.
- Fixed-size vehicle collision box (oriented by heading from `pose_m`) is ON by default; use `--no-live-view-collision-box` to hide it.

### Interactive goal-click game (pygame)

Left-click a goal on the map, pick a planner, and let `mpc` track the planned path.

```bash
conda run -n ros2py310 python game.py --profile repro_20260212_interactive_game_forest_a_v1
```

Planner hotkeys: `1`=hybrid A*, `2`=RRT*, `3`=grid A*, `4`=cnn-ddqn (requires `--rl-checkpoint <path>`).  
Other: `R` reset, `SPACE` pause, `P` replan.

## `runs/` layout and report artifacts

Output routing:
- If `--out <name>` is a bare name, outputs go to `runs/<name>/`.
- If `--out <path>` is a path, outputs go to `<path>/` as-is.

Typical structure (train + nested infer):

```text
runs/<out>/
  latest.txt
  train_YYYYMMDD_HHMMSS/
    configs/run.json
    train_flow.log
    models/
    infer/
      latest.txt
      YYYYMMDD_HHMMSS/
        configs/run.json
        table2_kpis_mean_raw.csv
        table2_kpis_raw.csv
```

Notes:
- For scripts/machines, prefer `*_raw.csv` files. The non-raw `table2_kpis_mean.csv` uses human-friendly column names.
- Final gates/tables should read `table2_kpis_mean_raw.csv`; failure distribution/diagnostics should read `table2_kpis_raw.csv`.

### KPI dictionary (`table2_kpis_*` columns, minimal)

- `success_rate`: success ratio in `[0,1]` (higher is better).
- `avg_path_length`: average planned path length in meters (lower is better).
- `path_time_s`: trajectory execution time in seconds (lower is better).
- `avg_curvature_1_m`: average curvature in `1/m` (lower is smoother).
- `planning_time_s`: planner time in seconds (lower is better).
- `tracking_time_s`: tracking/controller time in seconds (lower is better).
- `inference_time_s`: policy inference time in seconds (RL only; lower is better).
- `argmax_inadmissible_rate`: fraction of steps where `argmax(Q)` is inadmissible (diagnostic).
- `fallback_rate`: fraction of steps where inference-time fallback/override triggered (diagnostic; should be `0` in `strict-argmax` by definition).
- `failure_reason`: failure type label (only in `table2_kpis_raw.csv`).

## 版本总索引（v1 → v7p2p1）

> 说明：本索引用于统一 `docs/versions/` 的重编号口径；历史目录 `v3p1`~`v3p11` 保留原记录，未纳入本轮重编号；早期误混入版本链已于 2026-02-09 清理。当前稳定主线为 `v7p1`，`v7p2/v7p2p1` 属于失败归档分支。

| 版本 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|
| `v1` | `docs/versions/v1/` | `configs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke.json` | `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017` | `0.0 / 0.0` | `1.0 / 1.0` | 未通过 |
| `v2` | `docs/versions/v2/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246` | `0.0 / 0.0` | `1.0 / 1.0` | 未通过 |
| `v3` | `docs/versions/v3/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403` | `0.5 / 0.1` | `0.9 / 1.0` | 未通过 |

### 增量版本（v3p1 → v7p2p1）

| 版本 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|
| `v3p12` | `docs/versions/v3p12/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622` | `0.0 / 0.0` | `0.95 / 1.0` | 未通过 |
| `v4p1` | `docs/versions/v4p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10/20260210_135524` | `0.1 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p2` | `docs/versions/v4p2/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730` | `0.0 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p3` | `docs/versions/v4p3/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934` | `0.2 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p3p1` | `docs/versions/v4p3p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044` | `0.0 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v5` | `docs/versions/v5/` | `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1.json` | `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351` | `0.75 / 0.85` | `0.95 / 0.90` | 未通过 |
| `v6` | `docs/versions/v6/` | `configs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602` | `0.90 / 0.70` | `0.95 / 0.90` | 未通过 |
| `v6p1` | `docs/versions/v6p1/` | `configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414` | `0.70 / 0.95` | `0.95 / 0.90` | 未通过 |
| `v6p2` | `docs/versions/v6p2/` | `configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414` | `0.70 / 0.95` | `0.95 / 0.90` | 未通过 |
| `v6p2p2` | `docs/versions/v6p2p2/` | `configs/v6p2p2.json` | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20/20260219_123433` | `0.75 / 0.55` | `0.95 / 1.00` | 未通过（待 full） |
| `v6p2p3` | `docs/versions/v6p2p3/` | `configs/v6p2p3.json` | `runs/v6p2p3/train_20260219_142104/infer/20260219_145315` | `0.80 / 1.00` | `1.00 / 1.00` | 已运行（runs=5，待 full20） |
| `v7p1` | `docs/versions/v7p1/` | `configs/v7p1.json` | `runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927` | `1.00 / 1.00` | `1.00 / 1.00` | 稳定主线（runs=5，待 full20） |
| `v7p2` | `docs/versions/v7p2/` | `configs/v7p2.json` | `runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137` | `1.00 / 1.00` | `1.00 / 1.00` | 已运行（micro-smoke：episodes=40, runs=3） |
| `v7p2p1` | `docs/versions/v7p2p1/` | `configs/repro_20260220_v7p2p1_rollback_v7p1.json` | `runs/v7p2_es150/train_20260220_222056/infer/20260220_223016` | `0.85 / 0.65` | `0.95 / 1.00` | 失败归档，已回退到 `v7p1` |

- baseline-only（`--skip-rl`）输出不计入上表；请单独查看 `runs/outputs_forest_baselines/*`、`runs/repro_20260207_*` 等目录。
- 详细四件套请见 `docs/versions/README.md` 与各版本目录。

## Rigour and anti-cheating rules (mandatory)

- RL vs baseline comparison must use the same environment/suite and fixed start-goal pairs (no sample drift).
- Claiming gains requires matched evaluation budget and protocol; final claims must use short/long suites with `runs=20` each.
- Smoke results (`episodes=150`, `runs=3`) are screening-only and cannot be used as final claims.
- Inference policy naming must match implementation (`strict-argmax` vs `shielded/masked/hybrid`), with no hidden intervention.
- No cherry-picking: failed versions/runs must be archived, and missing outputs must be reported as `N/A` with reasons.
- Every claim must be traceable to artifacts: command line, `run_dir`, `run.json`, and `table2_kpis_mean_raw.csv`.

## Recommended iteration workflow (version-first)

For each new version candidate, use this default pipeline:

1. Pre-version snapshot (mandatory):
- Ensure clean workspace: `git status`.
- Snapshot and push before changes:
```bash
git add -A
git commit -m "snapshot: pre-<version>"
git push origin <branch>
```

2. Implement one small version delta only (single-purpose change).

3. Smoke gate (fixed quick loop):
```bash
conda run -n ros2py310 python train.py --profile <candidate> --episodes 150 --out <version>_smoke150 --device cuda
conda run -n ros2py310 python infer.py --profile <candidate> --models <version>_smoke150 --runs 3 --out <version>_smoke150
```

4. Go/No-Go rule:
- If smoke does not show clear gain, stop escalation and mark as failed version.
- Roll back mainline to previous stable version (for example, `v7p1`).

5. Archive immediately:
- Create `docs/versions/<version>/` four-doc bundle and log commands, run paths, KPIs, and failure reasons.
- Prepare next iteration as `<version+1>` (example: `v7p2p2`).

## Final acceptance gate (short/long suites, runs=20)

Final claims must be reported on both short and long suites with `runs=20` per suite.

Use `table2_kpis_mean_raw.csv` and compare `CNN-DDQN` against `Hybrid A*-MPC` with all conditions below:

- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

Also report smoothness in the same table via `avg_curvature_1_m` (lower is smoother). This is a mandatory reporting metric and optimization target for version selection.

If any suite fails any condition above, the final gate is considered failed.

Inference regime naming note: this repo distinguishes `strict-argmax` (legacy label: strict no-fallback) vs `shielded/masked/hybrid`. `strict-argmax` means pure `argmax(Q)` inference (no masking/top-k/stop-override/replacement/heuristic takeover/planner takeover); masks may be computed for logging only. If any inference-time intervention is enabled, label it `shielded/masked/hybrid` (not `strict-argmax`). Current mainline claim/reporting regime is `shielded/hybrid`.

### Strict-argmax vs hybrid re-eval (fixed pairs, diagnostic only)

To compare checkpoints fairly without random-pair drift, evaluate on **fixed random pairs** (short/long suites) and report two regimes. This section is for ablation/diagnostics; it is not the primary claim template for current mainline versions.

- `strict-argmax`: pass `--forest-no-fallback` (pure `argmax(Q)`).
- `hybrid/shielded`: pass `--no-forest-no-fallback` (allows stop-override + replacement only; no heuristic fallback).

Fixed pairs (forest_a, short/long suites, 20 each):

- `configs/repro_20260210_forest_a_pairs_short20_v1.json`
- `configs/repro_20260210_forest_a_pairs_long20_v1.json`

Template (reuse a profile to keep env/action-space settings consistent with the checkpoint):

```bash
PROFILE=repro_20260211_forest_a_cnn_ddqn_v5_smoke
MODELS_DIR="runs/<exp>/<train_timestamp>/models"

# strict-argmax (short)
conda run -n ros2py310 python infer.py --profile "$PROFILE" --baselines \\
  --envs forest_a::short --no-rand-two-suites --random-start-goal --runs 20 \\
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \\
  --models "$MODELS_DIR" --out repro_reval_strict_short_pairs20 \\
  --forest-no-fallback

# hybrid/shielded (short)
conda run -n ros2py310 python infer.py --profile "$PROFILE" --baselines \\
  --envs forest_a::short --no-rand-two-suites --random-start-goal --runs 20 \\
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \\
  --models "$MODELS_DIR" --out repro_reval_hybrid_short_pairs20 \\
  --no-forest-no-fallback
```

For the long suite, replace `forest_a::short` + `pairs_short20` with `forest_a::long` + `pairs_long20`.

## Demonstrations (DQfD)

By default, training uses `--demo-mode dqfd` (strict DQfD-style):

- Prioritized experience replay (PER) + importance sampling (IS) weights
- 1-step TD loss + n-step TD loss + large-margin expert loss + L2 regularization
- No behavior cloning CE loss (to match the DQfD definition)

To reproduce the previous demo stabilizer behavior, use `--demo-mode legacy`.

For A*+MPC expert training with curve optimization (shortcut + resample + minimum-turn-radius + dual-circle collision checks),
use `--forest-expert astar_mpc` together with `--forest-astar-opt-*` and `--forest_mpc_*` flags, or load profile:

```bash
conda run -n ros2py310 python train.py --profile repro_20260208_forest_a_cnn_ddqn_dqfd_astar_mpc_curveopt_300
```

Paper copies + BibTeX are archived under `paper/dqfd_refs/`.

## Baseline eval (no RL checkpoints required)

`--baselines all` now includes six baselines (fixed order):

1. `astar`
2. `hybrid_astar`
3. `rrt_star`
4. `astar_mpc` (`A*-MPC`)
5. `hybrid_astar_mpc` (`Hybrid A*-MPC`)
6. `rrt_mpc` (`RRT-MPC`)

Run all six on CPU:

```bash
conda run -n ros2py310 python infer.py --envs forest_a --out outputs_forest_baselines --baselines all --skip-rl --runs 5 --device cpu
```

Run only MPC-combo baselines:

```bash
conda run -n ros2py310 python infer.py --envs forest_a --out outputs_forest_mpc_baselines --baselines astar_mpc hybrid_astar_mpc rrt_mpc --skip-rl --runs 5 --device cpu
```

### Fixed random pairs (fair baseline comparison)

Use the frozen random-pair profile to compare planners on the exact same `(start, goal)` samples:

```bash
conda run -n ros2py310 python infer.py --profile repro_20260206_6baselines_fair_forest_a_fixedpairs --skip-rl
```

This profile reads pairs from `configs/repro_20260206_6baselines_fair_forest_a_pairs.json`.

Suite-split fixed pairs (short/long, 20 each):

- `configs/repro_20260210_forest_a_pairs_short20_v1.json`
- `configs/repro_20260210_forest_a_pairs_long20_v1.json`

Legacy `forest_baseline_mpc_*` profile keys are ignored during infer config loading (deprecated).

## Success definition

Forest bicycle success is:

- `reached_pose`: within goal tolerance, and (optionally) within a heading-to-goal tolerance
- `reached_stop`: stopped and nearly straight wheels (near-zero `|v|` and `|delta|`)
- `reached` / "success" == `reached_pose AND reached_stop`

Implemented in `forest_vehicle_dqn/env.py` (`AMRBicycleEnv._step_with_controls`, `_goal_pose_reached`, `_goal_stop_reached`).

More runnable examples + flag reference: [`runtxt.md`](runtxt.md).
