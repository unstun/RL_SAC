# configs/ 使用说明（profiles + 固定 pairs）

本目录包含两类“可复现输入”：

1) **profile 配置**：`configs/*.json`
   - 通过 `--profile <name>` 加载（`<name>` 为不带 `.json` 的文件名）。
   - `train.py` 读取 `train` 段；`infer.py` 读取 `infer` 段；`game.py` 读取 `game` 段（交互式 demo）。

2) **固定随机样本 pairs**：`configs/*pairs*.json`
   - 不是 profile；通过 `infer.py --rand-pairs-json <path>` 加载。
   - 用于随机起终点评测的“公平对比”（避免每次采样漂移）。

> 推荐命令风格：`conda run -n ros2py310 ...`（Ubuntu 24.04）。

## 快速导航（先看这里）

- 主线 profile（当前稳定回退基线）：`configs/v7p1.json`
- SAC 迁移目标：`v8`（配置待合入，先参考 `docs/plans/2026-02-20-sac-globalcnn-plan.md`）
- 配置总索引（推荐入口）：`configs/INDEX.md`
- 版本留档索引：`docs/versions/README.md`
- 运行结果索引：`docs/runs/README.md`

---

## 1) 用 profile 跑训练 / 推理

```bash
conda run -n ros2py310 python train.py --profile v7p1
conda run -n ros2py310 python infer.py  --profile v7p1
```

---

## 1.5) 用 profile 跑交互式 game（pygame）

`game.py` 会从 profile 的 `game` 段读取参数（如地图、planner、MPC 参数、窗口大小等）：

```bash
conda run -n ros2py310 python game.py --profile repro_20260212_interactive_game_forest_a_v1
```

> `--self-check` 不依赖 pygame 渲染窗口，用于快速检查规划+MPC 链路是否能跑通：
>
> `conda run -n ros2py310 python game.py --profile repro_20260212_interactive_game_forest_a_v1 --self-check`

---

## 2) 固定 pairs：baseline-only 公平评测（示例）

已有示例（固定 30 个 random pairs，forest_a）：

```bash
conda run -n ros2py310 python infer.py --profile repro_20260206_6baselines_fair_forest_a_fixedpairs --skip-rl
```

该 profile 会引用：

- `configs/repro_20260206_6baselines_fair_forest_a_pairs.json`

---

## 3) 固定 pairs：RL strict vs hybrid 双口径复评测（模板）

本仓库区分两种推理口径（命名必须与实现一致）：

- `strict-argmax`：推理仅 `argmax(Q)`（使用 `--forest-no-fallback`）
- `hybrid/shielded`：允许推理期干预（使用 `--no-forest-no-fallback`，可能触发 top-k/mask replacement/stop override；不启用启发式 fallback）

固定 pairs（short/long 各 20）文件：

- `configs/repro_20260210_forest_a_pairs_short20_v1.json`
- `configs/repro_20260210_forest_a_pairs_long20_v1.json`

以某个训练出的 `models/` 目录为例（替换成你的实际路径）：

```bash
MODELS_DIR="runs/<exp>/<train_timestamp>/models"

# strict-argmax (short)
conda run -n ros2py310 python infer.py \\
  --envs forest_a::short --random-start-goal --runs 20 \\
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \\
  --models "$MODELS_DIR" --rl-algos cnn-ddqn \\
  --forest-no-fallback

# hybrid/shielded (short)
conda run -n ros2py310 python infer.py \\
  --envs forest_a::short --random-start-goal --runs 20 \\
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \\
  --models "$MODELS_DIR" --rl-algos cnn-ddqn \\
  --no-forest-no-fallback
```

> long 套件同理：把 `forest_a::short` 与 `pairs_short20` 改为 `forest_a::long` 与 `pairs_long20`。
