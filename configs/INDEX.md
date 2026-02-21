# `configs/` 索引（2026-02-21）

> 目标：把“该用哪个 profile（配置名）”先讲清楚，避免从 100+ 文件里猜。

## 1) 主线/优先使用

| 类别 | 文件 | 用途 | 状态 |
|---|---|---|---|
| 稳定回退基线 | `configs/v7p1.json` | 当前训练/推理默认可执行基线 | 推荐 |
| SAC 迁移目标 | `docs/plans/2026-02-20-sac-globalcnn-plan.md` | `v8` 设计与实施计划入口（配置待合入） | 进行中 |
| 失败分支归档 | `configs/v7p2.json` | `v7p2` 试验分支留档 | 仅复盘 |
| 回退对照 | `configs/v6p2p3.json` | 主线前一稳定代口径对照 | 仅对照 |

## 2) 版本链配置（`v*.json`）

这些文件是版本号入口（`vxpx` 命名），建议优先用于版本复现实验：

- `configs/v6p2.json`
- `configs/v6p2p1.json`
- `configs/v6p2p2.json`
- `configs/v6p2p3.json`
- `configs/v7p1.json`
- `configs/v7p2.json`

> 规则：新增版本时，优先新增 `v*.json`，再在四件套中记录它与 `run_dir`（运行目录）的映射。

## 3) 历史别名配置（`forest_*.json`）

`forest_*`（早期语义化别名）保留用于兼容旧命令，不建议作为新版本主入口：

- `configs/forest_a_all6_300_cuda.json`
- `configs/forest_a_all6_300_cuda_latest.json`
- `configs/forest_a_all6_300_cuda_latest_final.json`
- `configs/forest_a_all6_300_cuda_latest_final_no_stuck.json`
- `configs/forest_a_best.json`
- `configs/forest_a_best_600_cuda.json`
- `configs/forest_a_best_latest.json`
- `configs/forest_a_best_latest_cuda.json`
- `configs/forest_a_best_two_suites.json`
- `configs/forest_a_cnn_ddqn_300_cuda_latest_final_no_stuck.json`
- `configs/forest_a_cnn_ddqn_300_cuda_latest_final_no_stuck_long_baselines_100pct.json`
- `configs/forest_a_cnn_ddqn_300_cuda_latest_final_no_stuck_long_baselines_tuned.json`

## 4) 实验快照配置（`repro_*.json`）

- 数量较多（当前 90+），用于“一次实验一份参数快照”的可追溯记录。
- 建议做法：新实验继续写 `repro_YYYYMMDD_<topic>.json`，但最终沉淀到版本链时再抽取 `v*.json` 作为主入口。

## 5) pairs 文件（固定样本集）

`pairs`（固定随机起终点集合）用于公平对比，不是 profile：

- 示例：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
- 示例：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
- 调用方式：`infer.py --rand-pairs-json <path>`

## 6) 建议命名与落库规则（从本轮开始）

1. 主线版本：`v*.json`（长期入口）。
2. 过程实验：`repro_YYYYMMDD_<topic>.json`（短期快照）。
3. 禁止在新流程中继续扩增 `forest_*_latest*` 类别。
4. 每个新 `v*.json` 都必须在 `docs/versions/<version>/` 四件套登记：
   - 配置路径
   - 代表命令
   - 代表 `run_dir`
   - `table2_kpis_mean_raw.csv` 与 `table2_kpis_raw.csv` 路径

## 7) 最小使用模板

```bash
conda run -n ros2py310 python train.py --profile v7p1
conda run -n ros2py310 python infer.py --profile v7p1
```
