# v8p2 Runs

## 代表 Run

| 类型 | run_dir | run_json | kpi |
| --- | --- | --- | --- |
| 训练 | `runs/v8p2-smoke1/train_20260221_224028` | `runs/v8p2-smoke1/train_20260221_224028/configs/run.json` | `runs/v8p2-smoke1/train_20260221_224028/training_stats.csv` |
| 推理 | `runs/v8p2-smoke1/train_20260221_224028/infer/20260221_224714` | `runs/v8p2-smoke1/train_20260221_224028/infer/20260221_224714/configs/run.json` | `runs/v8p2-smoke1/train_20260221_224028/infer/20260221_224714/table2_kpis_mean_raw.csv` |

## 可追溯 Run 列表

1. **v8p2-smoke1**（2026-02-21）
   - 训练：150ep, seed=21, CBF alpha=0.3, safety_margin=0.15
   - 推理：runs=3, short/mid/long
   - 结果：SR=0%（全套件），best_return=477.5
   - 命令：
     ```bash
     # 训练
     conda run -n ros2py310 python train.py --profile v8p2 --episodes 150 --out v8p2-smoke1 --no-progress
     # 推理
     conda run -n ros2py310 python infer.py --profile v8p2 --runs 3 --models runs/v8p2-smoke1/train_20260221_224028 --out v8p2-smoke1 --no-progress
     ```
