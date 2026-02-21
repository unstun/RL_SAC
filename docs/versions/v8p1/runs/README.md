# v8p1 Runs 索引

## 代表 run

| 类型 | run_dir | run_json | kpi |
|------|---------|----------|-----|
| smoke 训练 | `runs/v8-fix1/train_20260221_212443` | `runs/v8-fix1/train_20260221_212443/configs/run.json` | `runs/v8-fix1/train_20260221_212443/training_returns.csv` |
| smoke 推理 | `runs/v8/20260221_213357` | `runs/v8/20260221_213357/configs/run.json` | `runs/v8/20260221_213357/table2_kpis_mean_raw.csv` |

## 可追溯 run 列表

1. **fix1-train-20260221_212443**: 150ep, ubuntu-zt (RTX 5070 Ti), 7m28s, best_return=-252.6, 训练稳定
2. **fix1-infer-20260221_213357**: runs=3, short/mid/long, SR=0% 全套件, Hybrid A*-MPC SR=100%
