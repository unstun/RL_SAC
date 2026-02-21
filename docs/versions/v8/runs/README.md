# v8 Runs 索引

## 代表 run

| 类型 | run_dir | run_json | kpi |
|------|---------|----------|-----|
| smoke 训练 | `runs/v8-smoke/train_20260221_160435` | `runs/v8-smoke/train_20260221_160435/train_flow.log` | `runs/v8-smoke/train_20260221_160435/training_returns.csv` |
| smoke 推理 | `runs/v8-smoke/train_20260221_160435/infer/20260221_161349` | `runs/v8-smoke/train_20260221_160435/infer/20260221_161349/configs/run.json` | `runs/v8-smoke/train_20260221_160435/infer/20260221_161349/table2_kpis_mean_raw.csv` |

## 可追溯 run 列表

1. **smoke-train-20260221_160435**: 150 episodes, ubuntu-zt (RTX 5070 Ti), ~8m29s
2. **smoke-infer-20260221_161349**: runs=3, short/mid/long, success_rate=0% 全套件
