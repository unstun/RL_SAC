# v8p3 Runs

## 代表 Run

| 类型 | 路径 |
|------|------|
| 训练 run_dir | `runs/v8p3-smoke1/train_20260222_011340/` |
| 训练 flow_log | `runs/v8p3-smoke1/train_20260222_011340/train_flow.log` |
| 训练 CSV | `runs/v8p3-smoke1/train_20260222_011340/training_stats.csv` |
| 训练 config | `runs/v8p3-smoke1/train_20260222_011340/configs/run.json` |
| 推理 run_dir | `runs/v8p3-smoke1/train_20260222_011340/infer/20260222_012756/` |
| 推理 KPI | `runs/v8p3-smoke1/train_20260222_011340/infer/20260222_012756/table2_kpis_mean_raw.csv` |
| 推理 KPI (md) | `runs/v8p3-smoke1/train_20260222_011340/infer/20260222_012756/table2_kpis_mean.md` |

## 可追溯 Run 列表

| # | 日期 | 类型 | episodes/runs | SR(short) | SR(long) | 备注 |
|---|------|------|---------------|-----------|----------|------|
| 1 | 2026-02-22 | smoke 训练 | 150ep | - | - | alpha 崩塌到 0.003 |
| 2 | 2026-02-22 | smoke 推理 | 3 runs | 0% | 0% | 未通过 smoke 门槛 |
