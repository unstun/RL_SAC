# v9 运行记录

## 代表 Run

- 训练：`runs/v9-smoke1/train_20260222_111340`
- 推理：`runs/v9-smoke1/train_20260222_111340/infer/20260222_112253`
- KPI（均值）：`runs/v9-smoke1/train_20260222_111340/infer/20260222_112253/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v9-smoke1/train_20260222_111340/infer/20260222_112253/table2_kpis_raw.csv`

## 可追溯 Run 列表

| 类型 | run_dir | 状态 |
|------|---------|------|
| smoke 训练 | `runs/v9-smoke1/train_20260222_111340` | 完成（150ep，best ckpt at ep67） |
| smoke 推理 | `runs/v9-smoke1/train_20260222_111340/infer/20260222_112253` | 完成（runs=3，SR=66.7%） |
