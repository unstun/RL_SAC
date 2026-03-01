# v12 Runs 目录

## 代表性运行

- 训练 run: `runs/v11-abl-duel-mha/`（复用 V11 消融训练，Dueling+MHA, 150ep）
- 推理 run: `runs/v11-formal-duel-mha/20260224_160845/`
- KPI: `runs/v11-formal-duel-mha/20260224_160845/table2_kpis_mean.csv`

## 本版本已知运行

### 正式评测 (runs=20, short/long)

| 用途 | Run 目录 | 说明 |
|------|----------|------|
| 训练 | `runs/v11-abl-duel-mha/` | Dueling+MHA, 150ep, profile=v9p2 |
| 推理 | `runs/v11-formal-duel-mha/20260224_160845/` | runs=20, short+long, horizon=30 |

### Horizon 参数实验（附加）

| Horizon | 训练 Run | 推理 Run | SR |
|---------|----------|----------|----|
| 30 | `v11-abl-duel-mha` | `v11-formal-duel-mha/20260224_160845` | 0.95 |
| 15 | `v12-horizon15` (远端) | `v12-infer-horizon15/20260224_164453` | 0.0 |
| 10 | `v12-horizon10` (远端) | `v12-infer-horizon10/20260224_164455` | 0.2 |
