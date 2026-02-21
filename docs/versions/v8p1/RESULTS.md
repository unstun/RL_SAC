# v8p1 结果

## 数据来源

- KPI: `runs/v8/20260221_213357/table2_kpis_mean_raw.csv`
- 训练 CSV: `runs/v8-fix1/train_20260221_212443/training_returns.csv`

## Smoke 结果（150ep 训练 + runs=3 推理）

### 推理 KPI

| Suite | CNN-SAC SR | Hybrid A*-MPC SR |
|-------|-----------|-----------------|
| short | 0% | 100% |
| mid | 0% | 100% |
| long | 0% | 100% |

### 训练稳定性对比

| 指标 | v8 mid3 (修复前) | v8p1 fix1 (修复后) |
|------|-----------------|-------------------|
| 训练 return 趋势 | 发散 (-2K → -29K) | 稳定 (-900 → -400) |
| 最差 episode return | -211,244 | -5,330 |
| Q 值 (q1_mean) | 爆炸（无 log） | 稳定 (11 → 0.5) |
| Alpha | 未知 | 1.0 → 0.001 |
| Best return | -232.2 | -252.6 |
| 训练时长 | 1h32m (1000ep) | 7m28s (150ep) |

### 周期评估 (eval_return)

| Episode | eval_return |
|---------|-------------|
| 50 | -306.6 |
| 100 | -517.7 |
| 150 | -394.3 |

## 门槛检查

- SR(CNN-SAC) = 0% < SR(Hybrid A*-MPC) = 100% → **未通过**
- 原因：150ep smoke 不足以让 SAC 学会导航，但训练稳定性已修复

## failure_reason 分布

150ep smoke 未产出 failure_reason 字段（推理 runs=3 全部失败，均为 stuck/collision 终端）。
