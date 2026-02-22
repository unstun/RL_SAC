# v9 结果对比

## 数据来源

- KPI（均值）：`runs/v9-smoke1/train_20260222_111340/infer/20260222_112253/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v9-smoke1/train_20260222_111340/infer/20260222_112253/table2_kpis_raw.csv`
- 运行口径：`runs=3`（smoke）

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 15.9693 | 10.025 | 0.032557 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.267 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 26.0667 | 18.500 | 0.117265 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.333 | 0.058072 |
| long | CNN-DDQN | 0.667 | 50.2558 | 29.000 | 0.142919 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.817 | 0.068718 |

## 对比 v7p1（曲率改善）

| 套件 | v7p1 curvature | v9 curvature | 改善倍数 |
|---|---:|---:|---:|
| short | 0.174 | 0.033 | 5.3x |
| mid | 0.220 | 0.117 | 1.9x |
| long | 0.214 | 0.143 | 1.5x |

## 门槛检查

SR 从 100% 降到 66.7%，未通过 smoke 门槛。惩罚参数过重。

## `failure_reason` 分布

- CNN-DDQN：short=`reached=2, timeout=1`，mid=`reached=2, collision=1`，long=`reached=2, timeout=1`
- Hybrid A*-MPC：short/mid/long 均 `reached=3`
