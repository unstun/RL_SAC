# v7p1 结果对比

## 数据来源
- KPI（均值）：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_raw.csv`
- 运行口径：`runs=5`（来自 `runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/configs/run.json`）

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 1.00 | 16.7849 | 11.60 | 0.174422 |
| short | Hybrid A*-MPC | 1.00 | 16.8724 | 10.00 | 0.077256 |
| mid | CNN-DDQN | 1.00 | 26.6106 | 16.60 | 0.219962 |
| mid | Hybrid A*-MPC | 1.00 | 25.1525 | 13.87 | 0.079926 |
| long | CNN-DDQN | 1.00 | 51.9081 | 34.01 | 0.214457 |
| long | Hybrid A*-MPC | 1.00 | 43.0247 | 22.82 | 0.067176 |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 通过 | 通过 | 不通过 | 未通过 |
| mid | 通过 | 不通过 | 不通过 | 未通过 |
| long | 通过 | 不通过 | 不通过 | 未通过 |

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本次仅 runs=5）
- long（runs=20）：`N/A`（本次仅 runs=5）
- 结论：当前证据不可用于最终门槛结论。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：short=`reached=5`，mid=`reached=5`，long=`reached=5`（总计 `reached=15`）
- Hybrid A*-MPC：short=`reached=5`，mid=`reached=5`，long=`reached=5`（总计 `reached=15`）
