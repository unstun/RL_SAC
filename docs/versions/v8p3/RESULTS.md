# v8p3 RESULTS

## Smoke 结果（150ep 训练 + runs=3 推理）

### 推理 KPI（table2_kpis_mean_raw.csv）

| Suite | Algorithm | SR | avg_path_length | path_time_s |
|-------|-----------|---:|----------------:|------------:|
| short | CNN-SAC | 0% | N/A | N/A |
| short | Hybrid A*-MPC | 100% | 17.03 m | 10.27 s |
| mid | CNN-SAC | 0% | N/A | N/A |
| mid | Hybrid A*-MPC | 100% | 24.08 m | 13.33 s |
| long | CNN-SAC | 0% | N/A | N/A |
| long | Hybrid A*-MPC | 100% | 43.01 m | 22.82 s |

来源：`runs/v8p3-smoke1/train_20260222_011340/infer/20260222_012756/table2_kpis_mean_raw.csv`

### 训练信号

| 指标 | 值 |
|------|-----|
| best_return（raw） | 482,579,082.9（ep1，指数 potential 膨胀） |
| alpha（最终） | 0.003（从 1.008 单调下降） |
| eval_return（ep50） | -377.2 |
| eval_return（ep100） | -239.8 |
| eval_return（ep150） | -3279.6 |
| CBF 介入次数 | 3521 |
| 训练耗时 | 13m40s |

### 门槛检查

- SR(CNN-SAC) >= SR(Hybrid A*-MPC)：**未通过**（0% < 100%）
- **结论：未通过 smoke 门槛，不进入 full 评测**

### failure_reason 分布

所有 CNN-SAC 推理 run 均为 timeout/collision（SR=0%），无成功到达目标的 episode。

## 已知问题

1. `reward_potential_base=32` 导致 ep1 raw return 膨胀到 4.8 亿
2. TECRL 未能阻止 alpha 崩塌（0.003 @ 150ep）
3. 日志中 `tq=0.00` 是 key 不匹配（`target_q_r_mean` vs `target_q_mean`）
