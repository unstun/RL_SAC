# v8p2 RESULTS

## 数据来源

- KPI CSV：`runs/v8p2-smoke1/train_20260221_224028/infer/20260221_224714/table2_kpis_mean_raw.csv`
- 训练 CSV：`runs/v8p2-smoke1/train_20260221_224028/training_stats.csv`

## Smoke 推理结果（150ep 训练 + runs=3）

| Suite | Algorithm | SR | avg_path_length | path_time_s |
| --- | --- | --- | --- | --- |
| short | CNN-SAC | 0% | N/A | N/A |
| short | Hybrid A*-MPC | 100% | 17.03m | 10.27s |
| mid | CNN-SAC | 0% | N/A | N/A |
| mid | Hybrid A*-MPC | 100% | 24.08m | 13.33s |
| long | CNN-SAC | 0% | N/A | N/A |
| long | Hybrid A*-MPC | 100% | 43.01m | 22.82s |

## 门槛检查

- 最低门槛（SR > 0%）：**未通过**
- 进入 v8p3 门槛（short SR >= 20%）：**未通过**
- 最终门槛（runs=20）：未执行（smoke 阶段）

## 训练统计

| 指标 | 值 |
| --- | --- |
| Episodes | 150 |
| Best return | 477.5 |
| Last 10 mean return | -324.0 |
| Alpha (final) | 0.047 |
| CBF interventions (total) | 913 |
| Eval returns | -371.6, -105.6, -245.1 |
| 训练耗时 | 5m30s |

## failure_reason 分布

Smoke runs=3 中 CNN-SAC 全部失败（SR=0%），推测主要原因：
- 150ep 训练不足以让 SAC 策略收敛
- 训练中 best_return=477.5 说明策略有学习信号，但泛化不足

## 结论

v8p2 作为 smoke 阶段归档。CBF filter 和奖励塑形机制已验证可用，但 150ep 训练量不足以突破 SR=0%。需要更长训练或课程学习。
