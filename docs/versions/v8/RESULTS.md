# v8 结果

## Smoke 测试（150 episodes, runs=3）

数据来源: `runs/v8-smoke/train_20260221_160435/infer/20260221_161349/table2_kpis_mean_raw.csv`

| Suite | Algorithm | success_rate | avg_path_length | path_time_s |
|-------|-----------|-------------|-----------------|-------------|
| short | CNN-SAC | 0.0 | N/A | N/A |
| short | Hybrid A*-MPC | 1.0 | 17.03 | 10.27 |
| mid | CNN-SAC | 0.0 | N/A | N/A |
| mid | Hybrid A*-MPC | 1.0 | 24.08 | 13.33 |
| long | CNN-SAC | 0.0 | N/A | N/A |
| long | Hybrid A*-MPC | 1.0 | 43.01 | 22.82 |

## 门槛检查

- `success_rate(CNN-SAC) >= success_rate(Hybrid A*-MPC)`: **未通过**（0.0 vs 1.0）
- `avg_path_length(CNN-SAC) < avg_path_length(Hybrid A*-MPC)`: **N/A**（无成功轨迹）
- `path_time_s(CNN-SAC) < path_time_s(Hybrid A*-MPC)`: **N/A**（无成功轨迹）

## 训练回报趋势

- 初期（ep 1-10）: -3527 ~ -294
- 末期（ep 140-150）: -441 ~ -226
- 趋势：有改善信号，但 150 episodes 远不够收敛

## failure_reason 分布

Smoke runs=3 全部失败，failure_reason 均为超时/未到达目标（success_rate = 0%）。

## 结论

管线端到端验证通过。SAC 在 150 episodes 下未收敛，需要全量训练（≥1000 episodes）才能评估实际性能。
