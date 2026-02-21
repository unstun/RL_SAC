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

## 1000ep 中等规模测试（v8-mid3, 含稳定性修复）

数据来源: `runs/v8-mid3/train_20260221_190531/infer/20260221_203908/table2_kpis_mean_raw.csv`

修复内容: reward_scale=0.01, grad_clip_norm=1.0, BC pretrain 5000 steps, buffer 100K, 跳过 DQN demo

| Suite | Algorithm | success_rate | avg_path_length | path_time_s |
|-------|-----------|-------------|-----------------|-------------|
| short | CNN-SAC | 0.0 | N/A | N/A |
| mid | CNN-SAC | 0.0 | N/A | N/A |
| long | CNN-SAC | 0.0 | N/A | N/A |

### 训练回报趋势（v8-mid3）

- 初期（ep 1-5）: -336 ~ -572
- 中期（ep 500）: -244 ~ -67256（双峰分布）
- 末期（ep 991-1000）: -385 ~ -92882（严重发散）
- 训练时间: 1h32m（ubuntu-zt RTX 5070 Ti）

### 根因分析

SAC 在连续动作森林导航任务上训练发散，即使加了 reward scaling + grad clipping + BC 预训练。核心问题：
1. **奖励稀疏**: 几乎无法到达目标，只有负奖励信号
2. **连续空间探索困难**: 随机探索难以产生有效轨迹
3. **无安全过滤**: 离散 DQN 有 admissible-action masking，SAC 没有
4. **双峰回报**: 部分 episode 正常（-300~-500），部分灾难性（-50000+），说明 agent 容易陷入死循环
