# V21 RESULTS — M-DQN H×k_len 消融 + Negotiate Ensemble

## 训练消融矩阵

| 实验 | H | k_len | best_sr_all | best epoch | 结论 |
|------|---|-------|-------------|-----------|------|
| A | 60 | 0.20 | ~0.7 | - | 弱 |
| **B R1** | **60** | **0.30** | 0.7 | ep110 | 弱 |
| **B R2** | **60** | **0.30** | **1.000** | **ep150** | **最佳** |
| C | 80 | 0.20 | 0.800 | ep200 | H=80 不稳定 |
| D old | 80 | 0.30 | 0.700 | - | H=80 失败 |
| D new | 80 | 0.30 | 0.700 | ep80/100 | H=80 失败 |

B R2 是意外的第二次训练（与 C 同时启动导致 GPU 争抢），但结果最好。

## Smoke 评测 (runs=5, H=60)

| 模型 | ckpt | short SR | short path | long SR | long path |
|------|------|----------|-----------|---------|-----------|
| B R2 | ep150 | **1.0** | 16.66m | **1.0** | 50.50m |
| B R2 | ep120 | 0.6 | 15.57m | 0.2 | 43.64m |
| B R1 | ep110 | 0.6 | 17.21m | 0.8 | 49.62m |

ep150 远优于其他 checkpoint，ep120 严重退化。

## Single M-DQN Formal (runs=20, B R2 ep150, H=60)

| 模型 | short SR | short path | long SR | long path | compute |
|------|----------|-----------|---------|-----------|---------|
| **CNN-MDQN** | **1.00** | 16.48m | 0.85 | 51.03m | 0.68s/3.65s |
| Hybrid A*-MPC | 0.95 | 15.77m | 1.00 | 42.93m | 0.90s/3.81s |

**首次单模型 short SR 超越 A*-MPC (1.00 > 0.95)**!
M-DQN 计算时间也快于 A*-MPC (short: 0.68s vs 0.90s)。

## Negotiate Ensemble Smoke (K=20, H=60, runs=5)

| 模型 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| Negotiate (DDQN+M-DQN) | **1.0** | 16.64m | **1.0** | 48.38m |
| A*-MPC | 1.0 | 16.87m | 1.0 | 43.02m |

双 SR=1.0，short path 短于 A*-MPC (16.64m < 16.87m)。

## Negotiate Ensemble Formal (K=20, H=60, runs=20)

| 模型 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| CNN-DDQN (negotiate) | 0.95 | 16.08m | 0.90 | 46.64m |
| Hybrid A*-MPC | 0.95 | 15.77m | 1.00 | 42.93m |

Ensemble 提升 long SR (0.85→0.90) 但降低 short SR (1.00→0.95)。

## 全配置对比（Formal, runs=20）

| 模型 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| Single M-DQN V21 ep150 | **1.00** | 16.48m | 0.85 | 51.03m |
| Negotiate V21 (K=20) | 0.95 | **16.08m** | **0.90** | **46.64m** |
| V20 Negotiate (K=20) | 0.95 | ~14.55m | 0.95 | ~46.09m |
| A*-MPC | 0.95 | **15.77m** | **1.00** | **42.93m** |

## §13 门槛检查（单 M-DQN — short SR 最优）

| 条件 | 要求 | M-DQN | 通过? |
|------|------|-------|-------|
| SR(short) >= A* | >=0.95 | **1.00** | **是** |
| SR(long) >= A* | >=1.00 | 0.85 | **否** |
| path(short) < A* | <15.77m | 16.48m | **否** |
| path(long) < A* | <42.93m | 51.03m | **否** |

## §13 门槛检查（Negotiate Ensemble — 最均衡）

| 条件 | 要求 | V21 Negotiate | 通过? |
|------|------|--------------|-------|
| SR(short) >= A* | >=0.95 | 0.95 | **是(平)** |
| SR(long) >= A* | >=1.00 | 0.90 | **否** |
| path(short) < A* | <15.77m | 16.08m | **否** |
| path(long) < A* | <42.93m | 46.64m | **否** |

## 最终结论

1. **H=60 + k_len=0.30 是 M-DQN 最优超参**: 首次训练达到 sr_all=1.0
2. **单 M-DQN short SR=1.00 首次超越 A*-MPC**: 里程碑式进展
3. **但路径仍然长于 A*-MPC**: short +4.5%, long +18.9%
4. **V21 vs V20 negotiate**: SR 持平/略降，但 path 更长（16.08m vs 14.55m）
5. **H=80 全面失败**: 视距过大导致决策空间爆炸
6. **训练方差仍是核心瓶颈**: B R1 和 B R2 同配置结果差距巨大
7. §13 仍未通过，long SR 和 path 是剩余 blockers
