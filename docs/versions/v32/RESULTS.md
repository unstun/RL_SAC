# V32 RESULTS — Training-time Action Shield 消融

## 单模型对比

| 模型 | short SR | short path | long SR | long path | short inad | long inad |
|------|----------|-----------|---------|-----------|-----------|----------|
| V32-A DDQN+shield | **0.333** | 15.78m | **0.333** | 52.20m | **61.9%** | **62.6%** |
| V32-B M-DQN+shield | 0.667 | 17.31m | **1.00** | 47.71m | **15.3%** | **16.1%** |
| V16-C DDQN (no shield) | 0.85 | 14.43m | 1.00 | 46.00m | ~22% | ~30% |

## Ensemble 对比

| 配置 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| V32 ens (V16-C + V32-B) | 1.00 | 15.53m | 1.00 | 46.10m |
| V22-D ens (V16-C + V22-D M-DQN) | 0.95 | 15.75m | 1.00 | 46.11m |
| A*-MPC (formal) | 0.95 | 15.77m | 1.00 | 42.93m |

## 关键发现

1. **DDQN + shield = 灾难性失败**
   - inadmissible rate 从 22% 暴涨到 62%
   - 根因：训练时 agent 只在 admissible 空间探索，从未学到"为什么
     某些动作不好"。Q-landscape 在 inadmissible 动作上是随机的，
     推理时 argmax 频繁选中。

2. **M-DQN + shield = 有效但无增益**
   - inadmissible rate 从 ~30% 降到 16%（显著改善）
   - M-DQN 的 log-policy 正则化使其在受限空间仍能学好 Q 排序
   - 但 ensemble 路径长度无改善（46.10m ≈ 46.11m）

3. **Shield 对两种算法效果相反**
   - DDQN（硬 argmax）：必须见过 inadmissible 动作才能学会避免
   - M-DQN（软策略）：soft policy + entropy 正则化使其在受限空间也能泛化

## 结论

- 负面结果，不进 formal
- V22-D negotiate K=10 仍是全局最优
