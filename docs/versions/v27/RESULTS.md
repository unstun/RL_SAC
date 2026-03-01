# V27 RESULTS — Geodesic Goal Distance Field 观测通道

## 单模型对比（smoke runs=3）

| 模型 | short SR | short path | long SR | long path | short inad | long inad |
|------|----------|-----------|---------|-----------|-----------|----------|
| V27-A DDQN+geodist | **1.00** | 15.15m | 0.667 | 49.89m | 31.8% | 18.8% |
| V27-B M-DQN+geodist | 0.333 | 11.80m | 0.667 | 65.48m | 48.7% | 43.3% |
| V16-C DDQN (baseline) | 0.85 | 14.43m | **1.00** | 46.00m | ~22% | ~30% |
| V22-D M-DQN (baseline) | 1.00 | 16.48m | 0.85 | 51.03m | — | — |

## Ensemble 不可行

obs_dim 不兼容：V27 模型 obs_dim=298（10+2×144），V16-C obs_dim=154（10+1×144）。
无法将 V16-C 与 V27 模型放入同一 ensemble，因为共享同一 env。

## 关键发现

1. **DDQN + geodist = 中性偏负**
   - long inad 从 ~30% 降到 18.8%（改善），但 long SR 只有 0.667
   - short SR 提升到 1.00，但 path 更长（15.15m vs 14.43m）

2. **M-DQN + geodist = 灾难**
   - inad 暴涨到 43-49%（比 baseline 更差）
   - M-DQN 的 log-policy 正则化与 obs_dim 翻倍交互不良
   - 260ep 仍无法收敛到合理水平

3. **obs_dim 翻倍 = 学习难度翻倍**
   - 从 154→298 维，CNN 第一层 Conv2d(1→32) 变为 Conv2d(2→32)
   - 300ep 训练预算可能不足以让模型学会利用 geodesic 通道

## 结论

**负面结果，不进 formal。**
- 单模型均劣于 baseline，ensemble 因 obs_dim 不兼容而不可行
- V22-D negotiate K=10 仍是全局最优
