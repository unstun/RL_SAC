# v9p2 — 轻量路径+时间惩罚（CNN-DDQN + DQfD）

## 概述

在 v7p1 基础上增加轻量路径长度惩罚（k_len=0.02）和时间惩罚（k_t=0.15），
引导 agent 走更短路径、更快完成任务。

## 核心参数（相对 v7p1 的变更）

| 参数 | v7p1 | v9p2 |
|------|------|------|
| `forest_reward_k_len` | 0.0 | **0.02** |
| `forest_reward_k_t` | 0.0 | **0.15** |

其余参数与 v7p1 完全一致（profile=v7p1, episodes=150, seed=21）。

## 结论

- **Short 路径赢 baseline**（15.50m vs 15.77m, -1.7%）
- Long 场景路径 +6.7%、时间 +21.3%，未达最终门槛
- 参数甜蜜点极窄：k_len 增到 0.025 或 k_t 增到 0.16 即崩溃
- RL 最好单次 long run 路径 42.66m < baseline 42.72m，但方差大

## 状态

**当前最佳 RL 结果**，但未通过最终门槛（short/long 双套件全面超越 baseline）。
