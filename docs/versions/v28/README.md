# V28 — Geodesic Progress Reward

## 版本摘要

在进度奖励中将欧几里得距离替换为测地距离（Dijkstra绕障最短路），引导 agent 走真实可行的最短路径。

## 方法

**核心思想**：当前进度奖励 `k_p * (d_euclidean_before - d_euclidean_after)` 遇到障碍物时会产生误导——agent 绕墙走时直线距离不变甚至变远，得到负奖励，反而被惩罚正确的绕路行为。

改为测地距离 `k_p * (d_geodesic_before - d_geodesic_after)`，沿无障碍空间的最短路距离会随正确绕路单调递减，消除误导梯度。

**实现**：每个 episode 开始时（goal 确定后）运行一次 Dijkstra，缓存全局测地距离场；per-step 只做格子查找，计算开销极小。

## 代码改动

- `env.py`：新增 `reward_geodesic_progress` 参数、`_geo_reward_field` 缓存、`_geo_reward_dist()` 辅助方法
- `cli/train.py`：新增 `--reward-geodesic-progress` flag

## 主要结果（smoke runs=3）

| 配置 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| A baseline 150ep | 1.00 | 22.41m | 0.667 | 57.75m |
| B geodesic 150ep | 1.00 | 19.03m | 0.667 | **45.41m** |
| B2 geodesic 300ep | 0.667 | **15.89m** | **1.00** | 47.93m |
| A*-MPC (formal) | 0.95 | 15.77m | 1.00 | 42.93m |

## 结论

Geodesic reward 有效，long path 从 57m 降到 45-48m 范围。单模型尚未超越 V22-D ensemble，long path gap ~5m。

## 下一步

1. V28-B2 + V16-C negotiate ensemble 推理（可能接近 V22-D 甚至更优）
2. 扫描 periodic checkpoints 找最优 epoch
3. formal runs=20 验证

## 相关 runs

- `runs/v28-A-baseline/` — baseline 对照
- `runs/v28-B-geodesic/` — geodesic 150ep
- `runs/v28-B-geo-ep300/` — geodesic 300ep（主要结果）
- `runs/v28-A-smoke/`, `runs/v28-B-smoke/`, `runs/v28-B-ep300-smoke/` — smoke 评测
