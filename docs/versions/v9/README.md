# v9: 回归 CNN-DDQN + 曲率/平滑性惩罚强化

- 版本类型：**Major（v+1）**
- 上一版本：`v7p1`（CNN-DDQN 稳定基线，SR=100%）
- 本版口径：`shielded/hybrid`（训练与推理一致）
- 状态：**smoke 完成，SR=66.7%，未通过 smoke 门槛**

## 版本背景

v8 系列（v8→v8p4）尝试将算法从 CNN-DDQN 迁移到 SAC-Discrete，五个版本全部 SR=0%。
结论：离散动作空间下 SAC 的熵正则化机制失效，结构性不匹配无法通过调参弥补。
V9 回归 CNN-DDQN，保留 v8 作为负面证据，聚焦路径质量优化。

## 本版目标

在保持 SR=100% 的前提下，通过增大曲率/转向惩罚降低路径曲率，缩短路径长度和耗时。
核心差距（v7p1 vs Hybrid A*-MPC）：
- long 套件路径长 21%（51.91 vs 43.02）、耗时多 49%（34.01 vs 22.82）
- 曲率 2-3x（根因：路径太弯）

## 方法摘要

基于 v7p1 配置，仅调整三个奖励参数：
- `forest_reward_k_kappa`（曲率惩罚）：`0.2` → `0.5`（2.5x）
- `forest_reward_k_delta`（转向变化惩罚）：`0.8` → `1.5`（~2x）
- `forest_reward_k_len`（路径长度惩罚）：`0.0` → `0.05`（新增）

其余参数（网络结构、DQfD、课程、gating 等）保持 v7p1 不变。

## 关键命令

```bash
# 训练
conda run -n ros2py310 python train.py --profile v9 --episodes 150 --out v9-smoke1 --device cuda --progress --save-ckpt best
# 推理
conda run -n ros2py310 python infer.py --profile v9 --models v9-smoke1 --out v9-smoke1 --runs 3 --progress
```

## 代表 Run

- 训练：`runs/v9-smoke1/train_20260222_111340`
- 推理：`runs/v9-smoke1/train_20260222_111340/infer/20260222_112253`

## 结论

**Smoke SR=66.7%（short/mid/long 各 2/3 成功），未通过 smoke 门槛。**

曲率大幅改善（short: 0.174→0.033，5.3x；long: 0.214→0.143，1.5x），方向正确。
但惩罚过重导致 SR 下降（2 个 timeout + 1 个 collision）。

## 后续版本

- **v9p1**（A* corridor 惩罚）：long SR=0%，失败
- **v9p2**（k_len=0.02, k_t=0.15）：**当前最佳**，full runs=20 short path 赢 baseline，long 仍有差距
  - 详见 `docs/versions/v9p2/`
