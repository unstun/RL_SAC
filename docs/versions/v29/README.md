# V29 — Geodesic Reward M-DQN Ensemble

## 版本摘要

在 V22-D negotiate ensemble 基础上，将 M-DQN approver 替换为
用测地进度奖励（geodesic progress reward）训练的新 M-DQN，
测试 geodesic reward 是否能提升 approver 的路径质量。

## 方法

**核心思路**：V22-D 的 M-DQN approver 用欧几里得进度奖励训练，
审批时可能批准绕路方向。换用 geodesic reward 训练的 M-DQN，
期望 approver 更倾向于沿真实最短路方向审批。

**配置**：
- Proposer：V16-C DDQN（不变，H=45, k_len=0.10）
- Approver：V29-A M-DQN（H=45, k_len=0.30 + `--reward-geodesic-progress`）
- 融合：negotiate K=10

## 代码改动

零新增代码。复用 V28 的 `--reward-geodesic-progress` flag，
仅在训练 M-DQN 时加该 flag。

## 主要结果（smoke runs=3）

| 配置 | short SR | short path | long SR | long path |
|------|----------|------------|---------|-----------|
| V29-A ensemble smoke | 1.00 | 13.40m | 1.00 | 48.50m |
| V22-D K=10 (formal) | 0.95 | 15.75m | 1.00 | **46.11m** |
| A*-MPC (formal) | 0.95 | 15.77m | 1.00 | 42.93m |

## 结论

**负面结果。** Long path 48.50m 劣于 V22-D 46.11m，不进 formal。
