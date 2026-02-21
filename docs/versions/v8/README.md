# v8: SAC-GlobalCNN（连续动作 + 全局 48×48 地图）

## 版本目标

将强化学习算法从离散动作 CNN-DDQN 迁移到连续动作 SAC（Soft Actor-Critic），使用全局 48×48 三通道 CNN 编码器替代局部 12×12 占据栅格，以提升路径质量（更短路径、更平滑曲率）。

## 方法摘要

- **算法**: SAC（Soft Actor-Critic）+ 自动温度调节（learnable α）
- **网络**: GlobalCNNEncoder（4 层 Conv2d → 拼接标量） + SACActor（Gaussian policy + tanh squash） + SACCritic（Twin Q）
- **观测**: 3 通道 48×48 全局地图（占据/起点 blob/终点 blob） + 12 维标量（位姿、速度、目标方向等）
- **动作**: 连续 2 维（转向角速率 δ̇、加速度 a），tanh 压缩到 [-1, 1]
- **奖励**: 在 v7p1 基础上新增 `reward_k_len`（路径长度惩罚，默认 0.4）
- **预训练**: BC（行为克隆）预训练 5000 步，使用 Hybrid A*-MPC 专家轨迹

## 关键命令

```bash
# 训练
conda run -n ros2py310 python train.py --profile v8 --episodes 3000 --out v8

# 推理
conda run -n ros2py310 python infer.py --profile v8 --runs 20 --models runs/v8/<train_dir>
```

## 代表 run

- **smoke 训练**: `runs/v8-smoke/train_20260221_160435`（150 episodes, ubuntu-zt RTX 5070 Ti）
- **smoke 推理**: `runs/v8-smoke/train_20260221_160435/infer/20260221_161349`（runs=3）

## 结论

Smoke 测试（150 episodes）管线端到端通过，但 SAC 尚未收敛（success_rate = 0%）。训练回报从 -3527 改善到 -200~-400，有学习信号。需要更多 episodes（≥1000）才能评估实际性能。

## 下一步

1. 全量训练（3000 episodes）并评估 short/long runs=20
2. 若 success_rate 仍为 0，排查：动作映射范围、奖励尺度、BC 预训练质量
3. 对标 v7p1（CNN-DDQN）和 Hybrid A*-MPC 基线
