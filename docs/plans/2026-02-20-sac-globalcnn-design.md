# SAC-GlobalCNN 设计文档（v8 候选）

> 日期：2026-02-20
> 状态：已批准，待实施
> 目标版本：v8

## 1. 背景与动机

### 当前状态（v7p1）

- 算法：CNN-DDQN + DQfD，离散动作空间 15×15=225
- 观测：12×12 局部占用栅格 + 10 维标量
- 推理：shielded（top-k admissible replacement）
- 最佳结果（runs=5）：short/mid/long SR 均 1.00

### 核心差距

v7p1 在成功率上已追平 Hybrid A*-MPC，但路径质量全面落后：

- mid/long 路径长 20-50%
- 所有场景时间慢 15-50%
- 曲率高 2-3 倍

### 结构性病因

1. **全局盲区**：agent 只看 12×12 局部地图，无法全局规划 → 路径长
2. **离散动作粗糙**：225 个离散格点，控制精度不如 MPC → 曲率高
3. **Reward 失衡**：进度奖励 (k_p=12) 远大于时间惩罚 (k_t=0.1) → 不优化路径效率

## 2. 方案概述

**SAC-GlobalCNN**：用 SAC（连续动作）+ 全局地图观测 + 路径质量导向 Reward，
实现纯端到端 RL agent 在路径质量上全面超越 Hybrid A*-MPC。

### 论文叙事

纯 RL 替代传统规划：通过全局感知 + 连续控制 + 路径质量优化，
端到端 RL agent 无需手工 heuristic 或搜索算法，即可产生更短、更快、更平滑的路径。

## 3. 详细设计

### 3.1 算法：DDQN → SAC

- **Soft Actor-Critic**：连续动作空间，最大熵框架
- 动作输出：(steering_rate, acceleration) ∈ [-1, 1]²，tanh 压缩
- 映射到物理量：δ̇ = output[0] × max_delta_dot，a = output[1] × max_accel
- Twin Q-network + 自动温度调节（target entropy = -2）
- Soft target update（τ=0.005）

### 3.2 观测空间

**地图分支（Global CNN）：**

- 输入：整个地图下采样到 48×48，3 通道
  - Ch0：占用栅格（0=free, 1=obstacle）
  - Ch1：agent 位置热力图（高斯 blob）
  - Ch2：goal 位置热力图（高斯 blob）
- CNN 结构：
  - Conv2d(3, 32, 3×3, stride=2, pad=1) → ReLU → 24×24
  - Conv2d(32, 64, 3×3, stride=2, pad=1) → ReLU → 12×12
  - Conv2d(64, 128, 3×3, stride=2, pad=1) → ReLU → 6×6
  - Conv2d(128, 128, 3×3, stride=2, pad=1) → ReLU → 3×3
  - Flatten → 1152 dims

**标量分支：**

- 保留原有 10 维：(ax_n, ay_n, gx_n, gy_n, sin_ψ, cos_ψ, v_n, δ_n, α_n, od_n)
- 新增 2-4 维：prev_steering_rate, prev_acceleration, path_length_so_far_n
- 总计 12-14 维

**融合：**

- Concat(CNN_output, scalar_features) → 1164~1166 dims
- Actor MLP: → 256 → 256 → 2 (mean) + 2 (log_std)
- Critic MLP: state_dim + action_dim → 256 → 256 → 1 (×2 Twin Q)

### 3.3 Reward 工程

当前 reward 以进度奖励为主导（k_p=12 vs k_t=0.1），改为路径质量导向：

| 组件 | 当前值 | 目标值 | 目的 |
| ------ | -------- | -------- | ------ |
| 时间惩罚 k_t | 0.1 | 0.5~1.0 | 强烈鼓励快速到达 |
| 路径长度惩罚 k_len | 无 | 新增 0.3~0.5 | 每步 -k_len × step_distance |
| 曲率惩罚 k_kappa | 0.2 | 0.8~1.5 | 鼓励平滑路径 |
| 路径效率奖励 | 无 | 新增 | progress / distance_traveled |
| 进度奖励 k_p | 12.0 | 8.0 | 降低主导地位 |

核心思路：从"只要到达就行"变成"用最短、最快、最平滑的方式到达"。

### 3.4 训练流程（总计 ≤ 4 小时）

**Phase 0 — Behavior Cloning 预训练（~30 分钟）：**

- 用现有 Hybrid A*-MPC 专家轨迹训练 Actor
- 目标：让 SAC 从"会开车"的起点开始
- 大幅加速后续 RL 训练收敛

**Phase 1 — SAC 训练 + 专家数据混合（~2.5 小时）：**

- Replay buffer 中混入专家轨迹（占比 25%~50%）
- 自动熵调节（target entropy = -dim(action) = -2）
- Curriculum：先 short 场景，逐步加入 mid/long
- 每 N 步评估，保存最优 checkpoint

**Phase 2 — 纯 RL 微调（~30 分钟，可选）：**

- 去掉专家数据，纯 on-policy 微调
- 目的：让 agent 超越专家（Hybrid A*-MPC）

### 3.5 推理策略

- 使用 SAC 确定性策略（取高斯分布均值），不需要 ε-greedy
- 目标：strict-argmax（纯 RL 输出），不依赖 shielded inference
- 论文中报告两组结果：strict 和 shielded，展示 RL 本身能力

### 3.6 关键超参初始值

- lr_actor=3e-4, lr_critic=3e-4
- gamma=0.99
- batch_size=256
- replay_buffer_size=1e6
- tau=0.005（soft target update）
- alpha 自动调节

## 4. 对比总结

| 模块 | v7p1 (CNN-DDQN) | v8 (SAC-GlobalCNN) |
| ------ | ------------------ | --------------------- |
| 算法 | DDQN + DQfD | SAC + BC 预训练 |
| 动作空间 | 离散 15×15=225 | 连续 2D (δ̇, a) |
| 观测-地图 | 局部 12×12 (1ch) | 全局 48×48 (3ch) |
| 观测-标量 | 10 维 | 12-14 维 |
| Reward | 进度主导 | 路径质量主导 |
| 推理 | shielded | strict-argmax |
| 训练时间 | ~2h | ≤4h |

## 5. 风险与缓解

- 风险 1：全局 48×48 分辨率不够 → 升到 64×64 或加 local crop 通道
- 风险 2：SAC 训练不稳定 → BC 预训练 + 专家数据混合提供稳定起点
- 风险 3：4 小时内训不够 → 先跑 smoke (150 ep) 快速验证方向

## 6. 验收门槛

按 CLAUDE.md 第 13 条，最终需在 short/long 双套件 runs=20 下满足：

- success_rate(SAC-GlobalCNN) >= success_rate(Hybrid A*-MPC)
- avg_path_length(SAC-GlobalCNN) < avg_path_length(Hybrid A*-MPC)
- path_time_s(SAC-GlobalCNN) < path_time_s(Hybrid A*-MPC)
