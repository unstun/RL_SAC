# v11 CHANGES — 相对 v9p2 的改动

## 新增文件

- `forest_vehicle_dqn/modules.py` — NoisyLinear, CBAM (ChannelAttention + SpatialAttention), SpatialMHA

## 修改文件

### `forest_vehicle_dqn/networks.py`
- CNNQNetwork 新增 5 个 init 参数: cbam, noisy_net, mha, mha_heads, n_quantiles
- `_lin()` 辅助函数: noisy_net → NoisyLinear, 否则 → nn.Linear
- forward: conv → CBAM → SpatialMHA → flatten+concat → FC/Dueling
- Dueling+QR-DQN 组合: V stream 输出 n_quantiles, A stream 输出 n_actions*n_quantiles
- `reset_noise()`: 递归重置所有 NoisyLinear 子模块

### `forest_vehicle_dqn/agents.py`
- AgentConfig 新增: cbam, noisy_net, mha, mha_heads, n_quantiles
- `_q_values()`: 将 raw 网络输出转换为 Q 值（QR-DQN: reshape+mean）
- `_quantile_td_loss()`: Quantile Huber loss 实现
- `_next_q_quantiles()`: QR-DQN 目标分位数计算
- act/act_masked: NoisyNet 跳过 ε-greedy
- update(): QR-DQN 分支 (legacy + DQfD 双路径)

### `forest_vehicle_dqn/cli/train.py`
- 新增 5 个 CLI args: --cbam, --noisy-net, --mha, --mha-heads, --n-quantiles
- 修复 QR-DQN 兼容: 2处直接 agent.q() → agent._q_values() 包装

### `forest_vehicle_dqn/cli/infer.py`
- 修复 QR-DQN 兼容: 1处直接 agent.q() → agent._q_values() 包装

## Bug 修复

- QR-DQN action index overflow: 直接对 raw output argmax 时 index 超出 n_actions 范围
  - 原因: raw output = n_actions * n_quantiles, argmax 返回 [0, n_actions*n_quantiles)
  - 修复: 所有 agent.q() 直接调用 → agent._q_values() 包装后再 argmax
