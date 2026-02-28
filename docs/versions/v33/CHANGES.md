# V33 CHANGES

## 新文件

### `forest_vehicle_dqn/td3_networks.py` (~27 行)
- `TD3Actor(nn.Module)`: 确定性策略网络
  - 输入: GlobalCNNEncoder 编码的 maps + scalars
  - 输出: tanh(MLP) → [-1,1]² (delta_dot, accel)

### `forest_vehicle_dqn/td3_agent.py` (~254 行)
- `TD3Config`: TD3 超参数 (explore_noise=0.1, target_noise=0.2, policy_delay=2)
- `TD3Agent`: 完整 TD3 实现
  - actor + actor_target + SACCritic + critic_target
  - act(), observe(), update(), warmup_critic(), save(), load()

## 修改文件

### `forest_vehicle_dqn/cli/train.py`
- `train_one_sac()`: 新增 `agent_override` 参数支持 TD3 注入
- `canonical_all`: 加入 `"cnn-td3"`
- algo 解析循环: 支持 `"cnn-td3"` 解析
- algo dispatch: 新增 TD3 分支（构建 TD3Agent → 调用 train_one_sac）
- 模型文件名: 动态使用 `_model_name`（cnn-td3 or cnn-sac）
- SAC-specific 属性: 用 `getattr` 兼容 TD3 (alpha, pretrain_bc)

### `forest_vehicle_dqn/cli/infer.py`
- `canonical_all`: 加入 `"cnn-td3"`
- algo 解析循环: 支持 `"cnn-td3"`
- display name: `"cnn-td3": "CNN-TD3"`
- seed offset: `"cnn-td3": 110_000`
- obs_dim 检查: 跳过 TD3
- 模型加载: TD3Agent 分支（放入 sac_agents 复用 rollout_sac）
