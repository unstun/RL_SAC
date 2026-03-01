# V36 代码改动记录

## 新增文件

无新文件（所有改动都在已有文件中）

## 修改文件

### env.py

- `DualScaleWrapper.__getattr__`: 新增属性转发（gymnasium.Wrapper 不自动转发），
  修复 `train_one()` 访问 `env.map_spec.name` 时 AttributeError
- `global_map_flat()`: 新增方法，返回降采样后的全局地图 (8×8 flattened = 64-dim)
- `DualScaleWrapper`: 新增类，包装 AMRBicycleEnv，observation 追加 64-dim 全局地图

### networks.py

- `DualScaleCNNQNetwork`: 双路 CNN（local 12×12 + global 8×8），obs_dim=218
  - local_conv: Conv(1→32,k3s1) → Conv(32→64,k3s2) → Conv(64→64,k3s2)
  - global_conv: Conv(1→32,k3s1) → Conv(32→64,k3s2)
- `DRQNNetwork`: CNN encoder → fc → LSTM(256,256) → Q
  - forward(x, hidden) → (q_values, hidden_out)
  - init_hidden(batch, device)
- `HistTransformerNetwork`: CNN → proj(→128) → TransformerEncoder(2L,4H) over T=8 → Q
  - encode_step(x) → feature
  - forward(hist_buf: list[tensor]) → Q

### agents.py

- `AlgoArch`: 扩展 Literal 加 "cnn-dual"/"cnn-drqn"/"cnn-hist"
- `parse_rl_algo()`: 新增 cnn-dual-ddqn/cnn-drqn/cnn-histformer 分支
- `DQNFamilyAgent.__init__`: 新增 cnn-dual 分支（使用 DualScaleCNNQNetwork）
- `DQNFamilyAgent.load_checkpoint`: 新增 cnn-dual 分支
- `EpisodeReplayBuffer`: 新增类，存完整 episode，采样连续子序列
- `DRQNAgent`: 新增类，维护 LSTM hidden state，BPTT 训练
- `HistFormerAgent`: 新增类，维护 rolling feature buffer，Transformer 推理

### train.py

- `canonical_all`: 追加 cnn-dual-ddqn/cnn-drqn/cnn-histformer
- `has_dqn_algos`: 排除 cnn-drqn/cnn-histformer（使用独立训练函数）
- DQN 家族分支：
  - `_train_env` / `_train_demo_data` 局部变量
  - cnn-dual-ddqn 时用 DualScaleWrapper 包装 env，demo 设为 None
- cnn-drqn dispatch block: 调用 `train_one_drqn()`
- cnn-histformer dispatch block: 调用 `train_one_histformer()`
- `train_one_drqn()`: 独立训练循环（EpisodeReplayBuffer，LSTM hidden state）
- `train_one_histformer()`: 独立训练循环（rolling feature buffer）

### infer.py

- `canonical_all`: 追加三种新 algo
- `algo_label` / `algo_seed_offset`: 新增条目
- `dqn_algo_paths`: 排除三种新 algo（跳过 obs_dim 检查）
- `resolve_model_path`: 新增 fallback 查找 `models/<algo>.pt`（V36-A/C 直接保存在此路径）
- cnn-dual-ddqn agent 加载（DQNFamilyAgent，obs_dim=218，DualScaleWrapper）
- cnn-drqn/histformer agent 加载（DRQNAgent/HistFormerAgent + load()）
- rollout dispatch: cnn-dual-ddqn 调 rollout_agent，cnn-drqn/histformer 调专用函数
- `rollout_drqn()`: 维护 LSTM hidden state，每 rollout 开始 reset_episode()
- `rollout_histformer()`: 维护 rolling feature buffer，每 rollout 开始 reset_episode()
