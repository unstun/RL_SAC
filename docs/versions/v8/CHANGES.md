# v8 改动明细（相对 v7p1）

## 新增文件

| 文件 | 说明 |
|------|------|
| `forest_vehicle_dqn/sac_networks.py` | GlobalCNNEncoder + SACActor + SACCritic |
| `forest_vehicle_dqn/sac_agent.py` | SACAgent（含 SACConfig、SACReplayBuffer、BC 预训练、save/load） |
| `configs/v8.json` | v8 配置（SAC 超参 + 全局地图参数 + 路径长度惩罚） |
| `tests/test_sac_networks.py` | SAC 网络单元测试（3 tests） |
| `tests/test_sac_agent.py` | SACAgent 单元测试（3 tests） |
| `tests/test_global_obs.py` | 全局地图观测单元测试（3 tests） |

## 修改文件

| 文件 | 改动 |
|------|------|
| `forest_vehicle_dqn/env.py` | 新增 `get_global_map()`、`observe_sac()` 方法；新增 `reward_k_len` 构造参数（路径长度惩罚） |
| `forest_vehicle_dqn/cli/train.py` | 新增 `train_one_sac()` 函数；`main()` 中添加 `cnn-sac` 分支；注册 SAC CLI 参数 |
| `forest_vehicle_dqn/cli/infer.py` | 新增 `rollout_sac()` 函数；`main()` 中添加 `cnn-sac` 分支；注册 SAC CLI 参数 |

## 关键参数变更

| 参数 | v7p1 | v8 | 说明 |
|------|------|-----|------|
| `rl_algos` | `cnn-ddqn` | `cnn-sac` | 算法切换 |
| `forest_action_mode` | N/A（离散） | `continuous` | 连续动作空间 |
| `reward_k_len` | N/A | 0.4 | 路径长度惩罚权重 |
| `global_map_size` | N/A | 48 | 全局地图分辨率 |
| `global_map_channels` | N/A | 3 | 全局地图通道数 |
| `sac_batch_size` | N/A | 256 | SAC 批大小 |
| `sac_buffer_size` | N/A | 1000000 | 经验回放容量 |
| `sac_bc_pretrain_steps` | N/A | 5000 | BC 预训练步数 |
