# v9p1 CHANGES

## 代码改动

### `forest_vehicle_dqn/env.py`
- `__init__`: 新增 3 个参数 `reward_k_astar_dev`, `reward_astar_corridor_m`, `reward_astar_progress`
- `reset()`: 当 A* 引导启用时，调用 `_compute_astar_ref_path()` 计算参考路径
- `_step_with_controls()`:
  - 进度奖励：当 `reward_astar_progress=True` 时，用 A* 剩余距离替代欧几里得距离
  - 新增 A* 走廊惩罚：`-k_astar_dev * max(0, dist_to_astar - corridor_m)`
- 新增 `_compute_astar_ref_path()`: 调用 Grid A* 计算参考路径，转为连续坐标+累积长度
- 新增 `_astar_ref_query()`: 查询当前位置到 A* 路径的距离和剩余路径长度

### `forest_vehicle_dqn/cli/train.py`
- 新增 3 个 CLI 参数：`--forest-reward-k-astar-dev`, `--forest-reward-astar-corridor-m`, `--forest-reward-astar-progress`
- 传递到 `AMRBicycleEnv` 构造函数

### `configs/v9p1.json`
- 基于 v7p1，新增 A* 引导参数

## 未改动
- 网络结构、DQfD 预训练、课程学习、gating 策略均保持 v7p1 不变
- `k_kappa`, `k_delta`, `k_len` 保持 v7p1 默认值（不加额外曲率惩罚）
