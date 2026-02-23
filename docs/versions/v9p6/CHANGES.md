# V9P6 Changes (vs V9P2)

## 代码改动

### env.py
- 新增参数 `obs_cost_to_go_channel: bool = False`
- 新增方法 `_update_cost_obs()`：预计算下采样 Dijkstra cost-to-go 地图（12×12，归一化到 [-1,1]）
- `_observe()`：当 `obs_cost_to_go_channel=True` 时，标量 10→11 维（新增 cost_n），地图 1→2 通道
- `_set_goal_xy()` / `_update_start_dependent_fields()`：goal 变更时同步更新 cost obs

### networks.py
- `infer_flat_obs_cnn_layout()`：新增 `(11, 2)` 布局支持（11 标量 + 2×N×N 地图）

### cli/train.py
- 新增 CLI 参数 `--forest-obs-cost-to-go-channel`
- `demo_ce_lambda`：DQfD 模式下不再强制为 0，改为从 config 读取

### cli/infer.py
- 新增 CLI 参数 `--forest-obs-cost-to-go-channel`
- 传递 `obs_cost_to_go_channel` 到 `AMRBicycleEnv`

## 配置改动 (configs/v9p6.json vs v9p2.json)

| 参数 | V9P2 | V9P6 | 说明 |
|------|------|------|------|
| `forest_obs_cost_to_go_channel` | (无) | `true` | 2 通道地图 + cost_n 标量 |
| `forest_reward_dijkstra_progress` | (无) | `true` | cost-to-go progress |
| `forest_expert_exploration` | `false` | `true` | 训练时混合专家 |
| `forest_expert_prob_start` | (无) | `0.7` | 专家概率起始值 |
| `forest_expert_prob_final` | (无) | `0.0` | 专家概率终止值 |
| `forest_expert_prob_decay` | (无) | `0.6` | 衰减到 60% episodes |
| `demo_ce_lambda` | (无) | `1.0` | CE 行为克隆损失 |
| `no_terminate_on_stuck` | `true` | `false` | 启用 stuck 检测 |
