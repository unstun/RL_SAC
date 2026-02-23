# v9p5 改动清单

## 代码改动

### forest_vehicle_dqn/env.py

1. 构造函数新增参数 `reward_k_heading: float = 0.0`
2. `__init__` 体内新增 `self.reward_k_heading = float(reward_k_heading)`
3. `_step_with_controls` 中 Dijkstra 偏离惩罚之后插入：
   ```python
   if self.reward_k_heading > 0.0:
       reward += self.reward_k_heading * float(math.cos(alpha))
   ```

### forest_vehicle_dqn/cli/train.py

1. env 创建处新增传参：
   ```python
   reward_k_heading=float(getattr(args, "forest_reward_k_heading", 0.0)),
   ```

## 配置改动

### configs/v9p5.json（新建）

- 基于 v9p4.json
- 去掉 `forest_reward_k_dij_dev`（不设置，默认 0）
- 去掉 `dij_dev_anneal_start_frac` / `dij_dev_anneal_end_frac`
- 新增 `"forest_reward_k_heading": 0.3`
- `episodes`: 300 → 500

## 与 v9p2 的差异

仅新增 `forest_reward_k_heading=0.3` 和 `episodes=500`，其余参数完全一致。
