# V28 CHANGES — Geodesic Progress Reward

## 改动性质

**纯奖励函数改动**，零网络结构/观测空间变化。

## 核心改动

### env.py

| 改动 | 说明 |
|------|------|
| 新增参数 `reward_geodesic_progress: bool = False` | 控制是否启用测地进度奖励 |
| 新增 `self._geo_reward_field` 缓存 | episode 开始时缓存 Dijkstra 全分辨率距离场 |
| 重构 `_set_goal_xy` 测地计算块 | obs/reward 共用同一次 Dijkstra，避免重复计算 |
| 新增辅助方法 `_geo_reward_dist(x_m, y_m)` | 连续坐标→格子查找，返回测地距离（米） |
| `_step_with_controls` 新增分支 | `reward_geodesic_progress=True` 时用测地进度替换欧几里得进度 |

```python
# 旧：欧几里得进度
reward += self.reward_k_p * float(dist_before - dist_after)

# 新（geodesic=True）：测地进度
geo_b = self._geo_reward_dist(x_before, y_before)
geo_a = self._geo_reward_dist(x_m_after, y_m_after)
reward += self.reward_k_p * float(geo_b - geo_a)
```

### cli/train.py

新增 `--reward-geodesic-progress` flag（action=store_true，默认 False）。

## 训练命令

```bash
# Run A: baseline（V16-C 复现，150ep）
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 150 \
  --forest-reward-k-len 0.10 --forest-reward-k-p 12.0 \
  --forest-adm-horizon 45 \
  --out v28-A-baseline --device cuda"

# Run B: geodesic reward（150ep）
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 150 \
  --forest-reward-k-len 0.10 --forest-reward-k-p 12.0 \
  --forest-adm-horizon 45 --reward-geodesic-progress \
  --out v28-B-geodesic --device cuda"

# Run B2: geodesic reward（300ep，正式）
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.10 --forest-reward-k-p 12.0 \
  --forest-adm-horizon 45 --reward-geodesic-progress \
  --out v28-B-geo-ep300 --device cuda"
```
