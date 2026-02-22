# v9p1: A*-guided reward shaping (走廊惩罚 + A* 进度)

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p1`（CNN-DDQN 稳定基线，SR=100%）
- 本版口径：`shielded/hybrid`（训练与推理一致）
- 状态：**smoke 进行中**

## 版本背景

v9 尝试通过增大曲率/转向惩罚缩短路径，但 SR 降至 66.7%。
v9p1 换思路：不加额外惩罚，而是用 A* 最短路径引导 agent 走更短的路。

## 核心差距（v7p1 vs Hybrid A*-MPC, long 套件）

| 指标 | v7p1 | Hybrid A*-MPC | 差距 |
|------|------|---------------|------|
| SR | 100% | 100% | 持平 |
| path_length | 51.91m | 43.02m | +21% |
| path_time | 34.01s | 22.82s | +49% |

## 方法摘要

基于 v7p1 配置，新增两项 A*-guided reward shaping：

1. **A* 走廊惩罚**：每步计算 agent 到 A* 最短路径的距离，
   超过 corridor（1.5m）后线性惩罚 `k_astar_dev=1.0`
2. **A* 进度替代**：用 A* 路径剩余距离替代欧几里得距离
   作为进度奖励信号，只有沿最短路径前进才有正奖励

其余参数（网络结构、DQfD、课程、gating 等）保持 v7p1 不变。

## 新增参数

| 参数 | 值 | 说明 |
|------|----|------|
| `forest_reward_k_astar_dev` | 1.0 | A* 走廊偏离惩罚系数 |
| `forest_reward_astar_corridor_m` | 1.5 | 走廊容忍半径（米） |
| `forest_reward_astar_progress` | true | 启用 A* 进度替代 |

## 关键命令

```bash
# 训练 (smoke)
conda run -n ros2py310 python train.py --profile v9p1 --episodes 150 \
  --out v9p1-smoke1 --device cuda --progress --save-ckpt best
# 推理 (smoke)
conda run -n ros2py310 python infer.py --profile v9p1 \
  --models v9p1-smoke1 --out v9p1-smoke1 --runs 3 --progress
```

## 代表 Run

- 训练：`runs/v9p1-smoke1/` (待填)
- 推理：(待填)
