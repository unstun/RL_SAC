# v9p5 实施计划 — 航向对齐奖励 + 增加训练量

## 版本定位

- 上一版本：v9p4（失败，Dijkstra 偏离惩罚）
- 基线：v9p2（当前最佳，SR=95%/95%，short path 赢 baseline -1.7%）
- 本版核心：新增航向对齐奖励 `r_heading = k_h · cos(α_goal)` + episodes 300→500
- 状态：待实现

## 动机

v9p2 的主要瓶颈是曲率过高（short 0.127 vs baseline 0.084，long 0.160 vs 0.060）。
当前 progress reward 只看"离目标近了多少"（欧氏距离差），不关心 agent 朝哪个方向走。
航向对齐奖励鼓励 agent 面朝目标走直线，从而降低曲率、缩短路径、减少耗时。

关键优势：
- 纯局部信息（`α_goal` 已在观测空间中），不依赖任何外部规划器
- 不替换现有奖励项，仅叠加，风险低
- 学术上干净：observation 里已有 `cos(α_goal), sin(α_goal)`，reward 里加对应信号是自然的

## 实施步骤

### 步骤 1：env.py 构造函数新增参数

文件：`forest_vehicle_dqn/env.py`
位置：约第 761-765 行（在 `reward_k_dij_dev` 之后，`dij_dev_anneal_start_frac` 之前）

新增一行构造参数：

```python
        # Heading alignment reward (v9p5)
        reward_k_heading: float = 0.0,
```

在 `__init__` 体内（约第 880-910 行，其他 reward_k_* 赋值附近）新增：

```python
        self.reward_k_heading = float(reward_k_heading)
```

### 步骤 2：env.py 奖励计算新增航向项

文件：`forest_vehicle_dqn/env.py`
位置：`_step_with_controls` 方法，约第 1579 行（Dijkstra 偏离惩罚之后）

注意：`alpha`（目标相对角）在第 1464 行已经计算好了：
```python
alpha = self._goal_relative_angle_rad()  # 范围 [-π, π]
```

在第 1579 行之后插入：

```python
        # Heading alignment reward: encourage facing the goal (v9p5)
        if self.reward_k_heading > 0.0:
            reward += self.reward_k_heading * float(math.cos(alpha))
```

物理含义：
- `cos(α_goal) = 1.0` 当 agent 正对目标 → 最大正奖励
- `cos(α_goal) = 0.0` 当 agent 侧对目标 → 零奖励
- `cos(α_goal) = -1.0` 当 agent 背对目标 → 最大负惩罚

### 步骤 3：cli/train.py 传参

文件：`forest_vehicle_dqn/cli/train.py`
位置：约第 3915-3917 行（env 创建处，`reward_k_dij_dev` 传参附近）

新增一行：

```python
                reward_k_heading=float(getattr(args, "forest_reward_k_heading", 0.0)),
```

### 步骤 4：创建 v9p5.json 配置

文件：`configs/v9p5.json`

基于 v9p4.json，做以下修改：
1. 去掉 `forest_reward_k_dij_dev`（设为 0 或删除）
2. 去掉 `dij_dev_anneal_start_frac` / `dij_dev_anneal_end_frac`（或保留默认值）
3. 新增 `forest_reward_k_heading: 0.3`（初始值，可调）
4. `episodes` 从 300 改为 500
5. 更新 `_meta` 信息

关键参数对比（v9p2 → v9p5）：

| 参数 | v9p2 值 | v9p5 值 | 说明 |
|------|---------|---------|------|
| `episodes` | 300 | 500 | 增加训练量 |
| `forest_reward_k_t` | 0.15 | 0.15 | 不变 |
| `forest_reward_k_len` | 0.02 | 0.02 | 不变 |
| `forest_reward_k_delta` | 0.8 | 0.8 | 不变 |
| `forest_reward_k_dij_dev` | 0.0 | 0.0 | 不变（v9p4 的 0.05 去掉） |
| `forest_reward_k_heading` | N/A | **0.3** | 新增 |

k_heading=0.3 的量级分析：
- 每步航向奖励范围：[-0.3, +0.3]
- 对比 k_t=0.15（每步固定 -0.15）：同量级
- 对比 k_p=12.0 · Δd（典型 ~0.1-0.5/步）：约 1/4-1/2
- 不会压过 progress reward，但足以影响方向选择

### 步骤 5：创建 v9p5 留档四件套

在 `docs/versions/v9p5/` 下创建：

1. `README.md` — 版本总结（方法、参数、状态）
2. `CHANGES.md` — 具体改动清单
3. `RESULTS.md` — 结果对比（smoke 后填写）
4. `runs/README.md` — run 路径与口径

## 验证命令

```bash
# self-check
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 python train.py --self-check"
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 python infer.py --self-check"

# smoke 训练 (episodes=150)
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p5 --episodes 150 \
  --out v9p5_smoke --device cuda --progress"

# smoke 推理 (runs=3)
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python infer.py --profile v9p5 --models v9p5_smoke --out v9p5_smoke_r3 \
  --runs 3 --progress"
```

## 成功标准（smoke 门）

对比 v9p2 smoke 基线（SR=100%/100%，short path=16.52m）：
- SR 不低于 v9p2（short ≥ 90%，long ≥ 80%）
- 曲率有所下降（目标：short < 0.12，long < 0.15）
- 路径长度不恶化

若 smoke 通过，进入 full 评测（episodes=500, runs=20）。

## 风险点

1. k_heading 过大可能导致 agent 过度追求朝向而忽略避障 → 初始值 0.3 保守
2. 靠近目标时需要减速转向停车，航向奖励可能干扰 → 但 reached +1000 远大于 0.3
3. Demo 兼容性：专家本身就面朝目标走，demo reward 只会更高，不会 mismatch
4. 若 k_heading=0.3 效果不明显，可尝试 0.5 或 0.8（v9p5p1）
