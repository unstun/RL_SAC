# CNN-DDQN 系统架构文档

> 本文档描述 `RL_sac` 仓库中 CNN-DDQN 的完整模块组成与数据流。

## 模块总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练主循环 (train.py)                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Demo 收集 │→│ DQfD 预训练   │→│ Episode 循环│→│ 检查点保存  │  │
│  └──────────┘  └──────────────┘  └─────┬─────┘  └───────────┘  │
│                                        │                        │
│         ┌──────────────────────────────┼──────────────┐         │
│         ▼                              ▼              ▼         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐       │
│  │ Environment  │  │    Agent     │  │  Replay Buffer   │       │
│  │(AMRBicycleEnv)│  │(DQNFamily)  │  │  (PER/Stratified)│       │
│  └──────┬──────┘  └──────┬───────┘  └──────────────────┘       │
│         │                │                                      │
│         ▼                ▼                                      │
│  ┌─────────────┐  ┌──────────────┐                              │
│  │ Reward 塑形  │  │ CNN Q-Network│                              │
│  │ + 动力学模型 │  │ + Target Net │                              │
│  └─────────────┘  └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## 1. 神经网络 — CNNQNetwork

**文件**：`forest_vehicle_dqn/networks.py`

CNN Q-网络负责从「标量特征 + 全局占用栅格」映射到各离散动作的 Q 值。

### 输入解析（`infer_flat_obs_cnn_layout`）

观测向量是一维展平的，网络自动推断布局：

| 环境 | 标量维度 | 地图通道 | 地图尺寸 | 总维度 |
|------|---------|---------|---------|--------|
| AMRGridEnv | 5 | 1 | 12×12 | 149 |
| AMRBicycleEnv | 10 | 1 | 12×12 | 154 |

**标量特征（AMRBicycleEnv，10 维，按代码实际顺序）**：

- 位置：`[x_norm, y_norm]`（归一化到 [-1, 1]）
- 目标：`[goal_x_norm, goal_y_norm]`
- 航向：`[sin(ψ), cos(ψ)]`（注意 sin 在前）
- 速度/转向：`[v / v_max, δ / δ_max]`
- 目标相对角：`[α_goal / π]`（单值，非 cos/sin 分量）
- 障碍物距离：`[od_norm]`（最近障碍物距离，归一化到 [-1, 1]）

### 网络结构

```
标量 (10d) ──────────────────────────────────────┐
                                                  │ concat
地图 (1×12×12) → Conv2d(1→32, k3s1p1) + ReLU     │
               → Conv2d(32→64, k3s2p1) + ReLU     │
               → Conv2d(64→64, k3s2p1) + ReLU     │
               → Flatten ────────────────────────→ ⊕
                                                   │
                                          Linear(→256) + ReLU
                                          Linear(256→256) + ReLU
                                          Linear(256→35)  ← Q(s,a) 每个动作一个值
```

- 隐藏层维度：`hidden_dim=256`，层数：`hidden_layers=3`（AgentConfig 默认值；网络代码默认 2，但被配置覆盖）
- 输出维度 = 动作数（默认 35 = 7 转向角速度 × 5 加速度）
- 目标网络（`q_target`）结构相同，参数延迟同步

## 2. Agent — DQNFamilyAgent

**文件**：`forest_vehicle_dqn/agents.py`

统一的 DQN 系列 Agent，通过 `parse_rl_algo` 选择变体：

| 算法名 | 架构 | Q 目标计算 | 目标网络更新 |
| ------ | ---- | --------- | ----------- |
| `cnn-dqn` | CNN | `max_a Q_target(s', a)` | 硬更新 |
| `cnn-ddqn` | CNN | `Q_target(s', argmax_a Q(s', a))` | 硬更新 |
| `cnn-pddqn` | CNN | 同 ddqn | Polyak 软更新 |

### 2.1 动作选择

**ε-greedy 探索**（`act` / `act_masked`）：

- `ε = linear_epsilon(ep, start=0.9, final=0.01, decay=2000)`
- 以 ε 概率随机选动作，否则 `argmax Q(s, a)`
- `act_masked`：对不可行动作设 `-inf` 后再 argmax（配合 Action Shield）

### 2.2 训练更新

支持两种 demo 训练模式（`demo_mode` 参数）：

**`dqfd` 模式**（默认，严格 DQfD）：PER 采样 + 1-step TD + n-step TD + 专家边际损失 + L2 正则，无交叉熵。

```text
L_total = L_1step + λ_n · L_nstep + λ_margin · L_margin + λ_l2 · L_l2
```

**`legacy` 模式**：n-step TD + 专家边际 + 可选交叉熵（`demo_ce_lambda`），可搭配分层采样。

| 损失项 | 公式 | 作用 |
| ------ | ---- | ---- |
| 1-step TD | `SmoothL1(Q(s,a), r + γ Q_target(s', argmax Q(s')))` | 基础值估计 |
| n-step TD | 同上但用 n-step 累积回报 `R_n + γ^n Q_target` | 加速信用分配 |
| 专家边际 | `ReLU(max_{a'≠a_demo} Q(s,a') + margin - Q(s,a_demo))` | 仅对 demo 样本，强制专家动作 Q 值最高 |
| L2 正则 | `Σ ‖θ‖²` | 防过拟合（仅 dqfd 模式） |
| 交叉熵 | `CE(softmax(Q(s)), a_demo)` | 仅 legacy 模式，可选 |

### 2.3 目标网络同步

- 硬更新：每 `target_update_steps=1000` 步完全复制 `q → q_target`
- 软更新（pddqn）：每步 `θ_target ← (1-τ)θ_target + τ·θ_online`

### 2.4 DQfD 预训练（`pretrain_on_demos`）

在正式 RL 训练前，仅用专家 demo 缓冲区跑 `pretrain_steps=30000` 步更新，让 Q 网络先学到合理的值估计。

### 2.5 辅助可行性头（`aux_adm_head`，可选）

额外的 BCE 监督头，预测下一状态各动作的可行性掩码。仅训练时使用，推理时不参与决策。

## 3. 经验回放缓冲区 — ReplayBuffer

**文件**：`forest_vehicle_dqn/replay_buffer.py`

### 3.1 存储内容

每条转换包含：

| 字段 | 说明 |
| ---- | ---- |
| `obs, action, reward_1, next_obs_1, done_1` | 1-step 转换 |
| `reward_n, next_obs_n, done_n, n_steps_n` | n-step 转换（Agent 侧计算后写入） |
| `next_action_mask` | 下一状态的可行动作掩码 |
| `demo` | 是否为专家演示 |
| `flags` | 位标志：`FLAG_DEMO / FLAG_NEAR_GOAL / FLAG_STUCK / FLAG_HAZARD` |

### 3.2 采样策略

**优先经验回放（PER）**：

- Sum-tree 实现，O(log N) 采样
- 优先级：`p_i = (|δ_i| + ε)^α`，其中 demo 的 `ε=1.0`（远大于 agent 的 `ε=1e-3`）
- 重要性采样权重：`w_i = (N · P(i))^{-β} / max w`，β 从 0.6 线性退火到 1.0
- 专家 demo 永不被覆盖（缓冲区满时优先淘汰 agent 转换）

**分层采样（Stratified，可选）**：按 flag 分桶，保证各类转换的采样比例。

## 4. 环境 — AMRBicycleEnv

**文件**：`forest_vehicle_dqn/env.py`

### 4.1 动力学模型（后轴中心 Ackermann 自行车模型）

状态 `[x, y, ψ, v, δ]`，Euler 积分（`dt=0.05s`）：

```text
v_{t+1} = clip(v_t + a·dt, -v_max, v_max)
δ_{t+1} = clip(δ_t + δ̇·dt, -δ_max, δ_max)
x_{t+1} = x_t + v_{t+1}·cos(ψ_t)·dt
y_{t+1} = y_t + v_{t+1}·sin(ψ_t)·dt
ψ_{t+1} = wrap(ψ_t + (v_{t+1}/L)·tan(δ_{t+1})·dt)
```

关键物理参数：`L=0.6m`（轴距）、`v_max=2.0m/s`、`δ_max=27°`、`δ̇_max=60°/s`

**终止条件（两阶段目标到达）**：

- `goal_pose_reached`：距离 ≤ `goal_tolerance_m` 且 相对角 ≤ `goal_angle_tolerance_rad`
- `goal_stop_reached`：速度 ≤ `goal_stop_speed_m_s` 且 转向角 ≤ `goal_stop_delta_rad`
- 成功终止 = `pose_reached AND stop_reached`（必须同时满足位姿和停车条件）
- 碰撞 / 超时 / 卡住 → 失败终止

### 4.2 动作空间（离散化）

`build_ackermann_action_table_35`：7 个 δ̇ × 5 个 a = 35 个离散动作。

### 4.3 奖励塑形（森林场景）

| 信号 | 系数/公式 | 说明 |
| ---- | --------- | ---- |
| 到达目标 | `+reward_reached`（默认 +1000） | 终止奖励 |
| 碰撞 | `+reward_collision`（默认 -200） | 终止惩罚 |
| 距离进展 | `k_p · Δd_goal` | 鼓励接近目标（欧氏距离差） |
| 障碍物距离（legacy） | `-k_o · (1/(od+ε) - 1/(safe_d+ε))` | 靠近障碍物时惩罚，上限 `reward_obs_max` |
| CBF 对数屏障 | `c_cbf · log(max(od/h_max, 1e-6))` | 替代 legacy 障碍物惩罚（互斥） |
| 速度-障碍耦合 | `-k_v · (1 - od/safe_speed_d) · (v/v_max)²` | 近障碍时惩罚高速 |
| 软速度上限 | `-k_c · max(0, v - v_cap)²` | `v_cap = v_max · clip(od/safe_speed_d)` |
| 卡住 | `-stuck_penalty`（默认 -100） | 速度≈0 且未到达目标 |
| 时间步 | `-reward_step`（默认 -0.1） | 每步固定惩罚 |
| 路径长度 | `-k_len · Δpath`（v9p2） | 累积路径长度惩罚 |
| 转向变化 | `-k_delta · (Δδ)²` | 平滑转向 |
| 加速度变化 | `-k_a · (Δa)² · (v/v_max)²` | 平滑加速，低速时不惩罚 |
| 曲率 | `-k_kappa · tan²(δ)` | 惩罚大转向角 |
| A* 走廊偏离 | `-k_astar_dev · max(0, d_astar - corridor_m)`（v9p1） | 偏离 A* 参考路径时惩罚 |
| Dijkstra 偏离 | `-k_eff · d_dij`（v9p4） | 偏离 Dijkstra 最短路径，带退火调度 |
| 航向对齐 | `+k_heading · cos(α_goal)`（v9p5） | 鼓励朝向目标 |
| 目标区域对齐 | `-k_goal_v · (v/v_thr)² - k_goal_δ · (δ/δ_thr)²` | 进入目标区后鼓励停车+摆正 |
| 目标区域时间 | `-reward_goal_time` | 在目标区内每步惩罚，催促停车 |
| 离开目标区 | `-reward_goal_leave` | 进入目标区后又离开的惩罚 |

**Dijkstra 偏离退火调度**（`_effective_k_dij_dev`）：训练前 `anneal_start_frac` 不施加，之后线性增长到 `k_dij_dev`，在 `anneal_end_frac` 达到满值。

### 4.4 可行性约束（Action Shield）

`admissible_action_mask(horizon_steps, min_od_m, min_progress_m)`：

- 对每个候选动作前向模拟 `horizon` 步（向量化批量 rollout）
- 检查：无碰撞 + 障碍物距离 ≥ `min_od` + 目标进展 ≥ `min_progress`
- 返回布尔掩码，训练和推理均使用

**倒车逻辑**（`allow_reverse`）：

- 当无前进可行动作时，允许安全倒车作为逃脱手段
- 条件：`v_end < -reverse_v_min` 且 `_reverse_unlocked()`（低间隙或连续无进展）
- 训练和推理均可启用

### 4.5 专家策略（用于 Demo 收集）

| 专家 | 方法 | 说明 |
| ---- | ---- | ---- |
| `hybrid_astar` | Hybrid A* + 前瞻跟踪 | 默认专家 |
| `astar_mpc` | 网格 A* + 曲线优化 + MPC | 更平滑但更慢 |
| `hybrid_astar_mpc` | Hybrid A* + MPC 跟踪 | 最高质量 |

## 5. 训练管道

**文件**：`forest_vehicle_dqn/cli/train.py`

### 5.1 流程

```text
1. 创建环境 (AMRBicycleEnv)
2. 创建 Agent (DQNFamilyAgent)
3. 收集专家 Demo → 填入 PER 缓冲区
4. DQfD 预训练 (30k steps)
5. Episode 循环：
   ├─ 课程学习选择起终点距离
   ├─ ε-greedy / Action Shield 选动作
   ├─ 环境交互 → observe → n-step 缓冲
   ├─ 每 train_freq 步执行 update()
   ├─ 定期评估 (eval_every)
   └─ 保存最佳检查点
```

### 5.2 课程学习（Curriculum Learning）

**距离课程**（`--forest-curriculum`）：

- 训练初期起点靠近目标，随 episode 进展逐步扩大到全距离
- `curriculum_progress = clip(ep / (episodes × ramp), 0, 1)`
- 起点距离上界：`min_dist + progress × (full_dist - min_dist)`，带宽 `band_m`

**双套件训练**（`--forest-train-two-suites`）：

- short（6-14m）和 long（14m+）按比例混合
- 默认 short 占 35%，long 占 65%
- 90% 随机起终点 + 10% 固定起终点

**动态双套件课程**（`--forest-train-dynamic-curriculum`）：

- 根据最近评估的 short/long 成功率动态调整采样比例
- 成功率低于目标的套件获得更多采样机会
- 比例更新使用比例控制器，限制在 `[p_min, p_max]` 范围内

**PLR 课程**（Prioritized Level Replay，实验性）：

- 将距离范围离散化为多个 level
- 按 TD-error / episode return 优先采样困难 level
- 带 staleness 系数防止某些 level 长期不被采样

### 5.3 专家引导探索

- 以概率 `expert_prob` 使用专家策略替代 ε-greedy
- 概率线性衰减：`expert_prob = start + (final - start) · clip(ep / decay, 0, 1)`
- 根据最近成功率自适应调整

### 5.4 训练时动作掩码（Action Shield in Training）

训练时的动作掩码与推理不同，没有 fallback 逻辑，而是直接硬掩码。分三层生效：

**① 动作选择层**（`act_masked`，`cli/train.py` 中 `act_masked` 函数）：

- 当 `--forest-action-shield` 开启时，每步先算 `admissible_action_mask()`
- ε-greedy 随机探索：仅从可行动作中随机选（`np.nonzero(mask)`）
- 贪心选择：对不可行动作的 Q 值设 `-inf`，再 argmax

**② Replay Buffer 存储层**（`agents.py` 中 `observe` 方法）：

- 每条 transition 存储 `next_action_mask_1`（1-step 下一状态掩码）和 `next_action_mask_n`（n-step 下一状态掩码）
- 掩码来自 `env.admissible_action_mask()`，若全 False 则回退为全 True

**③ Q-target 计算层**（`agents.py` 中 `update` 方法）：

- `update()` 时从 batch 取出 `next_action_masks`
- 对 next state 的 Q 值做 `masked_fill(~mask, -inf)`
- DDQN：`argmax Q_online(s', masked)` → `Q_target(s', a*)`
- DQN：`max Q_target(s', masked)`
- 效果：不可行动作永远不会成为 TD 目标的 argmax 候选

### 5.5 Demo 收集（`collect_forest_demos`）

- 目标：收集 `target = 40 × learning_starts` 个专家转换
- 分桶采样：距离范围分 3 桶，每桶均匀收集
- 过滤：仅保留成功 episode，且满足 `progress_ratio ≥ 0.05`

## 6. 推理管道

**文件**：`forest_vehicle_dqn/cli/infer.py`

### 6.1 推理时 Fallback 策略（默认模式）

与训练时的硬掩码不同，推理采用分层 fallback，尽量选可行动作但不强制：

```text
a0 = argmax Q(s, a)
if a0 可行 → 执行 a0
elif top-k 中有可行动作 → 执行第一个可行的（k 由 --forest-topk 控制）
elif 全量 admissible_action_mask 中有可行的 → 执行 Q 值最高的可行动作
else → 执行 a0（允许失败，所有动作均不可行时的兜底）
```

**对比训练**：训练时直接用 mask 后的 ε-greedy，没有 top-k fallback 层。

### 6.1.1 Strict No-Fallback 模式（`--forest-no-fallback`）

- 纯 `argmax Q(s, a)` 推理，不做任何可行性检查或替换
- 仍可搭配 `--forest-action-shield` 做硬掩码（此时等同训练时行为）
- 用途：评估 Q 网络本身学到的安全性，不依赖推理时 shield

### 6.2 评测套件

| 套件 | 距离范围 | 用途 |
| ---- | -------- | ---- |
| short | 6-14m | 短距离性能 |
| long | 18m+ | 长距离性能（最终门槛） |

### 6.3 评测指标

- 成功率（SR）、路径长度（m）、路径时间（s）、曲率、计算时间
- 基线对比：Hybrid A*-MPC

## 7. 配置系统

**文件**：`forest_vehicle_dqn/config_io.py`、`configs/` 目录

- `--profile <name>`：从 `configs/<name>.json` 加载完整配置
- JSON 配置覆盖 argparse 默认值，CLI 参数再覆盖 JSON
- 训练和推理共用同一 profile，分 `train` / `infer` 两个顶层 key

## 8. 工具模块

| 模块 | 文件 | 功能 |
| ---- | ---- | ---- |
| Dijkstra 距离场 | `env.py: dijkstra_goal_dist_m` | 目标到所有自由格的最短路径距离 |
| Dijkstra 最短路径 | `env.py: dijkstra_shortest_path` | 8-连通网格最短路径坐标序列 |
| 网格 A* | `baselines/pathplan.py` | 全局路径规划 |
| Hybrid A* | `baselines/pathplan.py` | 考虑 Ackermann 约束的路径规划 |
| RRT* | `baselines/pathplan.py` | 采样式路径规划 |
| MPC 局部规划 | `baselines/mpc_local_planner.py` | 采样 MPC，网格搜索控制输入 |
| ε 调度器 | `schedules.py` | 线性/自适应 ε 衰减 |

## 9. 可改进模块一览（与性能瓶颈的关联）

| 模块 | 当前实现 | 潜在改进方向 | 预期影响 |
| ---- | -------- | ----------- | -------- |
| 动作空间 | 35 个离散动作 | 连续动作（SAC） | 降低曲率，缩短路径 |
| 观测空间 | 10 标量 + 12×12 地图 | 增加 Dijkstra 距离场通道 | 提供全局路径引导 |
| 奖励塑形 | 欧氏 progress + 轻量惩罚 | 势函数塑形 / 内在奖励 | 更稳定的学习信号 |
| 网络容量 | hidden=256, layers=3 | 增大 / 加注意力机制 | 更强表达能力 |
| 训练量 | 300 episodes | 500-1000 episodes | 更充分收敛 |
| 课程学习 | short 35% / long 65% | 自适应比例 | 针对性提升弱项 |
| Demo 质量 | Hybrid A* 前瞻跟踪 | Hybrid A*-MPC（更平滑） | 更好的模仿目标 |

## 10. 文件索引

```text
RL_sac/
├── train.py                              # 训练入口（CLI wrapper）
├── infer.py                              # 推理入口（CLI wrapper）
├── configs/                              # Profile JSON 配置
│   ├── config.json                       # 默认配置
│   ├── v7p1.json                         # 稳定基线
│   ├── v9p5.json                         # 最新版本
│   └── v9p4.json                         # Dijkstra 偏离（失败）
├── forest_vehicle_dqn/
│   ├── networks.py                       # MLPQNetwork / CNNQNetwork
│   ├── agents.py                         # DQNFamilyAgent
│   ├── replay_buffer.py                  # ReplayBuffer (PER/Stratified)
│   ├── env.py                            # AMRGridEnv / AMRBicycleEnv
│   ├── config_io.py                      # 配置加载与解析
│   ├── schedules.py                      # ε 衰减调度器
│   ├── metrics.py                        # KPI 计算（成功率、路径长度等）
│   ├── smoothing.py                      # 路径平滑
│   ├── runs.py                           # 实验目录管理
│   ├── runtime.py                        # 运行时配置
│   ├── live_view_pygame.py               # 训练实时可视化
│   ├── plr_curriculum.py                 # PLR 课程学习（实验性）
│   ├── sac_networks.py                   # SAC 网络（实验性）
│   ├── sac_agent.py                      # SAC Agent（实验性）
│   ├── maps/
│   │   └── forest.py                     # 森林地图定义
│   ├── cli/
│   │   ├── train.py                      # 训练主逻辑
│   │   ├── infer.py                      # 推理主逻辑
│   │   ├── benchmark.py                  # 基准测试
│   │   ├── game.py                       # 交互式游戏模式
│   │   └── precompute_forest_paths.py    # 预计算森林路径
│   ├── baselines/
│   │   ├── pathplan.py                   # A* / Hybrid A* / RRT*
│   │   └── mpc_local_planner.py          # 采样 MPC
│   └── third_party/
│       └── pathplan/                     # 第三方路径规划库（Hybrid A* / RRT* 实现）
└── docs/
    └── versions/                         # 各版本留档
```
