# v12 RESULTS — Dueling+MHA 正式评测

## 正式评测 (runs=20, short/long 双套件)

配置: **Dueling+MHA**（V11 消融最优 Config C）
训练 run: `v11-abl-duel-mha` (episodes=150)
推理 run: `v11-formal-duel-mha/20260224_160845`
KPI: `runs/v11-formal-duel-mha/20260224_160845/table2_kpis_mean.csv`

### Success Rate

| Suite | CNN-DDQN | Hybrid A\*-MPC | 差异 |
| --- | --- | --- | --- |
| short | 0.95 | 0.95 | = |
| long | 0.95 | 1.0 | -5% |

### Average Path Length (m, 仅成功 runs)

| Suite | CNN-DDQN | Hybrid A\*-MPC | 差异 |
| --- | --- | --- | --- |
| short | 16.41 | 15.77 | +4.1% |
| long | 47.34 | 42.93 | +10.3% |

### Path Time (s, 仅成功 runs)

| Suite | CNN-DDQN | Hybrid A\*-MPC | 差异 |
| --- | --- | --- | --- |
| short | 10.54 | 9.25 | +13.9% |
| long | 26.61 | 22.76 | +16.9% |

### Compute Time (s)

| Suite | CNN-DDQN | Hybrid A\*-MPC | 加速比 |
| --- | --- | --- | --- |
| short | 0.142 | 0.504 | **3.5x** |
| long | 0.328 | 1.944 | **5.9x** |

### 其他指标

| Suite | Curvature (1/m) | Max Corner (°) | Inadmissible Rate | Fallback Rate |
| --- | --- | --- | --- | --- |
| short | 0.142 | 29 | 0.142 | 0.0 |
| long | 0.158 | 29 | 0.156 | 0.0 |

## 门槛检查 (CLAUDE.md §13)

| 条件 | short | long | 通过? |
| --- | --- | --- | --- |
| SR(CNN) >= SR(A\*-MPC) | 0.95 = 0.95 ✅ | 0.95 < 1.0 ❌ | ❌ |
| path_length(CNN) < path_length(A\*-MPC) | 16.41 > 15.77 ❌ | 47.34 > 42.93 ❌ | ❌ |
| path_time(CNN) < path_time(A\*-MPC) | 10.54 > 9.25 ❌ | 26.61 > 22.76 ❌ | ❌ |

**最终门槛状态：未通过。**

## 失败分析

- short: SR 持平但路径长 4.1%、时间慢 13.9%
- long: SR 差 5%，路径长 10.3%、时间慢 16.9%
- 主要瓶颈：RL 在靠近障碍物时过度避让（大转弯），因 horizon=30 步常值动作掩码过于保守

## 积极面

- SR=0.95 相比 baseline v9p2 (SR=0.33) 提升 +188%
- 计算时间 CNN-DDQN 快 3.5-5.9x（无需 A\* 搜索+MPC 跟踪）
- 路径长度仅长 4-10%，路径时间仅慢 14-17%

---

## Horizon 参数实验（附加）

### 实验动机

RL 在障碍物附近大幅转弯的根因是 `forest_adm_horizon=30`（常值动作前瞻 1.5s），
对于 0.05s 决策频率的反应式智能体过于保守。尝试缩短 horizon 放宽掩码。

### 结果

| Horizon | 训练 run | 推理 SR | 备注 |
| --- | --- | --- | --- |
| 30 (default) | v11-abl-duel-mha | 0.95/0.95 | 正式配置 |
| 15 | v12-horizon15 | 0.0 | SR 崩塌 |
| 10 | v12-horizon10 | 0.2 | SR 崩塌 |

### 失败原因

DQfD 专家 demo 在 horizon=30 下生成，掩码与专家策略紧密耦合。
缩短 horizon 使训练时更多动作被标为可行，但 Q 网络学到的价值函数
与 horizon=30 专家不一致，导致 argmax_inadmissible_rate 从 14% 飙升到 46-50%。

### 结论

不可直接修改 horizon 参数。如需缩短 horizon，必须：

1. 用新 horizon 重新生成 DQfD 专家 demo
2. 或增加训练量至 500-1000ep 让 Q 网络适应
3. 或转向不依赖离散掩码的连续动作空间（如 SAC）
