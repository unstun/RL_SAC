# 消融实验归档 (Ablation Studies Archive)

> 本文件夹汇总所有消融实验的设计、结果与结论。
> 详细数据见各子文件，原始版本留档见 `docs/versions/v11/`、`docs/versions/v13/`。

## 实验一览

| 实验 | 版本 | 变量 | 配置数 | 最优配置 | 详情 |
| --- | --- | --- | --- | --- | --- |
| 模块消融 | V11 | CBAM/MHA/Dueling/NoisyNet/QR-DQN | 7 | Config C (Dueling+MHA) | [module_ablation.md](module_ablation.md) |
| Horizon 消融 | V13 | `forest_adm_horizon` = 25/20/15/10 | 4+基线 | H30 (综合最优) | [horizon_ablation.md](horizon_ablation.md) |
| Mask 衰减消融 | V14 | `forest_adm_persistence` = 5/10 | 5+基线 | **负面结果**，无衰减最优 | [mask_decay_ablation.md](mask_decay_ablation.md) |

## 固定条件

所有消融实验共享以下固定条件：

- **基础架构**: CNN-DDQN + DQfD (profile=v9p2)
- **训练**: episodes=150, GPU 串行
- **评测地图**: `forest_a`，随机起终点 (`--rand-two-suites`)
- **对照基线**: Hybrid A\*-MPC (SR=1.0/1.0, path=16.87/43.02m, time=10.0/22.82s)

---

## 综合结论（最重要）

### 1. 模块消融核心发现

**模块贡献排序**（按 SR 影响）：

| 排序 | 模块 | 结论 | 关键证据 |
| --- | --- | --- | --- |
| 1 | **MHA** (Spatial Multi-Head Attention) | **必须保留** | 移除后 long SR 0.8→0.2 |
| 2 | **Dueling** | 保留无害，作用不明确 | 移除后 long SR 反而 0.8→1.0 |
| 3 | **CBAM** | **有负面作用，禁用** | 移除后 SR 0.80→0.93（全面提升） |
| 4 | **NoisyNet** | **有害，禁用** | 加入后 long SR 0.8→0.0 |
| 5 | **QR-DQN(32)** | **有害，禁用** | 加入后全面下降；150ep 不够收敛 |

**最优配置**: **Config C = Dueling + MHA**（不含 CBAM）
- Smoke SR=0.93 (1.0/0.8/1.0)，全配置最高
- 正式评测 (runs=20): SR=0.95/0.95
- 相比 baseline v9p2 (SR=0.33) 提升 +188%

### 2. Horizon 消融核心发现

| 发现 | 说明 |
| --- | --- |
| **非单调行为** | H25/H20 比 H30 全面恶化，H15 反而最优 short SR=1.0 |
| **掩码是路径过长根因** | H10 成功时 path=15.40m < A\*-MPC 16.87m，time=9.57s < 10.0s |
| **SR 与路径不可兼得** | 缩短 horizon 能缩短路径，但严重损害 long SR |
| **H30 仍为综合最优** | SR=0.95/0.95，其他 horizon 均无法同时满足 SR 和路径 |

### 3. Mask 衰减消融核心发现 (V14, 负面结果)

| 发现 | 说明 |
| --- | --- |
| **衰减 rollout 全面劣于无衰减** | 所有衰减配置 (P5/P10) 的 long SR 均从 1.0 降至 0.8 或更低 |
| **rollout 轨迹不真实是根因** | 真实策略既不恒定也不线性衰减，衰减假设与实际行为不匹配 |
| **端到端训练更差** | P5-H30 训练+推理: short SR=0.6, long SR=0.2 |
| **Horizon 解耦 SR 更高但路径更长** | H15→H30: SR 1.0/1.0 但路径 +18%；H30→H30 路径更短，综合更优 |

### 4. 跨实验结论

1. **MHA 是最关键增强模块**——未来迭代必须保留
2. **CBAM/NoisyNet/QR-DQN 禁止默认启用**——已验证有负面作用
3. **Horizon 参数调优不能解决路径过长问题**——需要结构性改进
4. **掩码机制（`forest_adm_horizon`）确实是路径长的根因**——H10 已证明
5. **Mask 衰减不是解决方案**——V14 已排除，衰减假设与真实策略行为不匹配
6. **训练-推理 horizon 解耦提升 SR 但路径更长**——H15→H30: SR 1.0/1.0 但 path +18%，H30→H30 综合更优
7. **下一步方向**：以 H30→H30 为基线，通过 reward shaping / 增加训练量 / 连续动作空间 (SAC) 缩短路径

### 5. 最终门槛状态

**未通过 CLAUDE.md §13 门槛**。最优配置 (Dueling+MHA, H30) 正式评测：
- SR: 0.95 ≈ A\*-MPC (short), 0.95 < 1.0 (long) — ❌
- Path: 16.41m > 15.77m (short), 47.34m > 42.93m (long) — ❌
- Time: 10.54s > 9.25s (short), 26.61s > 22.76s (long) — ❌

**积极面**：计算时间快 3.5-5.9x，SR 相比 baseline 提升 188%。
