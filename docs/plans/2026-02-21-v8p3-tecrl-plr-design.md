# v8p3 设计文档：TECRL + 改进 Potential-Based Shaping + Syllabus PLR

> 日期：2026-02-21
> 基线版本：v8p2（CBF-Safe SAC，smoke SR=0%，best_return=477.5）
> 目标：通过三个正交改进突破 SR=0%

## 1. 问题诊断

v8→v8p1→v8p2 链路中已确认三个根因：

| # | 根因 | 证据 | v8p2 现状 |
|---|------|------|----------|
| 1 | alpha 下降过快 → 探索过早停止 | v8p1: 50ep 内 1.0→0.07 | lr_alpha=1e-4 缓解但未根治 |
| 2 | 奖励信号不够密集 | 训练中偶尔正回报但推理 SR=0% | potential-based + CBF log-barrier |
| 3 | 训练任务难度不匹配 | 随机 short/long 混合，早期 long 全失败 | 固定比例课程 |

## 2. 方案概述

### 2.1 TECRL：Reward-Entropy 分离（解决根因 1）

**论文依据**：Mind Your Entropy (2025, OpenReview)

**核心思路**：标准 SAC 用单一 Q 函数同时编码 reward 和 entropy bonus，导致 alpha 更新时 Q-target 非平稳。TECRL 分离为两套 critic：

- `Q_r(s,a)`：纯 reward critic，不含 alpha 项
- `Q_e(s,a)`：entropy critic，估计累积熵 `E[Σ γ^t (-log π(a_t|s_t))]`

alpha 不再出现在任何 critic 的 target 中，而是通过轨迹级熵约束调节：

```
α_loss = -α * (Q_e(s,a) + log π(a|s) - H_budget)
H_budget = ρ * H_0 / (1 - γ)    # ρ ∈ (0,1) 控制熵预算比例
```

**对 SACAgent 的改动**：
- 新增 `SACEntropyCritic`（结构与 `SACCritic` 相同，但 target 不含 alpha）
- `update()` 分为三步：(1) reward critic update, (2) entropy critic update, (3) actor update with combined objective
- alpha update 使用轨迹级约束而非单步 target_entropy
- 新增超参：`entropy_budget_ratio`（ρ，推荐 0.5-0.8）

### 2.2 改进 Potential-Based Reward Shaping（解决根因 2）

**论文依据**：Improving the Effectiveness of Potential-Based Reward Shaping (AAMAS 2025)

**核心改动**（在 v8p2 的 `reward_c_prog` 基础上）：

1. **偏置修正**：`Φ_b(s) = Φ(s) + b/(γ-1)`，其中 `b = (1-γ)*Q_init - r_∞`
   - 消除 Q 初始化与 shaped reward 之间的 mismatch
   - 对于我们的场景：`Q_init ≈ 0`（网络初始化），`r_∞ ≈ -step_penalty`

2. **指数 potential**：`Φ(s) = e^(base * normalized_progress)` 替代线性 `dist_k - γ*dist_{k+1}`
   - `base=32`（论文推荐），减少小进展时的错误 shaping 信号
   - `normalized_progress = 1 - dist_to_goal / initial_dist`

3. **保留 CBF log-barrier reward**：v8p2 的 `reward_c_cbf * log(h/h_max)` 不变

**对 env.py 的改动**：
- `_compute_reward()` 中 progress 分量替换为指数 potential + 偏置
- 新增参数：`reward_potential_base`（指数底数）、`reward_potential_bias`（偏置项）
- 向后兼容：`reward_potential_base=0` 时退回 v8p2 线性 potential

### 2.3 Syllabus PLR 课程学习（解决根因 3）

**论文依据**：Syllabus (RLC 2025 Best Tooling Award)，Prioritized Level Replay

**核心思路**：PLR 维护一个"难度等级"分布，按每个等级的 learning potential（TD-error 或 GAE magnitude）优先采样。高学习价值的等级被更频繁地训练。

**集成方案**：
- 将 start-goal 距离离散化为 N 个等级（如 [6m, 10m, 14m, 20m, 30m, 42m]）
- 每个等级对应一个 Syllabus "level"
- 训练循环中：Syllabus 选择下一个 level → 设置对应 `rand_min_dist_m` / `rand_max_dist_m` → 训练 → 回报 episode score 给 Syllabus
- Syllabus 自动调整各等级采样概率

**对 train.py 的改动**：
- 新增 `SyllabusPLRCurriculum` wrapper 类（约 60 行）
- `train_one_sac()` 新增参数：`use_syllabus_plr`、`plr_levels`、`plr_staleness_coef`
- 替换现有 `forest_curriculum` / `forest_train_two_suites` 逻辑（当 `use_syllabus_plr=True` 时）
- 新增依赖：`pip install syllabus-rl`

## 3. 配置变更（v8p2 → v8p3）

| 参数 | v8p2 | v8p3 | 说明 |
|------|------|------|------|
| `sac_use_tecrl` | N/A | true | 启用 TECRL 双 critic |
| `sac_entropy_budget_ratio` | N/A | 0.6 | 轨迹熵预算比例 ρ |
| `sac_lr_entropy_critic` | N/A | 3e-4 | entropy critic 学习率 |
| `reward_potential_base` | N/A | 32 | 指数 potential 底数 |
| `reward_potential_bias` | N/A | auto | 自动计算偏置 |
| `use_syllabus_plr` | N/A | true | 启用 Syllabus PLR |
| `plr_levels` | N/A | [6,10,14,20,30,42] | 距离等级（米） |
| `plr_staleness_coef` | N/A | 0.3 | PLR 陈旧度系数 |
| `forest_curriculum` | true | false | 禁用旧课程（PLR 接管） |
| `forest_train_two_suites` | true | false | 禁用旧双套件（PLR 接管） |

## 4. 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `forest_vehicle_dqn/sac_agent.py` | 重构 | 新增 entropy critic + TECRL update 逻辑 |
| `forest_vehicle_dqn/sac_networks.py` | 新增 | `SACEntropyCritic` 网络定义 |
| `forest_vehicle_dqn/env.py` | 修改 | 指数 potential + 偏置修正 |
| `forest_vehicle_dqn/cli/train.py` | 修改 | Syllabus PLR 集成 + 新 CLI 参数 |
| `forest_vehicle_dqn/plr_curriculum.py` | 新增 | Syllabus PLR wrapper |
| `configs/v8p3.json` | 新增 | v8p3 配置 |
| `tests/test_tecrl.py` | 新增 | TECRL 单元测试 |
| `tests/test_plr_curriculum.py` | 新增 | PLR 课程单元测试 |

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| TECRL 双 critic 显存翻倍 | 中 | 共享 CNN encoder（reward/entropy critic 共用 encoder，只分离 MLP head） |
| Syllabus 与现有训练循环集成困难 | 中 | 先写 adapter 层，保持现有 reset_options 接口不变 |
| 指数 potential 导致 reward 爆炸 | 低 | reward_clip 已有 [-500, 1100]，且 potential 归一化到 [0,1] |
| 三个改动叠加调试困难 | 高 | 分步验证：先 TECRL-only smoke → 加 potential → 加 PLR |

## 6. 验证计划

1. **self-check**：`python train.py --self-check` + `python infer.py --self-check` 通过
2. **单元测试**：`test_tecrl.py` + `test_plr_curriculum.py` 全绿
3. **smoke（150ep + runs=3）**：观察 SR 是否 > 0%，alpha 是否稳定
4. **full（3000ep + runs=20）**：short/long 双套件，对标 Hybrid A*-MPC

## 7. 参考文献

- [Mind Your Entropy (TECRL)](https://openreview.net/forum?id=tcbh5eKGPr) - reward-entropy 分离
- [Improving Potential-Based Reward Shaping](https://arxiv.org/html/2502.01307v1) - 偏置修正 + 指数 potential
- [Syllabus](https://github.com/RyanNavillus/Syllabus) - PLR 课程学习库
- [SASR (ICLR 2025)](https://github.com/mahaozhe/SASR) - 自适应奖励塑形（备选）
- [OmniSafe](https://github.com/PKU-Alignment/omnisafe) - Safe RL 参考实现
- [Meta SAC-Lag](https://arxiv.org/html/2408.07962v1) - Lagrangian 自动调参（备选）
- [Corrected SAC](https://arxiv.org/html/2410.16739v1) - tanh 修正（备选）
