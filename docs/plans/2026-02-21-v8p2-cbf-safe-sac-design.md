# v8p2 设计文档：CBF-Safe SAC

> 日期：2026-02-21
> 基线版本：v8p1（稳定性修复，SR=0%，回报稳定 -900→-400）
> 目标：突破 SR=0%，为后续课程学习（v8p3）奠定基础

## 1. 问题诊断

v8p1 解决了训练发散问题，但 SR 仍为 0%。根因：

1. **连续空间探索困难**：随机探索几乎不可能碰到目标
2. **无安全过滤**：离散 DQN 有 admissible-action masking，SAC 没有
3. **Alpha 衰减过快**：50ep 内从 1.0→0.07，过早杀死探索
4. **奖励信号不够密集**：虽有 progress reward，但非严格 potential-based

## 2. 方案概述

v8p2 包含三个模块（课程学习延后到 v8p3）：

| 模块     | 方法                              | 参考文献                                                      |
| -------- | --------------------------------- | ------------------------------------------------------------- |
| 安全过滤 | Discrete-time CBF Safety Filter   | CBF-RL (Yang et al., 2025), SAC-CBF (arXiv:2503.08479, 2025) |
| 奖励塑形 | CBF-aware potential-based shaping | Heuristic dense reward shaping (ISA Trans, 2024)              |
| 探索修复 | Alpha 学习率 + 目标熵调整         | v8p1 实验发现                                                 |

## 3. 模块 1：Discrete-time CBF Safety Filter

### 3.1 Barrier Function 定义

```python
h(x) = min_od(x, y, ψ) - safety_margin
```

- `min_od(x, y, ψ)`：两圆车身模型（`TwoCircleFootprint`）在 EDT 距离场上的最小距离减去碰撞阈值
- `safety_margin = 0.15m`（约 1/3 车身半径 0.424m）
- `h(x) >= 0` 表示安全状态

### 3.2 离散时间 CBF 约束

每步检查：

```text
h(x_{k+1}) >= (1 - α_cbf) * h(x_k)
```

- `α_cbf = 0.3`：允许 barrier 值每步最多衰减 30%
- 当 `h(x_k)` 较大时约束宽松（远离障碍物可自由行动）
- 当 `h(x_k)` 接近 0 时约束严格（接近障碍物必须远离）

### 3.3 投影机制

当 SAC 输出 `u_raw = (δ̇, a)` 违反 CBF 约束时：

1. 用 `bicycle_integrate_one_step` 模拟一步得到 `x_{k+1}(u_raw)`
2. 计算 `violation = (1 - α_cbf) * h(x_k) - h(x_{k+1}(u_raw))`
3. 若 `violation > 0`（不安全）：
   a. 数值梯度：`∇_u h ≈ [h(u+ε) - h(u-ε)] / 2ε`，对 δ̇ 和 a 各一次（4 次额外 rollout）
   b. 线性化投影：`u_safe = u_raw + (violation / ||∇_u h||²) * ∇_u h`
   c. Clamp 到 [-1, 1]²
4. 若投影后仍不安全：fallback 到 `(0, -1)`（全力刹车）

### 3.4 训练 vs 推理

- **训练时**：CBF filter 应用于环境交互，replay buffer 存 `(s, u_safe, r, s', done)`
- **推理时**：同样应用 CBF filter，确保安全
- SAC 策略梯度基于 buffer 中的 `u_safe`，策略逐渐学会输出安全动作

### 3.5 实现位置

- `env.py`：新增 `cbf_safe_action(delta_dot_raw, a_raw, alpha_cbf, safety_margin)` 方法
- 复用现有 `_od_and_collision_at_pose_m()`、`bicycle_integrate_one_step()`

## 4. 模块 2：CBF-aware 奖励塑形

### 4.1 距离进展奖励（替代旧 k_p）

```python
r_progress = c_prog * (dist_k - γ * dist_{k+1})
```

- `c_prog = 5.0`
- 严格 potential-based（Φ(s) = -c * dist_to_goal），保证不改变最优策略
- 替代现有 `k_p * (dist_before - dist_after)`（非 potential-based）

### 4.2 CBF 安全奖励（新增）

```python
r_cbf = c_cbf * log(max(h(x) / h_max, eps))
```

- `c_cbf = 1.0`，`h_max = 2.0`（归一化），`eps = 1e-6`
- 当 `h(x)` 接近 0（危险）时 `r_cbf → -∞`（log-barrier 效应）
- 当 `h(x)` 较大（安全）时 `r_cbf` 接近 0（不干扰正常学习）
- 与 CBF filter 的约束方向一致：agent 学到的安全行为和 filter 的投影方向对齐

### 4.3 保留的现有奖励项

- 时间惩罚 `k_t`、转向平滑 `k_delta`、加速平滑 `k_a`、曲率 `k_kappa`
- 路径长度 `k_len`、速度-间距耦合 `k_v`
- 终端奖励/惩罚（到达/碰撞/卡住）

### 4.4 移除/替换的奖励项

- `k_p`（progress）→ 替换为 `r_progress`（potential-based 版本）
- `k_o`（obstacle proximity `1/od`）→ 替换为 `r_cbf`（log-barrier 版本）

## 5. 模块 3：Alpha 衰减修复

| 参数             | v8p1 | v8p2 | 理由                             |
| ---------------- | ---- | ---- | -------------------------------- |
| `lr_alpha`       | 3e-4 | 1e-4 | 降低 alpha 学习率，防止过快衰减  |
| `target_entropy` | -2.0 | -1.0 | 提高目标熵，保持更多探索         |

## 6. 配置变更汇总

```json
{
  "alpha_cbf": 0.3,
  "safety_margin_m": 0.15,
  "c_prog": 5.0,
  "c_cbf": 1.0,
  "h_max": 2.0,
  "lr_alpha": 1e-4,
  "target_entropy": -1.0,
  "reward_k_p": 0.0,
  "reward_k_o": 0.0
}
```

## 7. 验证计划

### 7.1 Smoke（150ep + runs=3）

- 训练 150ep，观察：
  - 回报趋势是否稳定（不发散）
  - CBF 介入频率（期望：早期高，后期低）
  - Alpha 衰减速度（期望：比 v8p1 慢）
  - SR 是否突破 0%
- 推理 runs=3（short/mid/long）

### 7.2 成功标准

- **最低门槛**：SR > 0%（至少有 1 次到达目标）
- **进入 v8p3 门槛**：short SR >= 20%
- **若 SR 仍为 0%**：归档为失败版本，分析 CBF 介入日志决定下一步

## 8. 风险与缓解

| 风险                                     | 影响         | 缓解                                    |
| ---------------------------------------- | ------------ | --------------------------------------- |
| CBF 投影过于保守，agent 无法前进         | SR=0%        | 调低 α_cbf 或 safety_margin             |
| 数值梯度不准确导致投影失败               | 碰撞率不降   | 增加 ε 或改用多步 rollout               |
| Potential-based shaping 改变了奖励尺度   | 训练不稳定   | 调整 c_prog 和 reward_scale             |
| Alpha 修复后探索过多，Q 值不稳定         | 训练发散     | 保留 v8p1 的 Q clamp 和 Huber loss      |

## 9. 参考文献

- [CBF-RL: Safety Filtering RL with Control Barrier Functions](https://arxiv.org/abs/2510.14959) (Yang et al., 2025)
- [SAC-based CBF Adaptation for Robust Navigation](https://arxiv.org/html/2503.08479v1) (arXiv, 2025)
- [FCSRL: Feasibility Consistent Representation Learning for Safe RL](https://github.com/czp16/FCSRL) (ICML 2024)
- [ActSafe: Active Exploration with Safety Constraints](https://openreview.net/forum?id=aKRADWBJ1I) (ICLR 2025)
- [Heuristic dense reward shaping for map-free navigation](https://www.sciencedirect.com/science/article/abs/pii/S0019057824005032) (ISA Trans, 2024)
