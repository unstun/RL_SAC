# v8p2: CBF-Safe SAC

> 日期：2026-02-21
> 基线版本：v8p1（SAC-GlobalCNN 稳定性修复，SR=0%）
> 目标：通过 CBF 安全过滤 + CBF-aware 奖励塑形 + alpha 衰减修复，突破 SR=0%

## 方法摘要

在 v8p1 基础上新增三个模块：

1. **Discrete-time CBF Safety Filter**：在 `env.py` 新增 `cbf_safe_action()` 方法，基于 barrier function `h(x) = min_od - safety_margin` 实现离散时间 CBF 约束投影。训练和推理时均应用。
2. **CBF-aware 奖励塑形**：用 potential-based progress reward (`c_prog * (dist_k - γ * dist_{k+1})`) 替代旧 `k_p`；用 log-barrier reward (`c_cbf * log(h/h_max)`) 替代旧 `k_o`。
3. **Alpha 衰减修复**：`lr_alpha` 从 3e-4 降至 1e-4，`target_entropy` 从 -2.0 提高至 -1.0。

## 关键配置

| 参数 | v8p1 | v8p2 |
|------|------|------|
| `sac_lr_alpha` | 3e-4 | 1e-4 |
| `sac_target_entropy` | -2.0 | -1.0 |
| `forest_reward_k_p` | 8.0 | 0.0 |
| `forest_reward_k_o` | (默认1.5) | 0.0 |
| `forest_cbf_alpha` | N/A | 0.3 |
| `forest_cbf_safety_margin_m` | N/A | 0.15 |
| `forest_reward_c_prog` | N/A | 5.0 |
| `forest_reward_c_cbf` | N/A | 1.0 |

## 关键命令

```bash
# 训练
conda run -n ros2py310 python train.py --profile v8p2 --episodes 150 --out v8p2-smoke1 --no-progress
# 推理
conda run -n ros2py310 python infer.py --profile v8p2 --runs 3 --models runs/v8p2-smoke1/train_20260221_224028 --out v8p2-smoke1 --no-progress
```

## 代表 Run

- 训练：`runs/v8p2-smoke1/train_20260221_224028`
- 推理：`runs/v8p2-smoke1/train_20260221_224028/infer/20260221_224714`

## Smoke 结果（150ep + runs=3）

| Suite | CNN-SAC SR | Hybrid A*-MPC SR |
|-------|-----------|-----------------|
| short | 0% | 100% |
| mid | 0% | 100% |
| long | 0% | 100% |

- 训练 best_return=477.5（正值，说明训练中有 episode 到达目标）
- CBF 总介入次数：913（150ep）
- Alpha 衰减：1.0 → 0.047（比 v8p1 慢，符合预期）

## 结论

SR 仍为 0%。150ep smoke 训练不足以让 SAC 策略泛化到推理。但训练中出现正回报（best=477.5），CBF filter 正常工作（913 次介入），alpha 衰减速度已改善。

## 下一步

1. 延长训练至 1000-3000ep 观察 SR 是否突破 0%
2. 若仍为 0%，考虑 v8p3 课程学习（curriculum learning）
3. 分析 CBF 介入频率随训练进展的变化趋势
