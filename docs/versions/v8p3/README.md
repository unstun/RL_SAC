# v8p3: TECRL + 指数 Potential + Syllabus PLR

## 版本目标

在 v8p2（CBF-Safe SAC, SR=0%, best_return=477.5）基础上，通过三个正交改进突破 SR=0%：

1. **TECRL**（reward-entropy 分离）：分离 reward-critic 和 entropy-critic，防止 alpha 过快衰减
2. **指数 Potential-Based Reward Shaping**：`Φ(s) = exp(base * progress)` 替代线性 potential，提供更密集的接近目标奖励
3. **Syllabus PLR**（Prioritized Level Replay）：按 TD-error 优先采样高学习价值的难度等级

## 方法摘要

- TECRL：新增 `SACEntropyCritic`（twin Q 网络估计累积熵），actor loss = -Q_r - α*Q_e，alpha 通过轨迹级熵约束更新
- 指数 potential：`reward_potential_base=32`，`c_prog=5.0`，偏置修正 `bias=0.0`
- PLR：6 个难度等级 [6,10,14,20,30,42] 米，替代旧的 two-suites 和 linear curriculum
- 保留 v8p2 的 CBF safety filter（alpha=0.3）

## 关键命令

```bash
# 训练
conda run -n ros2py310 python train.py --profile v8p3 --episodes 150 --out v8p3-smoke1
# 推理
conda run -n ros2py310 python infer.py --profile v8p3 --runs 3 \
  --models runs/v8p3-smoke1/train_20260222_011340 --out v8p3-smoke1
```

## 代表 Run

- 训练：`runs/v8p3-smoke1/train_20260222_011340/`
- 推理：`runs/v8p3-smoke1/train_20260222_011340/infer/20260222_012756/`

## 结论

**Smoke 结果：SR=0%（short/mid/long 全部失败）**，未通过 smoke 门槛。

关键问题：
1. **Alpha 仍然崩塌**：从 1.008 → 0.003（150ep），TECRL 未能阻止 alpha 单调下降
2. **指数 potential base=32 过大**：ep1 的 raw return 膨胀到 4.8 亿（buffer 中 reward 已 clip，训练不受影响，但 best_return 追踪失真）
3. **日志 key 不匹配**：TECRL 返回 `target_q_r_mean`，日志读 `target_q_mean`，导致 tq 始终显示 0.00

## 下一步

- 降低 `reward_potential_base`（如 4~8），避免指数爆炸
- 调整 `entropy_budget_ratio`（当前 0.6 可能过松），或给 alpha 设下界（如 `alpha_min=0.01`）
- 修复日志 key 不匹配问题
- 考虑增加训练 episodes（150ep 可能不足以让 TECRL 收敛）
