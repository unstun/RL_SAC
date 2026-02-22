# v8p4: v8p3 结构性缺陷修复（alpha_min + PBRS + budget + log keys）

## 版本目标

修复 v8p3 的四个结构性缺陷，1000ep smoke 验证：

1. **alpha_min 下界**：新增 `alpha_min=0.01`，防止 alpha 崩塌到 0.003
2. **PBRS 公式修正**：γ 应在新状态上（`γΦ(s') - Φ(s)`），原代码反了
3. **指数 potential base 降低**：`base=4`（原 32），避免 reward 膨胀
4. **entropy_budget_ratio 提高**：`ρ=2.0`（原 0.6），给 TECRL 更多探索预算
5. **日志 key 对齐**：TECRL 返回 dict 补充 `target_q_mean`/`target_q_max`

## 方法摘要

- 在 v8p3 基础上做最小修复，不引入新功能
- `SACConfig` 新增 `alpha_min` 字段，alpha 更新后 clamp `log_alpha`
- 标准 SAC 和 TECRL 两处 `alpha_opt.step()` 后均加 clamp

## 关键命令

```bash
# 训练
conda run -n ros2py310 python train.py --profile v8p4 --episodes 1000 --out v8p4-smoke1
# 推理
conda run -n ros2py310 python infer.py --profile v8p4 --runs 3 \
  --models runs/v8p4-smoke1/train_20260222_090846 --out v8p4-smoke1
```

## 代表 Run

- 训练：`runs/v8p4-smoke1/train_20260222_090846/`
- 推理：`runs/v8p4-smoke1/train_20260222_090846/infer/20260222_101831/`

## 结论

**Smoke 结果：SR=0%（short/mid/long 全部失败）**，未通过 smoke 门槛。

改善点：

1. **alpha_min 生效**：alpha 从 1.0 → 0.010 后稳定（v8p3 崩塌到 0.003）
2. **tq 非零**：日志 key 对齐后 tq 正常显示（-1.2 ~ -1.4）
3. **best_return=-230.6**（v8p3 为 N/A 因 reward 膨胀失真）

未解决问题：

- 模型仍未学会到达目标（SR=0%），1000ep 训练不足以收敛
- eval_return 波动大（-282 ~ -2077），策略不稳定

## 下一步

- 增加训练 episodes（3000+）或调整学习率
- 考虑更激进的课程策略（先只训短距离）
- 检查 reward shaping 信号是否足够引导策略向目标移动
