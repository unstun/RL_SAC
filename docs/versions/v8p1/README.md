# v8p1: SAC 稳定性修复（方案 C）

## 版本目标

修复 v8 SAC-GlobalCNN 训练发散问题。v8 训练 1000ep 后 return 爆到 -211K，Q 值爆炸，SR=0%。

## 方法摘要

在 v8 基础上实施方案 C（全面稳定性修复），共 11 项改动分 4 组：

- **关键修复**: 恢复 `terminate_on_stuck=true`（消除 -300/步累积惩罚）、修复 flow_log bug
- **Q 值稳定性**: 奖励裁剪 [-500, +1100]、target Q 裁剪 [-50, +50]、alpha 梯度裁剪、Huber loss 替代 MSE
- **训练质量**: gamma 0.99→0.98、critic warmup 2000 步、周期确定性评估（每 50ep）
- **监控**: update() 返回 Q 值统计、增强训练 CSV（training_stats.csv）

## 关键命令

```bash
# 训练
conda run -n ros2py310 python train.py --profile v8 --episodes 3000 --out v8-fix1

# 推理
conda run -n ros2py310 python infer.py --profile v8 --runs 20 --models runs/v8-fix1/<train_dir>
```

## 代表 run

- **smoke 训练**: `runs/v8-fix1/train_20260221_212443`（150ep, ubuntu-zt RTX 5070 Ti, 7m28s）
- **smoke 推理**: `runs/v8/20260221_213357`（runs=3, short/mid/long）

## 结论

**稳定性修复成功**：训练 return 从发散（-2K→-29K→-211K）变为稳定（-900→-400），Q 值始终在 [0.5, 11] 范围内。但 150ep smoke SR=0%（预期，episode 数不足）。

**新发现**：alpha 下降过快（50ep 内 1.0→0.07），探索过早停止，可能影响长训练收敛。

## 下一步

1. 调整 alpha 衰减速度（降低 `lr_alpha` 或提高 `target_entropy`）
2. 跑 1000ep+ 长训练验证 SR 是否能突破 0%
3. 若 SR 仍为 0%，考虑更密集的奖励塑形（progress reward）
