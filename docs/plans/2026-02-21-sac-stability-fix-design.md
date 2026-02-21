# SAC 稳定性修复设计（方案 C）

日期：2026-02-21
状态：已批准，待实施

## 背景

v8 SAC-GlobalCNN 训练发散（1000ep，SR=0%）。根因：`no_terminate_on_stuck=true` 导致
stuck 惩罚 -300/步 × 1000+ 步 = -300K 回报，Q 值爆炸。

## 修复分组

### 第一组：关键修复
1. **恢复 terminate_on_stuck=true** — 删除 v8.json 中 `no_terminate_on_stuck`
2. **修复 flow_log bug** — train_one_sac 传入 flow_log_fp

### 第二组：Q 值稳定性
3. **奖励裁剪** — observe() 中 clip reward 到 [-500, +1100]
4. **Target Q 裁剪** — update() 中 clamp target_q 到 [-50, +50]
5. **Alpha 梯度裁剪** — alpha 更新加 grad_clip_norm
6. **Huber Loss** — critic 用 smooth_l1_loss 替代 mse_loss

### 第三组：训练质量
7. **Gamma 0.99→0.98** — 缩短有效视野，降低 Q 值量级
8. **Critic Warmup 2000 步** — BC 后先预热 critic 再启动 actor
9. **周期确定性评估** — 每 50ep 跑 1 个 eval episode

### 第四组：监控
10. **Q 值统计** — update() 返回 q1_mean, target_q_mean 等
11. **CSV 增强** — training_returns.csv 增加 loss/alpha/Q 列

## 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| sac_reward_clip_min | -500.0 | 奖励下界 |
| sac_reward_clip_max | 1100.0 | 奖励上界 |
| sac_target_q_min | -50.0 | Target Q 下界 |
| sac_target_q_max | 50.0 | Target Q 上界 |
| sac_use_huber_loss | true | Critic 用 Huber loss |
| sac_critic_warmup_steps | 2000 | Critic 预热步数 |
| sac_eval_interval | 50 | 评估间隔（episode） |

## 文件改动

- `configs/v8.json` — 删 2 行 + 改 1 行 + 加 7 行
- `forest_vehicle_dqn/sac_agent.py` — ~50 行
- `forest_vehicle_dqn/cli/train.py` — ~40 行

## 验证

- Smoke: episodes=150, infer runs=3
- 成功标准: return 不发散 + SR > 0%
