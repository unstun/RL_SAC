# v8p1 改动明细（相对 v8）

## 改动文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `configs/v8.json` | 修改 | 删除 `no_terminate_on_stuck`（train+infer）、gamma 0.99→0.98、新增 7 个 SAC 参数 |
| `forest_vehicle_dqn/sac_agent.py` | 修改 | SACConfig 扩展 6 字段、observe() 奖励裁剪、update() Huber loss + target Q clamp + alpha grad clip + Q 统计、新增 warmup_critic() |
| `forest_vehicle_dqn/cli/train.py` | 修改 | flow_log_fp 透传修复、SACConfig 新字段透传、critic warmup 调用、周期确定性评估、增强 CSV、CLI 参数注册 |
| `docs/plans/2026-02-21-sac-stability-fix-design.md` | 新增 | 方案 C 设计文档 |

## 参数变更

| 参数 | v8 旧值 | v8p1 新值 |
|------|---------|-----------|
| `no_terminate_on_stuck` (train) | `true` | 删除（默认 false） |
| `no_terminate_on_stuck` (infer) | `true` | 删除（默认 false） |
| `gamma` | `0.99` | `0.98` |
| `sac_reward_clip_min` | 无 | `-500.0` |
| `sac_reward_clip_max` | 无 | `1100.0` |
| `sac_target_q_min` | 无 | `-50.0` |
| `sac_target_q_max` | 无 | `50.0` |
| `sac_use_huber_loss` | 无 | `true` |
| `sac_critic_warmup_steps` | 无 | `2000` |
| `sac_eval_interval` | 无 | `50` |

## 代码改动详情

### sac_agent.py
- `SACConfig`: 新增 `reward_clip_min/max`, `target_q_min/max`, `use_huber_loss`, `critic_warmup_steps`
- `observe()`: 存入 buffer 前 `np.clip(reward, clip_min, clip_max)`
- `update()`: target_q 计算后 `.clamp(min, max)`；critic loss 改为 `smooth_l1_loss`（可配置）；alpha 更新加 `clip_grad_norm_`；返回 dict 新增 `q1_mean`, `q2_mean`, `target_q_mean`, `target_q_max`, `reward_batch_mean`
- 新增 `update_critic_only()`: 仅更新 critic（跳过 actor/alpha）
- 新增 `warmup_critic(steps)`: 循环调用 `update_critic_only()`

### train.py (train_one_sac)
- 函数签名新增 `flow_log_fp` 参数
- `make_progress_writer(progress, flow_log_fp=flow_log_fp)` 修复 flow_log 不写入 bug
- SACConfig 构造新增 6 个字段从 `sac_cfg_dict` 读取
- `learning_starts` 首次触发时执行 critic warmup
- 每 `eval_interval` episode 跑 1 个确定性评估 episode
- episode 结束后收集 stats dict（loss/alpha/Q 值）
- 训练结束写 `training_stats.csv`（增强版 CSV）
- 调用处传入 `flow_log_fp=flow_log_fp`
- CLI 注册 7 个新参数
