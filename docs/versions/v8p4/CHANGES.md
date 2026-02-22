# v8p4 CHANGES（相对 v8p3）

## 代码改动

### `forest_vehicle_dqn/sac_agent.py`

1. 新增 `import math`
2. `SACConfig` 新增字段：`alpha_min: float = 0.0`
3. 标准 SAC `update()` 的 `alpha_opt.step()` 后新增 alpha clamp：
   - `if self.cfg.alpha_min > 0: log_alpha.data.clamp_(min=math.log(alpha_min))`
4. TECRL `update_tecrl()` 的 `alpha_opt.step()` 后新增同样 clamp
5. `update_tecrl()` 返回 dict 新增兼容 key：
   - `"target_q_mean"` → `target_q_r.mean().item()`
   - `"target_q_max"` → `target_q_r.max().item()`

### `forest_vehicle_dqn/env.py`

1. PBRS 公式修正（L1361-1362）：
   - old: `(phi_now + bias) - gamma * (phi_prev + bias)`
   - new: `gamma * (phi_now + bias) - (phi_prev + bias)`

### `forest_vehicle_dqn/cli/train.py`

1. CLI 新增 `--sac-alpha-min`（default=0.0）
2. `train_one_sac()` 签名新增 `alpha_min: float = 0.0`
3. `SACConfig` 构造新增 `alpha_min=...` 传递
4. 调用处新增 `alpha_min=...` 传参

### `configs/v8p4.json`（新增）

- `reward_potential_base`: 32 → 4
- `sac_entropy_budget_ratio`: 0.6 → 2.0
- `sac_alpha_min`: 0.01（新增）
- infer 区 `models`/`out`: v8p3 → v8p4
