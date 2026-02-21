# v8p2 CHANGES（相对 v8p1）

## 新增文件

- `configs/v8p2.json` — CBF-Safe SAC 配置
- `tests/test_cbf_safety.py` — CBF safety filter 单元测试（3 个）
- `docs/plans/2026-02-21-v8p2-cbf-safe-sac-design.md` — 设计文档
- `docs/plans/2026-02-21-v8p2-cbf-safe-sac-plan.md` — 实施计划

## 修改文件

### `forest_vehicle_dqn/env.py`

1. `AMRBicycleEnv.__init__` 新增参数：
   - `reward_c_prog`（potential-based progress reward 系数，0=用旧 k_p）
   - `reward_c_cbf`（CBF log-barrier reward 系数，0=用旧 k_o）
   - `cbf_h_max`（barrier 归一化上界）
   - `gamma`（折扣因子，用于 potential-based shaping）
2. 新增方法 `cbf_safe_action()`：discrete-time CBF 安全过滤器
   - 计算 barrier `h = min_od - safety_margin`
   - CBF 约束：`h_next >= (1 - α) * h_curr`
   - 违反时：数值梯度线性化投影 → clamp → 验证 → fallback brake
3. 奖励函数修改：
   - `reward_c_prog > 0` 时：`c_prog * (dist_k - γ * dist_{k+1})` 替代 `k_p * (dist_before - dist_after)`
   - `reward_c_cbf > 0` 时：`c_cbf * log(max(h/h_max, ε))` 替代 `k_o * (1/od - 1/safe_dist)`
   - 系数为 0 时保留旧逻辑（向后兼容）

### `forest_vehicle_dqn/cli/train.py`

1. 新增 CLI 参数：`--forest-cbf-alpha`, `--forest-cbf-safety-margin-m`, `--forest-reward-c-prog`, `--forest-reward-c-cbf`, `--forest-cbf-h-max`, `--forest-reward-k-o`
2. `train_one_sac()` 新增参数：`cbf_alpha`, `cbf_safety_margin_m`
3. SAC 训练循环：action → step 之间插入 CBF filter
4. SAC eval 循环：同样插入 CBF filter
5. 新增 `cbf_interventions` 计数器 + CSV 列 + 日志输出

## 配置变更（v8 → v8p2）

| 参数 | v8 | v8p2 | 说明 |
| --- | --- | --- | --- |
| `sac_lr_alpha` | 3e-4 | 1e-4 | 降低 alpha 学习率 |
| `sac_target_entropy` | -2.0 | -1.0 | 提高目标熵 |
| `forest_reward_k_p` | 8.0 | 0.0 | 禁用旧 progress reward |
| `forest_reward_k_o` | (默认) | 0.0 | 禁用旧 obstacle penalty |
| `forest_cbf_alpha` | N/A | 0.3 | CBF 衰减率 |
| `forest_cbf_safety_margin_m` | N/A | 0.15 | CBF 安全裕度 |
| `forest_reward_c_prog` | N/A | 5.0 | potential-based progress |
| `forest_reward_c_cbf` | N/A | 1.0 | log-barrier safety |
| `forest_cbf_h_max` | N/A | 2.0 | barrier 归一化上界 |
