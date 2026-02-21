# v8p4 设计：修复 v8p3 三个结构性缺陷

## 背景

v8p3（TECRL + 指数 Potential + PLR）smoke 结果 SR=0%，相比 v8p2 实际退步：
- alpha 从 v8p2 的 0.047 加速崩塌到 0.003（TECRL entropy budget 过宽松）
- 指数 potential base=32 导致 reward 被 clip 成二值信号
- PBRS 公式中 γ 放错位置（Φ(s') - γ·Φ(s) 而非标准 γ·Φ(s') - Φ(s)）
- 日志 key 不匹配导致 tq 始终显示 0.00

## 改动清单

### 1. TECRL entropy budget 修复 + alpha 下界

- SACConfig 新增 `alpha_min: float = 0.0`
- `entropy_budget_ratio` 从 0.6 调到 2.0
- alpha 更新后加 clamp：`log_alpha.data.clamp_(min=log(alpha_min))`
- 影响文件：`sac_agent.py`

### 2. 指数 potential base 降到 4 + 修正 PBRS 公式

- config: `reward_potential_base` 32 → 4
- env.py PBRS 公式修正：
  - 旧：`c_prog * ((phi_now + bias) - gamma * (phi_prev + bias))`
  - 新：`c_prog * (gamma * (phi_now + bias) - (phi_prev + bias))`
- 影响文件：`env.py`, `configs/v8p4.json`

### 3. 日志 key 对齐

- `update_tecrl()` 返回 dict 补充 `target_q_mean` 和 `target_q_max`
- 影响文件：`sac_agent.py`

### 4. v8p4 config

基于 v8p3.json，改动：
- `reward_potential_base`: 32 → 4
- `sac_entropy_budget_ratio`: 0.6 → 2.0
- `sac_alpha_min`: 0.01（新增）

### 5. 验证

- self-check 通过
- smoke 1000ep 训练 + runs=3 推理
- 观察指标：alpha 稳定性（≥0.01）、reward 范围、tq 非零
