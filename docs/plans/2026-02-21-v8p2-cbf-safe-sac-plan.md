# v8p2 CBF-Safe SAC 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 SAC 添加 discrete-time CBF 安全过滤 + CBF-aware 奖励塑形 + alpha 衰减修复，突破 SR=0%。

**Architecture:** 在 env.py 新增 `cbf_safe_action()` 方法实现 CBF 投影；修改奖励函数替换 k_p/k_o 为 potential-based + log-barrier；在 train.py 的 SAC 训练循环中调用 CBF filter。

**Tech Stack:** Python 3.10, PyTorch, NumPy, conda env `ros2py310`

---

### Task 1: 创建 v8p2 配置文件

**Files:**
- Create: `configs/v8p2.json`

**Step 1: 创建 v8p2.json**

基于 `configs/v8.json`，修改以下字段：

```json
{
  "_meta": {
    "title": "v8p2 CBF-Safe SAC (safety filter + potential shaping + alpha fix)",
    "date": "2026-02-21",
    "summary": "v8p1 + discrete-time CBF safety filter, CBF-aware reward shaping, alpha decay fix.",
    "design_doc": "docs/plans/2026-02-21-v8p2-cbf-safe-sac-design.md"
  },
  "train": {
    "...保留 v8 所有字段...",
    "out": "v8p2",
    "sac_lr_alpha": 1e-4,
    "sac_target_entropy": -1.0,
    "forest_reward_k_p": 0.0,
    "forest_reward_k_o": 0.0,
    "forest_cbf_alpha": 0.3,
    "forest_cbf_safety_margin_m": 0.15,
    "forest_reward_c_prog": 5.0,
    "forest_reward_c_cbf": 1.0,
    "forest_cbf_h_max": 2.0
  },
  "infer": {
    "...保留 v8 所有字段...",
    "out": "v8p2",
    "models": "v8p2"
  }
}
```

**Step 2: 验证 JSON 合法**

Run: `conda run -n ros2py310 python -c "import json; json.load(open('configs/v8p2.json'))"`
Expected: 无输出（成功）

**Step 3: Commit**

```bash
git add configs/v8p2.json
git commit -m "feat(v8p2): add CBF-Safe SAC config"
```

---

### Task 2: 实现 CBF 安全过滤器（env.py）

**Files:**
- Modify: `forest_vehicle_dqn/env.py` — 新增 `cbf_safe_action()` 方法
- Create: `tests/test_cbf_safety.py`

**Step 1: 写失败测试**

创建 `tests/test_cbf_safety.py`：

```python
"""CBF safety filter 单元测试。"""
import math
import numpy as np
import pytest

def test_cbf_safe_action_passthrough():
    """安全动作应直通不修改。"""
    from forest_vehicle_dqn.env import ForestEnv
    env = ForestEnv(map_name="forest_a")
    env.reset(seed=42)
    dd_out, a_out, info = env.cbf_safe_action(0.0, 0.1)
    assert not info["cbf_intervened"], "安全动作不应被 CBF 修改"
    assert dd_out == pytest.approx(0.0, abs=1e-6)
    assert a_out == pytest.approx(0.1, abs=1e-6)

def test_cbf_safe_action_intervenes_near_obstacle():
    """接近障碍物时 CBF 应介入修正动作。"""
    from forest_vehicle_dqn.env import ForestEnv
    env = ForestEnv(map_name="forest_a")
    env.reset(seed=42)
    _, _, info = env.cbf_safe_action(0.0, 1.0, safety_margin=100.0)
    assert info["cbf_intervened"], "接近障碍物时 CBF 应介入"

def test_cbf_safe_action_output_in_bounds():
    """CBF 输出动作应在 [-1, 1] 范围内。"""
    from forest_vehicle_dqn.env import ForestEnv
    env = ForestEnv(map_name="forest_a")
    env.reset(seed=42)
    for dd_raw, a_raw in [(-1, -1), (1, 1), (0, 0), (-0.5, 0.8)]:
        dd_out, a_out, _ = env.cbf_safe_action(dd_raw, a_raw)
        assert -1.0 <= dd_out <= 1.0, f"dd_out={dd_out} 超出范围"
        assert -1.0 <= a_out <= 1.0, f"a_out={a_out} 超出范围"
```

**Step 2: 运行测试确认失败**

Run: `conda run -n ros2py310 python -m pytest tests/test_cbf_safety.py -v`
Expected: FAIL — `AttributeError: 'ForestEnv' object has no attribute 'cbf_safe_action'`

**Step 3: 实现 `cbf_safe_action()`**

在 `forest_vehicle_dqn/env.py` 的 `step_continuous()` 方法之后（约 line 1441）新增方法。

核心逻辑：
1. 计算当前 barrier 值 `h_curr = od_curr - safety_margin`
2. 若 `h_curr <= 0`：emergency brake `(0, -1)`
3. 模拟一步 `bicycle_integrate_one_step` → 计算 `h_next`
4. CBF 约束：`h_next >= (1 - α) * h_curr`
5. 若违反：数值梯度（中心差分，4 次额外 rollout）→ 线性化投影
6. 投影后验证；若仍不安全 → fallback brake

完整实现见设计文档 `docs/plans/2026-02-21-v8p2-cbf-safe-sac-design.md` 第 3 节。

**Step 4: 运行测试确认通过**

Run: `conda run -n ros2py310 python -m pytest tests/test_cbf_safety.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add forest_vehicle_dqn/env.py tests/test_cbf_safety.py
git commit -m "feat(v8p2): add CBF safety filter to ForestEnv"
```

---

### Task 3: 实现 CBF-aware 奖励塑形（env.py）

**Files:**
- Modify: `forest_vehicle_dqn/env.py:600-610` — 新增构造参数
- Modify: `forest_vehicle_dqn/env.py:1328-1362` — 替换 progress 和 obstacle 奖励

**Step 1: 添加构造参数**

在 `ForestEnv.__init__` 参数列表中新增 `reward_c_prog`, `reward_c_cbf`, `cbf_h_max`, `gamma`。
在 body 中赋值并预计算 `self._cbf_collision_thr`。

**Step 2: 修改奖励函数**

- 当 `reward_c_prog > 0`：用 `c_prog * (dist_before - γ * dist_after)` 替代 `k_p * (dist_before - dist_after)`
- 当 `reward_c_cbf > 0`：用 `c_cbf * log(max(h/h_max, ε))` 替代 `k_o * (1/od - 1/safe_dist)`
- 当系数为 0 时保留旧逻辑（向后兼容）

**Step 3: 运行 self-check**

Run: `conda run -n ros2py310 python train.py --self-check`
Expected: PASS

**Step 4: Commit**

```bash
git add forest_vehicle_dqn/env.py
git commit -m "feat(v8p2): add CBF-aware reward shaping"
```

---

### Task 4: 注册 CLI 参数 + 训练循环集成（train.py）

**Files:**
- Modify: `forest_vehicle_dqn/cli/train.py` — CLI args + env 构造 + SAC 循环

**Step 1: 注册 CLI 参数**

新增 `--forest-cbf-alpha`, `--forest-cbf-safety-margin-m`, `--forest-reward-c-prog`, `--forest-reward-c-cbf`, `--forest-cbf-h-max`, `--forest-reward-k-o`。

**Step 2: 传递到 ForestEnv 构造**

在 env 构造调用中新增对应参数传递。

**Step 3: SAC 训练循环集成 CBF filter**

在 `train_one_sac()` 的 action → step 之间插入 CBF filter 调用。
同样修改 eval 循环和 BC pretrain rollout。
添加 `cbf_interventions` 计数器和日志。

**Step 4: 运行 self-check**

Run: `conda run -n ros2py310 python train.py --self-check`
Expected: PASS

**Step 5: Commit**

```bash
git add forest_vehicle_dqn/cli/train.py
git commit -m "feat(v8p2): integrate CBF filter into SAC training loop"
```

---

### Task 5: 全量测试

**Step 1:** `conda run -n ros2py310 python -m pytest tests/test_cbf_safety.py -v` → 3 passed
**Step 2:** `conda run -n ros2py310 python train.py --self-check` → PASS
**Step 3:** `conda run -n ros2py310 python infer.py --self-check` → PASS

---

### Task 6: 同步远端 + Smoke 训练（150ep）

**Step 1: rsync 同步代码到 ubuntu-zt**

```bash
rsync -avz --exclude='runs/' --exclude='__pycache__/' --exclude='.git/' \
  /home/sun/phdproject/dqn/RL_sac/ ubuntu-zt:~/phdproject/dqn/RL_sac/
```

**Step 2: 远端 smoke 训练**

```bash
ssh ubuntu-zt "source ~/miniconda3/etc/profile.d/conda.sh && \
  cd ~/phdproject/dqn/RL_sac && \
  conda run -n ros2py310 python train.py --profile v8p2 \
    --episodes 150 --out v8p2-smoke1 --no-progress"
```

**Step 3: 回传结果**

```bash
rsync -avz ubuntu-zt:~/phdproject/dqn/RL_sac/runs/v8p2-smoke1/ \
  /home/sun/phdproject/dqn/RL_sac/runs/v8p2-smoke1/
```

---

### Task 7: Smoke 推理（runs=3）+ 结果分析

**Step 1: 远端推理**

```bash
ssh ubuntu-zt "source ~/miniconda3/etc/profile.d/conda.sh && \
  cd ~/phdproject/dqn/RL_sac && \
  conda run -n ros2py310 python infer.py --profile v8p2 \
    --runs 3 --models runs/v8p2-smoke1/<best_train_dir> --no-progress"
```

**Step 2: 回传 + 分析 `table2_kpis_mean_raw.csv`**

判断标准：
- SR > 0% → 进入 Task 8 归档，考虑 v8p3 课程学习
- SR = 0% → 分析 CBF 日志和回报曲线，调参后重试

---

### Task 8: 版本归档（v8p2 四件套）

**Files:**
- Create: `docs/versions/v8p2/README.md`
- Create: `docs/versions/v8p2/CHANGES.md`
- Create: `docs/versions/v8p2/RESULTS.md`
- Create: `docs/versions/v8p2/runs/README.md`
- Modify: `docs/versions/README.md`

按 CLAUDE.md 第 16 条归档标准记录全部信息。

```bash
git add docs/versions/v8p2/
git commit -m "docs(v8p2): archive smoke results"
```