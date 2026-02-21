# v8p3 TECRL + Improved Potential-Based Shaping + Syllabus PLR 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 v8p2 (CBF-Safe SAC, SR=0%) 基础上，通过 TECRL reward-entropy 分离、改进 potential-based 奖励塑形、Syllabus PLR 课程学习三个正交改进突破 SR=0%。

**Architecture:** TECRL 分离 reward-critic 和 entropy-critic 解决 alpha 下降过快；指数 potential + 偏置修正提供更密集的奖励信号；Syllabus PLR 按 TD-error 优先采样高学习价值的难度等级。三个改进正交，保留 v8p2 的 CBF safety filter。

**Tech Stack:** PyTorch, syllabus-rl (PLR), numpy, 现有 forest_vehicle_dqn 框架

**设计文档:** `docs/plans/2026-02-21-v8p3-tecrl-plr-design.md`

---

## 前置条件

- 分支：从当前 `rollback/v6p2p3-20260219_212621` 开始
- 确保 `git status` clean（先 commit 未提交的改动）
- 确保 `python train.py --self-check` 和 `python infer.py --self-check` 通过

---

### Task 0: Git 快照 + 安装依赖

**Step 1: Git 快照**

```bash
git add -A && git commit -m "chore: pre-v8p3 snapshot"
git push origin rollback/v6p2p3-20260219_212621
```

**Step 2: 安装 Syllabus**

```bash
conda run -n ros2py310 pip install syllabus-rl
```

**Step 3: 验证安装**

```bash
conda run -n ros2py310 python -c "import syllabus; print(syllabus.__version__)"
```

Expected: 版本号输出，无报错。

**Step 4: 远端同步安装**

```bash
ssh ubuntu-zt "source ~/miniconda3/etc/profile.d/conda.sh && conda run -n ros2py310 pip install syllabus-rl"
```

**Step 5: Commit**

```bash
git add -A && git commit -m "chore(v8p3): install syllabus-rl dependency"
```

---

### Task 1: TECRL — SACEntropyCritic 网络

**Files:**
- Modify: `forest_vehicle_dqn/sac_networks.py` (末尾追加)

**Step 1: 在 `sac_networks.py` 末尾新增 `SACEntropyCritic` 类**

与 `SACCritic` 结构完全相同（twin Q），但语义上用于估计累积熵。
复用 `SACCritic` 的代码，仅改类名和 docstring：

```python
class SACEntropyCritic(nn.Module):
    """Twin Q-networks for cumulative entropy estimation (TECRL)."""

    def __init__(self, encoder: GlobalCNNEncoder, action_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        feat = encoder.feature_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(feat, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feat, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, maps, scalars, actions):
        h = self.encoder(maps, scalars)
        ha = torch.cat([h, actions], dim=1)
        return self.q1(ha), self.q2(ha)
```

**Step 2: 更新 `sac_agent.py` 的 import**

```python
from forest_vehicle_dqn.sac_networks import (
    GlobalCNNEncoder, SACActor, SACCritic, SACEntropyCritic,
)
```

**Step 3: Commit**

```bash
git add forest_vehicle_dqn/sac_networks.py forest_vehicle_dqn/sac_agent.py
git commit -m "feat(v8p3): add SACEntropyCritic network for TECRL"
```

---

### Task 2: TECRL — SACConfig 扩展 + SACAgent 双 critic 初始化

**Files:**
- Modify: `forest_vehicle_dqn/sac_agent.py`

**Step 1: SACConfig 新增 TECRL 字段**

在 `SACConfig` dataclass 末尾（`critic_warmup_steps` 之后）追加：

```python
    # TECRL (reward-entropy separation)
    use_tecrl: bool = False
    lr_entropy_critic: float = 3e-4
    entropy_budget_ratio: float = 0.6  # ρ: trajectory entropy budget
```

**Step 2: SACAgent.__init__ 新增 entropy critic**

在 `self.critic_target` 初始化之后、`# Automatic entropy tuning` 之前插入：

```python
        # TECRL: entropy critic (estimates cumulative entropy)
        self.use_tecrl = config.use_tecrl
        if self.use_tecrl:
            enc_ent = GlobalCNNEncoder(
                config.map_size, config.map_channels, config.scalar_dim)
            self.entropy_critic = SACEntropyCritic(
                enc_ent, config.action_dim, config.hidden_dim
            ).to(self.device)
            self.entropy_critic_target = copy.deepcopy(self.entropy_critic)
            for p in self.entropy_critic_target.parameters():
                p.requires_grad_(False)
            self.entropy_critic_opt = torch.optim.Adam(
                self.entropy_critic.parameters(), lr=config.lr_entropy_critic)
            # H_budget = ρ * H_0 / (1 - γ), H_0 = -dim(action)
            self.entropy_budget = (
                config.entropy_budget_ratio * (-config.action_dim)
                / (1.0 - config.gamma)
            )
```

**Step 3: Commit**

```bash
git add forest_vehicle_dqn/sac_agent.py
git commit -m "feat(v8p3): extend SACConfig and SACAgent init for TECRL"
```

---

### Task 3: TECRL — 重写 update() 方法

**Files:**
- Modify: `forest_vehicle_dqn/sac_agent.py`

**核心改动**：当 `use_tecrl=True` 时，`update()` 方法替换为 TECRL 逻辑：

1. **Reward critic update**：target 不含 alpha 项
   - `target_q_r = r + γ * (1-d) * min(Q_r_target(s',a'))`
2. **Entropy critic update**：target 为单步熵
   - `target_q_e = -log π(a'|s') + γ * (1-d) * min(Q_e_target(s',a'))`
3. **Actor update**：组合 reward + alpha * entropy
   - `actor_loss = (-Q_r(s,a_new) - α * Q_e(s,a_new)).mean()`
4. **Alpha update**：轨迹级熵约束
   - `α_loss = -α * (Q_e(s,a_new) + log π(a_new|s) - H_budget)`
5. **Soft target update**：reward critic + entropy critic 都做

**Step 1: 新增 `update_tecrl()` 方法**

在 `update()` 方法之后新增。具体代码见设计文档 §2.1。
关键伪代码：

```python
def update_tecrl(self) -> Dict[str, float]:
    # sample batch (same as update())
    # 1. reward critic: target without alpha
    with torch.no_grad():
        n_action, n_log_prob = self.actor.sample(...)
        tq1_r, tq2_r = self.critic_target(...)
        target_q_r = rewards_t + (1-dones_t) * gamma * torch.min(tq1_r, tq2_r)
    # ... critic loss + step

    # 2. entropy critic: target = -log_prob + gamma * Q_e_target
    with torch.no_grad():
        tq1_e, tq2_e = self.entropy_critic_target(...)
        target_q_e = (-n_log_prob.unsqueeze(1)
                      + (1-dones_t) * gamma * torch.min(tq1_e, tq2_e))
    # ... entropy critic loss + step

    # 3. actor: maximize Q_r + alpha * Q_e
    new_action, log_prob = self.actor.sample(...)
    q_r = torch.min(*self.critic(...))
    q_e = torch.min(*self.entropy_critic(...))
    actor_loss = (-q_r - alpha * q_e).mean()

    # 4. alpha: trajectory entropy constraint
    alpha_loss = -(self.log_alpha * (
        q_e.detach() + log_prob.detach().unsqueeze(1)
        - self.entropy_budget)).mean()

    # 5. soft target update both critics
```

**Step 2: 修改 `update()` 入口分发**

在现有 `update()` 开头加判断：

```python
def update(self) -> Dict[str, float]:
    if self.use_tecrl:
        return self.update_tecrl()
    # ... 原有逻辑不变
```

**Step 3: 更新 save/load 方法**

`save()` 和 `load()` 中增加 entropy critic 相关 state_dict。

**Step 4: Commit**

```bash
git add forest_vehicle_dqn/sac_agent.py
git commit -m "feat(v8p3): implement TECRL update with reward-entropy separation"
```

---

### Task 4: 改进 Potential-Based Reward Shaping

**Files:**
- Modify: `forest_vehicle_dqn/env.py`

**Step 1: AMRBicycleEnv.__init__ 新增参数**

在 `reward_c_cbf` 参数之后追加：

```python
reward_potential_base: float = 0.0,  # 指数 potential 底数，0=用 v8p2 线性
reward_potential_bias: float = 0.0,  # 偏置修正项
```

在 `__init__` body 中保存：

```python
self.reward_potential_base = float(reward_potential_base)
self.reward_potential_bias = float(reward_potential_bias)
self._initial_dist = None  # 每 episode reset 时记录
```

**Step 2: reset() 中记录初始距离**

在 `reset()` 方法返回前，记录 `self._initial_dist = dist_to_goal`。

**Step 3: 替换 progress reward 计算**

在 reward 计算区域（约 L1341），当 `reward_potential_base > 0` 时：

```python
if self.reward_potential_base > 0.0 and self._initial_dist > 0:
    base = self.reward_potential_base
    prog_now = 1.0 - dist_after / self._initial_dist
    prog_prev = 1.0 - dist_before / self._initial_dist
    phi_now = math.exp(base * max(0, min(1, prog_now)))
    phi_prev = math.exp(base * max(0, min(1, prog_prev)))
    bias = self.reward_potential_bias
    shaped = self.reward_c_prog * (
        (phi_now + bias) - self._gamma * (phi_prev + bias))
    reward += shaped
```

否则退回 v8p2 的线性 potential 逻辑（已有代码不变）。

**Step 4: Commit**

```bash
git add forest_vehicle_dqn/env.py
git commit -m "feat(v8p3): exponential potential-based reward shaping with bias"
```

---

### Task 5: Syllabus PLR Curriculum Wrapper

**Files:**
- Create: `forest_vehicle_dqn/plr_curriculum.py`

**Step 1: 创建 PLR wrapper**

```python
"""Syllabus PLR curriculum wrapper for forest navigation."""
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

try:
    from syllabus.curricula import PrioritizedLevelReplay
    from syllabus.core import make_curriculum_env
    HAS_SYLLABUS = True
except ImportError:
    HAS_SYLLABUS = False


class ForestPLRCurriculum:
    """Maps distance levels to Syllabus PLR tasks."""

    def __init__(self, levels_m: List[float],
                 staleness_coef: float = 0.3,
                 seed: int = 0):
        assert HAS_SYLLABUS, "pip install syllabus-rl"
        self.levels_m = sorted(levels_m)
        n = len(self.levels_m)
        self.plr = PrioritizedLevelReplay(
            task_space=list(range(n)),
            staleness_coef=staleness_coef,
            num_processes=1,
        )
        self._rng = np.random.default_rng(seed)

    def sample_level(self) -> int:
        """Return next level index."""
        return self.plr.sample()[0]

    def level_to_dist_range(self, idx: int):
        """Convert level index to (min_dist, max_dist)."""
        lo = self.levels_m[idx]
        hi = (self.levels_m[idx + 1]
              if idx + 1 < len(self.levels_m) else 0.0)
        return float(lo), float(hi)

    def report(self, level_idx: int, episode_return: float):
        """Report episode result to PLR."""
        self.plr.update_on_demand(
            level_idx, episode_return)
```

**Step 2: Commit**

```bash
git add forest_vehicle_dqn/plr_curriculum.py
git commit -m "feat(v8p3): add Syllabus PLR curriculum wrapper"
```

---

### Task 6: 训练循环集成（train.py）

**Files:**
- Modify: `forest_vehicle_dqn/cli/train.py`

**Step 1: 新增 CLI 参数**

在 argparse 区域（约 L3034 附近）追加：

```
--sac-use-tecrl          (store_true)
--sac-entropy-budget-ratio (float, default=0.6)
--sac-lr-entropy-critic  (float, default=3e-4)
--reward-potential-base   (float, default=0.0)
--reward-potential-bias   (float, default=0.0)
--use-syllabus-plr       (store_true)
--plr-levels             (str, default="6,10,14,20,30,42")
--plr-staleness-coef     (float, default=0.3)
```

**Step 2: train_one_sac() 新增参数**

在 `train_one_sac()` 签名中追加对应参数，并传入 `SACConfig`。

**Step 3: PLR 集成到训练循环**

在 episode 循环开头，当 `use_syllabus_plr=True` 时：

```python
if plr_curriculum is not None:
    level_idx = plr_curriculum.sample_level()
    lo, hi = plr_curriculum.level_to_dist_range(level_idx)
    reset_options = {"rand_min_dist_m": lo, "rand_max_dist_m": hi, ...}
```

episode 结束后：

```python
if plr_curriculum is not None:
    plr_curriculum.report(level_idx, ep_return)
```

**Step 4: 传递 reward shaping 参数到 env**

在 env 构造时传入 `reward_potential_base` 和 `reward_potential_bias`。

**Step 5: 传递 TECRL 参数到 SACConfig**

在 `sac_config = SACConfig(...)` 中追加 TECRL 字段。

**Step 6: Commit**

```bash
git add forest_vehicle_dqn/cli/train.py
git commit -m "feat(v8p3): integrate TECRL + PLR + exp-potential into training loop"
```

---

### Task 7: v8p3 配置文件

**Files:**
- Create: `configs/v8p3.json`

**Step 1: 基于 v8p2.json 创建 v8p3.json**

复制 `configs/v8p2.json` → `configs/v8p3.json`，修改以下字段：

```json
{
  "_meta": {
    "title": "v8p3 TECRL + Exp-Potential + Syllabus PLR",
    "date": "2026-02-21"
  },
  "train": {
    "out": "v8p3",
    "sac_use_tecrl": true,
    "sac_entropy_budget_ratio": 0.6,
    "sac_lr_entropy_critic": 3e-4,
    "reward_potential_base": 32,
    "reward_potential_bias": 0.0,
    "use_syllabus_plr": true,
    "plr_levels": "6,10,14,20,30,42",
    "plr_staleness_coef": 0.3,
    "forest_curriculum": false,
    "forest_train_two_suites": false
  }
}
```

其余字段保持与 v8p2 一致。

**Step 2: Commit**

```bash
git add configs/v8p3.json
git commit -m "feat(v8p3): add v8p3 config"
```

---

### Task 8: 单元测试

**Files:**
- Create: `tests/test_tecrl.py`
- Create: `tests/test_plr_curriculum.py`

**Step 1: TECRL 单元测试**

```python
"""Tests for TECRL (reward-entropy separated SAC)."""
import torch
from forest_vehicle_dqn.sac_agent import SACAgent, SACConfig

def test_tecrl_agent_creates():
    cfg = SACConfig(use_tecrl=True, buffer_size=100)
    agent = SACAgent(cfg, device="cpu", seed=0)
    assert hasattr(agent, "entropy_critic")
    assert hasattr(agent, "entropy_budget")

def test_tecrl_update_returns_stats():
    cfg = SACConfig(use_tecrl=True, buffer_size=100,
                    batch_size=4)
    agent = SACAgent(cfg, device="cpu", seed=0)
    # fill buffer with dummy data
    for _ in range(10):
        obs = {"maps": torch.randn(3,48,48).numpy(),
               "scalars": torch.randn(12).numpy()}
        act = torch.randn(2).numpy()
        agent.observe(obs, act, -1.0, obs, False)
    stats = agent.update()
    assert "critic_loss" in stats
    assert "alpha" in stats
```

**Step 2: PLR 单元测试**

```python
"""Tests for ForestPLRCurriculum."""
import pytest
from forest_vehicle_dqn.plr_curriculum import (
    ForestPLRCurriculum, HAS_SYLLABUS)

@pytest.mark.skipif(not HAS_SYLLABUS,
                    reason="syllabus-rl not installed")
def test_plr_sample_and_report():
    plr = ForestPLRCurriculum([6,10,14,20,30,42])
    idx = plr.sample_level()
    assert 0 <= idx < 6
    lo, hi = plr.level_to_dist_range(idx)
    assert lo >= 6.0
    plr.report(idx, -100.0)  # should not raise
```

**Step 3: 运行测试**

```bash
conda run -n ros2py310 pytest tests/test_tecrl.py tests/test_plr_curriculum.py -v
```

**Step 4: Commit**

```bash
git add tests/test_tecrl.py tests/test_plr_curriculum.py
git commit -m "test(v8p3): add TECRL and PLR curriculum unit tests"
```

---

### Task 9: Self-check + Smoke 测试

**Step 1: 本地 self-check**

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

Expected: 两个都 PASS。

**Step 2: 同步到远端**

```bash
rsync -avz --exclude='runs/' --exclude='.git/' \
  /home/sun/phdproject/dqn/RL_sac/ \
  ubuntu-zt:~/phdproject/dqn/RL_sac/
```

**Step 3: 远端 smoke 训练（150ep）**

```bash
ssh ubuntu-zt "source ~/miniconda3/etc/profile.d/conda.sh && \
  cd ~/phdproject/dqn/RL_sac && \
  conda run -n ros2py310 python train.py \
    --profile v8p3 --episodes 150 \
    --out v8p3-smoke1 --no-progress"
```

**Step 4: 远端 smoke 推理（runs=3）**

```bash
ssh ubuntu-zt "source ~/miniconda3/etc/profile.d/conda.sh && \
  cd ~/phdproject/dqn/RL_sac && \
  conda run -n ros2py310 python infer.py \
    --profile v8p3 --runs 3 \
    --models runs/v8p3-smoke1/<train_dir> \
    --out v8p3-smoke1 --no-progress"
```

**Step 5: 回传结果**

```bash
rsync -avz ubuntu-zt:~/phdproject/dqn/RL_sac/runs/v8p3-smoke1/ \
  /home/sun/phdproject/dqn/RL_sac/runs/v8p3-smoke1/
```

**Step 6: 分析结果**

检查：SR > 0%？alpha 是否稳定？best_return？CBF 介入次数？

---

### Task 10: 版本留档（v8p3 四件套）

**Files:**
- Create: `docs/versions/v8p3/README.md`
- Create: `docs/versions/v8p3/CHANGES.md`
- Create: `docs/versions/v8p3/RESULTS.md`
- Create: `docs/versions/v8p3/runs/README.md`

**Step 1: 创建目录**

```bash
mkdir -p docs/versions/v8p3/runs
```

**Step 2: 填写四件套**

按 AGENTS.md 第 16 条归档标准，记录：
- README.md：版本目标、方法摘要、关键命令、代表 run、结论
- CHANGES.md：相对 v8p2 的改动明细
- RESULTS.md：smoke 结果、指标、门槛检查
- runs/README.md：run 路径与口径

**Step 3: 更新 docs/versions/README.md 索引**

**Step 4: Commit**

```bash
git add docs/versions/v8p3/
git commit -m "docs(v8p3): add version archive (four-piece set)"
```

---

## 执行顺序总结

| Task | 内容 | 预计改动量 |
|------|------|-----------|
| 0 | Git 快照 + 安装 syllabus-rl | 0 行代码 |
| 1 | SACEntropyCritic 网络 | ~30 行 |
| 2 | SACConfig + Agent 初始化 | ~25 行 |
| 3 | TECRL update() 方法 | ~80 行 |
| 4 | 指数 potential 奖励塑形 | ~20 行 |
| 5 | Syllabus PLR wrapper | ~50 行 |
| 6 | 训练循环集成 | ~40 行 |
| 7 | v8p3 配置文件 | ~120 行 JSON |
| 8 | 单元测试 | ~50 行 |
| 9 | Self-check + Smoke | 0 行代码 |
| 10 | 版本留档 | 文档 |

**总代码改动量**：约 300 行（不含 JSON 配置和文档）。

---

**Plan saved to:** `docs/plans/2026-02-21-v8p3-implementation-plan.md`
