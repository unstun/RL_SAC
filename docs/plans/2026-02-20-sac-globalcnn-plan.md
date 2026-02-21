# SAC-GlobalCNN 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 CNN-DDQN 替换为 SAC + 全局观测 + 路径质量 Reward，在路径质量上超越 Hybrid A*-MPC。

**Architecture:** 新增 SAC Actor-Critic 网络和 SACAgent 类（与现有 DQNFamilyAgent 并行），修改 env 支持 48×48 全局地图观测，调整 reward 权重，复用现有 `step_continuous()` 接口。

**Tech Stack:** PyTorch, Gymnasium, NumPy, scipy (EDT), 现有 forest_vehicle_dqn 框架

**设计文档:** `docs/plans/2026-02-20-sac-globalcnn-design.md`

---

## Task 1: SAC 网络（Actor + Critic）

**Files:**

- Create: `forest_vehicle_dqn/sac_networks.py`
- Test: `tests/test_sac_networks.py`

**Step 1: 写失败测试**

```python
# tests/test_sac_networks.py
import torch
from forest_vehicle_dqn.sac_networks import GlobalCNNEncoder, SACActor, SACCritic

def test_encoder_output_shape():
    enc = GlobalCNNEncoder(map_size=48, map_channels=3, scalar_dim=12)
    maps = torch.randn(4, 3, 48, 48)
    scalars = torch.randn(4, 12)
    out = enc(maps, scalars)
    assert out.shape == (4, enc.feature_dim)

def test_actor_output_shape():
    enc = GlobalCNNEncoder(map_size=48, map_channels=3, scalar_dim=12)
    actor = SACActor(enc, action_dim=2, hidden_dim=256)
    maps = torch.randn(4, 3, 48, 48)
    scalars = torch.randn(4, 12)
    action, log_prob = actor.sample(maps, scalars)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert (action >= -1).all() and (action <= 1).all()  # tanh squash

def test_critic_output_shape():
    enc = GlobalCNNEncoder(map_size=48, map_channels=3, scalar_dim=12)
    critic = SACCritic(enc, action_dim=2, hidden_dim=256)
    maps = torch.randn(4, 3, 48, 48)
    scalars = torch.randn(4, 12)
    actions = torch.randn(4, 2)
    q1, q2 = critic(maps, scalars, actions)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)
```

**Step 2: 运行测试确认失败**

Run: `conda run -n ros2py310 python -m pytest tests/test_sac_networks.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'forest_vehicle_dqn.sac_networks'"

**Step 3: 实现 SAC 网络**

```python
# forest_vehicle_dqn/sac_networks.py
"""SAC Actor-Critic networks with Global CNN encoder."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class GlobalCNNEncoder(nn.Module):
    """CNN encoder for 48x48 global map + scalar features."""
    def __init__(self, map_size: int = 48, map_channels: int = 3,
                 scalar_dim: int = 12):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.conv = nn.Sequential(
            nn.Conv2d(map_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, map_channels, map_size, map_size)
            conv_out = self.conv(dummy).view(1, -1).shape[1]
        self.feature_dim = conv_out + scalar_dim

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        h = self.conv(maps).flatten(1)
        return torch.cat([h, scalars], dim=1)

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

class SACActor(nn.Module):
    """Gaussian policy with tanh squashing."""
    def __init__(self, encoder: GlobalCNNEncoder, action_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(encoder.feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor):
        h = F.relu(self.fc1(self.encoder(maps, scalars)))
        h = F.relu(self.fc2(h))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, maps: torch.Tensor, scalars: torch.Tensor):
        mean, log_std = self.forward(maps, scalars)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()  # reparameterization trick
        action = torch.tanh(x)
        # log_prob with tanh correction
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def deterministic(self, maps: torch.Tensor, scalars: torch.Tensor):
        mean, _ = self.forward(maps, scalars)
        return torch.tanh(mean)

class SACCritic(nn.Module):
    """Twin Q-networks."""
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

    def forward(self, maps: torch.Tensor, scalars: torch.Tensor,
                actions: torch.Tensor):
        h = self.encoder(maps, scalars)
        ha = torch.cat([h, actions], dim=1)
        return self.q1(ha), self.q2(ha)
```

**Step 4: 运行测试确认通过**

Run: `conda run -n ros2py310 python -m pytest tests/test_sac_networks.py -v`
Expected: 3 PASS

**Step 5: 提交**

```bash
git add forest_vehicle_dqn/sac_networks.py tests/test_sac_networks.py
git commit -m "feat(v8): add SAC actor-critic networks with global CNN encoder"
```

---

## Task 2: SAC Agent 类

**Files:**

- Create: `forest_vehicle_dqn/sac_agent.py`
- Test: `tests/test_sac_agent.py`

**Step 1: 写失败测试**

```python
# tests/test_sac_agent.py
import numpy as np
from forest_vehicle_dqn.sac_agent import SACAgent, SACConfig

def test_agent_act_returns_continuous():
    cfg = SACConfig(map_size=48, map_channels=3, scalar_dim=12)
    agent = SACAgent(cfg, device="cpu")
    obs = {"maps": np.random.randn(3, 48, 48).astype(np.float32),
           "scalars": np.random.randn(12).astype(np.float32)}
    action = agent.act(obs, explore=True)
    assert action.shape == (2,)
    assert np.all(action >= -1) and np.all(action <= 1)

def test_agent_update_returns_losses():
    cfg = SACConfig(map_size=48, map_channels=3, scalar_dim=12, batch_size=4)
    agent = SACAgent(cfg, device="cpu")
    obs = {"maps": np.random.randn(3, 48, 48).astype(np.float32),
           "scalars": np.random.randn(12).astype(np.float32)}
    for _ in range(8):
        action = agent.act(obs, explore=True)
        agent.observe(obs, action, -1.0, obs, False)
    losses = agent.update()
    assert "critic_loss" in losses
    assert "actor_loss" in losses
    assert "alpha" in losses

def test_agent_save_load(tmp_path):
    cfg = SACConfig(map_size=48, map_channels=3, scalar_dim=12)
    agent = SACAgent(cfg, device="cpu")
    path = tmp_path / "sac_test.pt"
    agent.save(path)
    agent2 = SACAgent(cfg, device="cpu")
    agent2.load(path)
```

**Step 2: 运行测试确认失败**

Run: `conda run -n ros2py310 python -m pytest tests/test_sac_agent.py -v`
Expected: FAIL

**Step 3: 实现 SACAgent**

核心实现要点（`forest_vehicle_dqn/sac_agent.py`）：

```python
@dataclass
class SACConfig:
    map_size: int = 48
    map_channels: int = 3
    scalar_dim: int = 12
    action_dim: int = 2
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    target_entropy: float = -2.0  # -dim(action)

class SACAgent:
    def __init__(self, config: SACConfig, *, device="cpu", seed=0): ...
    def act(self, obs: dict, *, explore=True) -> np.ndarray: ...
    def observe(self, obs, action, reward, next_obs, done): ...
    def update(self) -> dict[str, float]: ...
    def pretrain_bc(self, demos, *, steps=5000): ...
    def save(self, path) / load(self, path): ...
```

关键实现：
- `act()`: explore=True 时用 `actor.sample()`，explore=False 时用 `actor.deterministic()`
- `update()`: 从 buffer 采样 → 更新 critic (Twin Q, min target) → 更新 actor (max Q + entropy) → 更新 alpha → soft update target critic
- `pretrain_bc()`: 用专家数据做 MSE loss 训练 actor
- Replay buffer: 存储 `(maps, scalars, action, reward, next_maps, next_scalars, done)`

**Step 4: 运行测试确认通过**

Run: `conda run -n ros2py310 python -m pytest tests/test_sac_agent.py -v`
Expected: 3 PASS

**Step 5: 提交**

```bash
git add forest_vehicle_dqn/sac_agent.py tests/test_sac_agent.py
git commit -m "feat(v8): add SACAgent with replay buffer and BC pretraining"
```

---

## Task 3: 环境全局观测

**Files:**

- Modify: `forest_vehicle_dqn/env.py` (AMRBicycleEnv)
- Test: `tests/test_global_obs.py`

**Step 1: 写失败测试**

```python
# tests/test_global_obs.py
import numpy as np
from forest_vehicle_dqn.env import AMRBicycleEnv

def test_global_obs_shape():
    # 需要一个简单的 MapSpec 来构造 env
    env = make_test_env(global_map_size=48)  # helper
    obs, _ = env.reset()
    assert "maps" in obs or hasattr(obs, "shape")
    gmap = env.get_global_map(channels=3)
    assert gmap.shape == (3, 48, 48)
    assert gmap.dtype == np.float32
```

**Step 2: 实现 `get_global_map()` 方法**

在 `AMRBicycleEnv` 中新增方法：
- 将 `self._grid`（完整占用栅格）下采样到 48×48
- Ch0: 占用栅格（二值）
- Ch1: agent 位置高斯 blob
- Ch2: goal 位置高斯 blob
- 返回 `np.ndarray` shape `(3, 48, 48)`, dtype `float32`

新增 `_observe_sac()` 方法返回 dict `{"maps": (3,48,48), "scalars": (12-14,)}`，
与现有 `_observe()` 并行，通过配置开关选择。

**Step 3-5: 测试、验证、提交**

```bash
git add forest_vehicle_dqn/env.py tests/test_global_obs.py
git commit -m "feat(v8): add global map observation for SAC"
```

---

## Task 4: Reward 权重调整

**Files:**

- Modify: `forest_vehicle_dqn/env.py` (reward 参数默认值不改，通过 config 传入)
- Create: `configs/v8.json`

**Step 1: 创建 v8 配置文件**

基于 v7p1.json，修改以下字段：

```json
{
  "rl_algos": ["cnn-sac"],
  "episodes": 3000,
  "gamma": 0.99,
  "reward_k_p": 8.0,
  "reward_k_t": 0.8,
  "reward_k_kappa": 1.0,
  "reward_k_len": 0.4,
  "forest_action_mode": "continuous",
  "sac_lr_actor": 3e-4,
  "sac_lr_critic": 3e-4,
  "sac_tau": 0.005,
  "sac_batch_size": 256,
  "sac_buffer_size": 1000000,
  "sac_bc_pretrain_steps": 5000,
  "global_map_size": 48,
  "global_map_channels": 3
}
```

**Step 2: 在 env 中新增 `reward_k_len` 参数**

在 `_reward()` 中新增路径长度惩罚项：
```python
reward -= self.reward_k_len * step_distance
```

**Step 3: 提交**

```bash
git add configs/v8.json forest_vehicle_dqn/env.py
git commit -m "feat(v8): add v8 config with SAC params and path length penalty"
```

---

## Task 5: 训练集成

**Files:**

- Modify: `forest_vehicle_dqn/cli/train.py`

**Step 1: 在训练入口添加 SAC 分支**

在 agent 实例化处添加：
```python
if "sac" in algo:
    from forest_vehicle_dqn.sac_agent import SACAgent, SACConfig
    sac_cfg = SACConfig(...)  # 从 config JSON 构造
    agent = SACAgent(sac_cfg, device=device, seed=seed)
else:
    agent = DQNFamilyAgent(...)
```

**Step 2: 修改 episode 循环支持连续动作**

```python
if isinstance(agent, SACAgent):
    obs_dict = env.observe_sac()  # 返回 {"maps": ..., "scalars": ...}
    action = agent.act(obs_dict, explore=True)
    delta_dot = action[0] * env.model.delta_dot_max_rad_s
    a = action[1] * env.model.a_max_m_s2
    next_obs_raw, reward, done, truncated, info = env.step_continuous(
        delta_dot_rad_s=delta_dot, a_m_s2=a)
    next_obs_dict = env.observe_sac()
    agent.observe(obs_dict, action, reward, next_obs_dict, done)
```

**Step 3: 添加 BC 预训练阶段**

在 RL 训练前：
```python
if isinstance(agent, SACAgent) and bc_pretrain_steps > 0:
    demos = collect_forest_demos_continuous(env, ...)
    agent.pretrain_bc(demos, steps=bc_pretrain_steps)
```

**Step 4: 提交**

```bash
git add forest_vehicle_dqn/cli/train.py
git commit -m "feat(v8): integrate SAC agent into training loop"
```

---

## Task 6: 推理集成

**Files:**

- Modify: `forest_vehicle_dqn/cli/infer.py`

**Step 1: 在 rollout 函数中添加 SAC 分支**

```python
if isinstance(agent, SACAgent):
    obs_dict = env.observe_sac()
    action = agent.act(obs_dict, explore=False)  # deterministic
    delta_dot = action[0] * env.model.delta_dot_max_rad_s
    a = action[1] * env.model.a_max_m_s2
    next_obs, reward, done, truncated, info = env.step_continuous(
        delta_dot_rad_s=delta_dot, a_m_s2=a)
```

**Step 2: 提交**

```bash
git add forest_vehicle_dqn/cli/infer.py
git commit -m "feat(v8): integrate SAC agent into inference loop"
```

---

## Task 7: Self-check 验证

**Step 1: 运行 self-check**

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

Expected: 两个都 PASS

**Step 2: 提交并推送**

```bash
git push origin HEAD
```

---

## Task 8: Smoke 测试

**Step 1: 在 ubuntu-zt 上同步并运行 smoke 训练**

```bash
# 本地 → 远端同步
rsync -avz --exclude='runs/' ./ ubuntu-zt:~/phdproject/dqn/RL_sac/

# 远端 smoke 训练 (150 episodes)
ssh ubuntu-zt "conda run -n ros2py310 python ~/phdproject/dqn/RL_sac/train.py \
  --config configs/v8.json --episodes 150 --tag v8-smoke"
```

**Step 2: Smoke 推理 (runs=3)**

```bash
ssh ubuntu-zt "conda run -n ros2py310 python ~/phdproject/dqn/RL_sac/infer.py \
  --checkpoint runs/v8-smoke/best.pt --runs 3 --suites short,mid,long"
```

**Step 3: 回传结果并分析**

```bash
rsync -avz ubuntu-zt:~/phdproject/dqn/RL_sac/runs/v8-smoke/ ./runs/v8-smoke/
```

对比 v7p1 smoke 结果，判断方向是否正确。

**Step 4: 归档到 `docs/versions/v8/`**

按版本四件套要求创建 README.md、CHANGES.md、RESULTS.md、runs/README.md。

---

## Task 9: 迭代调优（如 smoke 方向正确）

根据 smoke 结果调整：
- Reward 权重（k_t, k_len, k_kappa 的比例）
- 训练 episodes 数
- BC 预训练步数
- 全局地图分辨率

目标：在 full 评测（runs=20, short+long）上通过门槛。
