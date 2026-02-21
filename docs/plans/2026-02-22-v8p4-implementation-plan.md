# v8p4 Implementation Plan: 修复 v8p3 三个结构性缺陷

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 v8p3 的 TECRL entropy budget、指数 potential base、PBRS 公式和日志 key 四个问题，1000ep smoke 验证。

**Architecture:** 在 v8p3 代码基础上做最小修复（4 个文件），不引入新功能。

**Tech Stack:** Python 3.10, PyTorch, conda env ros2py310

---

### Task 0: GitHub 快照

**Step 1:** 确认工作区 clean

Run: `git status`
Expected: clean

**Step 2:** Push 当前状态

Run: `git push`

---

### Task 1: SACConfig 新增 alpha_min 字段

**Files:**
- Modify: `forest_vehicle_dqn/sac_agent.py:42-44`

**Step 1:** 在 SACConfig 的 `entropy_budget_ratio` 字段后新增：

```python
    alpha_min: float = 0.0  # minimum alpha (0 = no floor)
```

**Step 2:** 验证 import

Run: `conda run -n ros2py310 python -c "from forest_vehicle_dqn.sac_agent import SACConfig; c=SACConfig(alpha_min=0.01); print(c.alpha_min)"`
Expected: `0.01`

---

### Task 2: alpha 更新后加 clamp（标准 SAC + TECRL 两处）

**Files:**
- Modify: `forest_vehicle_dqn/sac_agent.py:230,333`

**Step 1:** 在标准 SAC 的 `self.alpha_opt.step()`（L230）后加：

```python
        if self.cfg.alpha_min > 0:
            with torch.no_grad():
                self.log_alpha.data.clamp_(min=math.log(self.cfg.alpha_min))
```

**Step 2:** 在 TECRL 的 `self.alpha_opt.step()`（L333）后加同样的 clamp。

**Step 3:** 在文件顶部 import 区确认有 `import math`（如无则添加）。

**Step 4:** 验证

Run: `conda run -n ros2py310 python -c "from forest_vehicle_dqn.sac_agent import SACAgent, SACConfig; a=SACAgent(SACConfig(alpha_min=0.01)); print('OK')"`
Expected: `OK`

**Step 5:** Commit

```bash
git add forest_vehicle_dqn/sac_agent.py
git commit -m "feat(v8p4): add alpha_min floor to SACConfig and alpha update"
```

---

### Task 3: 日志 key 对齐（update_tecrl 返回 dict）

**Files:**
- Modify: `forest_vehicle_dqn/sac_agent.py:346-356`

**Step 1:** 在 `update_tecrl()` 返回 dict 中补充两个兼容 key：

```python
        return {
            "critic_loss": critic_loss.item(),
            "ent_critic_loss": ent_critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "q1_mean": q1_r.mean().item(),
            "q2_mean": q2_r.mean().item(),
            "q_e_mean": q_e_new.mean().item(),
            "target_q_r_mean": target_q_r.mean().item(),
            "target_q_mean": target_q_r.mean().item(),  # 兼容日志
            "target_q_max": target_q_r.max().item(),     # 兼容日志
            "reward_batch_mean": rewards_t.mean().item(),
        }
```

**Step 2:** Commit

```bash
git add forest_vehicle_dqn/sac_agent.py
git commit -m "fix(v8p4): align TECRL log keys with training loop"
```

---

### Task 4: 修正 PBRS 公式（env.py）

**Files:**
- Modify: `forest_vehicle_dqn/env.py:1361-1362`

**Step 1:** 将 L1361-1362 从：

```python
                reward += self.reward_c_prog * (
                    (phi_now + bias) - self._gamma * (phi_prev + bias))
```

改为标准 PBRS（γ 在新状态上）：

```python
                reward += self.reward_c_prog * (
                    self._gamma * (phi_now + bias) - (phi_prev + bias))
```

**Step 2:** Commit

```bash
git add forest_vehicle_dqn/env.py
git commit -m "fix(v8p4): correct PBRS formula (gamma on new state)"
```

---

### Task 5: train.py 新增 --sac-alpha-min CLI 参数并传递

**Files:**
- Modify: `forest_vehicle_dqn/cli/train.py:~3302,~2365,~2405,~4004`

**Step 1:** 在 CLI 参数区（`--sac-entropy-budget-ratio` 附近 ~L3302）新增：

```python
    ap.add_argument("--sac-alpha-min", type=float, default=0.0)
```

**Step 2:** 在 `train_one_sac()` 签名（~L2365）新增参数 `alpha_min: float = 0.0`

**Step 3:** 在 SACConfig 构造（~L2405）新增：

```python
        alpha_min=float(sac_cfg_dict.get("sac_alpha_min", alpha_min)),
```

**Step 4:** 在调用 `train_one_sac()` 处（~L4004）新增传参：

```python
                        alpha_min=float(getattr(args, "sac_alpha_min",
                                                sac_cfg_raw.get("sac_alpha_min", 0.0))),
```

**Step 5:** 验证 import

Run: `conda run -n ros2py310 python train.py --self-check`
Expected: PASS

**Step 6:** Commit

```bash
git add forest_vehicle_dqn/cli/train.py
git commit -m "feat(v8p4): add --sac-alpha-min CLI arg and pass to SACConfig"
```

---

### Task 6: 创建 configs/v8p4.json

**Files:**
- Create: `configs/v8p4.json`

**Step 1:** 复制 v8p3.json，修改以下字段：

- `"reward_potential_base": 4`（原 32）
- `"sac_entropy_budget_ratio": 2.0`（原 0.6）
- `"sac_alpha_min": 0.01`（新增）
- infer 区 `"models": "v8p4"`，`"out": "v8p4"`

**Step 2:** 验证 JSON 合法

Run: `conda run -n ros2py310 python -c "import json; json.load(open('configs/v8p4.json')); print('OK')"`
Expected: `OK`

**Step 3:** Commit

```bash
git add configs/v8p4.json
git commit -m "feat(v8p4): add v8p4 config (base=4, budget=2.0, alpha_min=0.01)"
```

---

### Task 7: self-check

Run:
```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```
Expected: 两个都 PASS

---

### Task 8: 远端同步 + smoke 训练（1000ep）

**Step 1:** 同步到远端

```bash
rsync -avz --exclude='runs/' --exclude='.git/' --exclude='__pycache__/' \
  /home/sun/phdproject/dqn/RL_sac/ ubuntu-zt:~/phdproject/dqn/RL_sac/
```

**Step 2:** 远端 smoke 训练

```bash
ssh ubuntu-zt "cd ~/phdproject/dqn/RL_sac && \
  conda run -n ros2py310 python train.py \
    --profile v8p4 --episodes 1000 --out v8p4-smoke1"
```

**Step 3:** 观察关键指标
- alpha 是否稳定在 ≥0.01
- reward 是否在合理范围（不再膨胀到百万级）
- tq 是否有非零值
- eval_return 趋势

---

### Task 9: smoke 推理（runs=3）

```bash
ssh ubuntu-zt "cd ~/phdproject/dqn/RL_sac && \
  conda run -n ros2py310 python infer.py \
    --profile v8p4 --runs 3 \
    --models runs/v8p4-smoke1/<train_dir> --out v8p4-smoke1"
```

回传结果：
```bash
rsync -avz ubuntu-zt:~/phdproject/dqn/RL_sac/runs/v8p4-smoke1/ \
  /home/sun/phdproject/dqn/RL_sac/runs/v8p4-smoke1/
```

---

### Task 10: 版本归档（四件套）

**Files:**
- Create: `docs/versions/v8p4/README.md`
- Create: `docs/versions/v8p4/CHANGES.md`
- Create: `docs/versions/v8p4/RESULTS.md`
- Create: `docs/versions/v8p4/runs/README.md`
- Modify: `docs/versions/README.md`（新增 v8p4 行）

Commit:
```bash
git add docs/versions/v8p4/ docs/versions/README.md
git commit -m "docs(v8p4): archive smoke results"
```
