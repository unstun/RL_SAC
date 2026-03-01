# V20 CHANGES — 相对 V16/V19 的改动

## 代码改动

### 1. agents.py — M-DQN 训练算法

- `AlgoBase` 扩展: `Literal["dqn", "ddqn"]` → `Literal["dqn", "ddqn", "mdqn"]`
- `parse_rl_algo()`: 支持 `cnn-mdqn` / `mlp-mdqn`
- `AgentConfig`: 新增 3 个 M-DQN 超参 (`mdqn_tau=0.03`, `mdqn_alpha=0.9`, `mdqn_l0=-1.0`)
- `update()` legacy 模式: 新增 `elif self.base_algo == "mdqn":` 分支
  - Soft value: V(s') = Σ_a π(a|s')[Q_tgt(s',a) - τ log π(a|s')]
  - Log-policy augmentation: target += α·τ·log π(a|s)
- `update()` DQfD 模式: 同上，在 `_scalar_next_q()` 和 target 计算中新增 M-DQN 分支

### 2. train.py — M-DQN 训练支持

- `canonical_all` 新增 mdqn
- `algo_cfgs` / `algo_labels` / `pref` 新增 mdqn 映射

### 3. infer.py — 跨 algo ensemble 加载

- `_ENS_ALGOS = ["cnn-dqn", "cnn-ddqn", "cnn-mdqn", "cnn-pddqn"]`
- ensemble 加载: 遍历 `_ENS_ALGOS` 自动搜索每个 ensemble dir 中存在的模型
- 所有 ensemble 模型映射到 primary algo key 参与 Q-value averaging

## 训练参数（与 V16-C 完全相同）

| 参数 | 值 |
|------|-----|
| `reward_k_len` | 0.10 |
| `reward_k_p` | 12.0 |
| `forest_adm_horizon` | 45 |
| `eps_decay` | 100 |
| `episodes` | 300 |
| `--dueling --mha` | 启用 |

M-DQN 额外超参（NeurIPS 2020 推荐值）: `tau=0.03, alpha=0.9, l0=-1.0`
