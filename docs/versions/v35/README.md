# V35 — 进一步缩短路径（V34-D 基础上消融）

## 动机

V34-D 发现 V32-B shield M-DQN 作为 approver 使路径缩短（-0.83m long），
但 short SR 代价过高（0.80 < 0.95）。

V35 在此基础上进一步探索更短路径的可行配置，分两个方向：

| 方向 | 描述 | 成本 |
|------|------|------|
| **V35-A** | K 上扫（K=15/20），V16-C + V32-B shield | 零训练 |
| **V35-B** | 角色互换：V22-D M-DQN 提议 + V32-B shield 过滤 | 零训练 |

## 核心假设

- V35-A：更大 K → shield approver 有更多候选 → 选出更短路径（SR 可能进一步下降）
- V35-B：V22-D M-DQN（k_len=0.30）提议路径优化候选集，V32-B 从中安全过滤；
  预期：比 DDQN 提议更优化的候选集 → 更短路径

## 模型路径

| 模型 | 路径 |
|------|------|
| V16-C DDQN | `runs/v16-C-ep190/` |
| V22-D M-DQN | `runs/v22-D-mdqn-H45-kl030/train_20260227_023501/` |
| V32-B M-DQN+shield | `runs/v32-B-mdqn-shield/train_20260227_225149/` |

## 评测口径

- smoke：runs=3，rand-two-suites，forest_a，H=45，seed=77
- formal：runs=20，rand-two-suites，forest_a，H=45，seed=42

## 结论

见 [RESULTS.md](RESULTS.md)
