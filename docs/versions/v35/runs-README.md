# V35 Runs 索引

## 模型来源

| 模型 | 路径 |
|------|------|
| V16-C DDQN | `runs/v16-C-ep190/` |
| V22-D M-DQN | `runs/v22-D-mdqn-H45-kl030/train_20260227_023501/` |
| V32-B M-DQN+shield | `runs/v32-B-mdqn-shield/train_20260227_225149/` |

## V35-A：K 上扫 smoke（runs=3, seed=77）

| 目录 | 配置 | 结论 |
|------|------|------|
| `v35-A-neg-k15-smoke/20260227_200647` | V16-C+V32-B shield, K=15 | short 1.00/15.11m, long 1.00/47.61m |
| `v35-A-neg-k20-smoke/20260227_200651` | V16-C+V32-B shield, K=20 | short 1.00/**14.02m**, long 1.00/47.48m |

## V35-B：角色互换 smoke（runs=3, seed=77）

| 目录 | 配置 | 结论 |
|------|------|------|
| `v35-B-neg-k10-smoke/20260227_200804` | V22-D+V32-B shield, K=10 | short 1.00/12.88m, long **0.333**/53.56m |
| `v35-B-neg-k15-smoke/20260227_200809` | V22-D+V32-B shield, K=15 | short 0.667, long **0.000** |

## 注意

- V35-B 完全失败：V22-D M-DQN inad_rate=0.75 不适合作 proposer
- V35-A K=20 short path=14.02m 亮眼，但 long path 47.5m 劣于 V34-D 基准（45.28m）
- Formal 待定：long path 恶化问题需权衡
