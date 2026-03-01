# V24 Runs 目录索引

## 模型来源（与 V22-D/V23 相同）

| 模型 | 路径 |
|------|------|
| V16-C DDQN | `runs/v16-C-ep190/` |
| V22-D M-DQN | `runs/v22-D-mdqn-H45-kl030/train_20260227_023501/` |

## Smoke (runs=5)

| 目录 | 配置 | 结论 |
|------|------|------|
| `v24-P1-smoke/` | prefilter + negotiate K=10 | SR=1.0/1.0, inadm=0.000 |

## Formal (runs=20)

| 目录 | 配置 | 结论 |
|------|------|------|
| `v24-P1-formal/` | prefilter + negotiate K=10 | long +1.46m FAIL |

## 结论

负面结果。预过滤消除 inadmissible 但路径更长，V22-D baseline 仍最优。
