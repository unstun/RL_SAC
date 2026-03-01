# V23 Runs 目录索引

## 模型来源（与 V22-D 相同）

| 模型 | 路径 |
|------|------|
| V16-C DDQN | `runs/v16-C-ep190/` |
| V22-D M-DQN | `runs/v22-D-mdqn-H45-kl030/train_20260227_023501/` |

## Smoke (runs=5)

| 目录 | 配置 | 结论 |
|------|------|------|
| `v23-R1-smoke/` | negrelax K=10/20 | SR=1.0/1.0, =baseline |
| `v23-R2-smoke/` | negrelax K=10/30 | SR=1.0/1.0, =baseline |
| `v23-R3-smoke/` | negrelax K=8/25 | SR=1.0/1.0, 稍差 |
| `v23-S1-smoke/` | negsoft α=0.7,τ=0.1 | SR=1.0/1.0 |
| `v23-S2-smoke/` | negsoft α=0.5,τ=0.1 | short SR=0.8 FAIL |
| `v23-S3-smoke/` | negsoft α=0.7,τ=0.03 | SR=0.8/0.8 FAIL |

## Formal (runs=20)

| 目录 | 配置 | 结论 |
|------|------|------|
| `v23-R2-formal/` | negrelax K=10/30 | **=V22-D baseline 完全相同** |
| `v23-S1-formal/` | negsoft α=0.7,τ=0.1 | long -0.15m, short +0.10m |

## 结论

负面结果。融合算法改进不影响 long path，瓶颈在 admissibility fallback。
