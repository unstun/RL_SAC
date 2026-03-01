# V22 Runs 目录索引

## 模型来源

| 模型 | 路径 | 说明 |
|------|------|------|
| V16-C DDQN | `runs/v16-C-ep190/` | H=45, k_len=0.10, ep190 |
| V20 M-DQN | `runs/v20-mdqn/train_20260225_041530/` | H=45, k_len=0.10, ep170 |
| **V22-D M-DQN** | `runs/v22-D-mdqn-H45-kl030/train_20260227_023501/` | H=45, k_len=0.30, ep300 |

## V22-A: K Sweep（推理消融, V16-C + V20 M-DQN negotiate）

### Smoke (runs=5)

| 目录 | K | 结论 |
|------|---|------|
| `v22-A-K05-smoke/` | 5 | SR=1.0/1.0, path=17.99m/45.81m |
| `v22-A-K08-smoke/` | 8 | SR=1.0/1.0, path=17.69m/45.59m |
| `v22-A-K10-smoke/` | 10 | SR=1.0/1.0, path=17.47m/44.44m |
| `v22-A-K12-smoke/` | 12 | SR=1.0/1.0, path=16.56m/46.72m |
| `v22-A-K15-smoke/` | 15 | SR=0.8/0.8 |
| `v22-A-K25-smoke/` | 25 | SR=1.0/1.0, path=17.72m/44.41m |
| `v22-A-K30-smoke/` | 30 | SR=0.8/1.0 |

### Formal (runs=20, top-3)

| 目录 | K | 结论 |
|------|---|------|
| `v22-A-K10-formal/` | **10** | **SR=1.00/0.95**, path=16.58m/45.17m |
| `v22-A-K12-formal/` | 12 | SR=0.90/0.90, path=15.44m/45.86m |
| `v22-A-K25-formal/` | 25 | SR=1.00/0.85, path=16.38m/45.44m |

## V22-B: PBRS Training（FAILED）

| 目录 | 配置 | 结论 |
|------|------|------|
| `v22-B1-pbrs-linear/` | c_prog=12, base=0 (linear) | 训练 OK, smoke 0.6/0.4 |
| `v22-B1-smoke/` | B1 smoke | SR 严重退化 |
| `v22-B2-pbrs-exp3/` | c_prog=12, base=3 (exp) | 训练 OK, smoke 0.8/0.6 |
| `v22-B2-smoke/` | B2 smoke | SR 退化 |

## V22-C: Enhanced DQfD（FAILED）

| 目录 | 配置 | 结论 |
|------|------|------|
| `v22-C1-dqfd-lam16/` | demo_lambda=16 | early-stop ep150 |
| `v22-C1-smoke/` | C1 smoke | SR=0.6/0.6 |

## V22-D: M-DQN H=45 + k_len=0.30 Negotiate Ensemble

| 目录 | 配置 | 结论 |
|------|------|------|
| `v22-D-mdqn-H45-kl030/` | 训练 (300ep) | best_ratio=10.233, 最稳定 |
| `v22-D-smoke/` | 单模型 smoke | SR=0.8/0.6 (单模型弱) |
| `v22-D-ens-K10-smoke/` | negotiate K=10 smoke | SR=1.0/1.0 |
| `v22-D-ens-K20-smoke/` | negotiate K=20 smoke | SR=1.0/1.0 |
| `v22-D-ens-K10-formal/` | **negotiate K=10 formal** | **SR=0.95/1.00** ★ |
| `v22-D-ens-K20-formal/` | negotiate K=20 formal | SR=1.00/0.90 |

## 最优结果

**V22-D negotiate K=10 formal** (`v22-D-ens-K10-formal/20260226_090348/`)
- short: SR=0.95, path=15.75m, compute=0.71s
- long: SR=1.00, path=46.11m, compute=1.69s
- §13: 通过 3/6 条件（首次 long SR=1.00 + short path < A*）
