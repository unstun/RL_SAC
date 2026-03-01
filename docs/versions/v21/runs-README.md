# V21 Runs 目录

## 训练 Runs（远端 ubuntu-zt）

| Run | 实验 | H | k_len | best_sr_all | 说明 |
| --- | --- | --- | --- | --- | --- |
| v21-mdqn-H60-kl020/train_20260226_193302 | A | 60 | 0.20 | ~0.7 | 弱 |
| v21-mdqn-H60-kl030/train_20260226_195008 | B R1 | 60 | 0.30 | 0.7 | 弱 |
| **v21-mdqn-H60-kl030/train_20260226_234042** | **B R2** | **60** | **0.30** | **1.000** | **最佳** |
| v21-mdqn-H80-kl020/train_20260226_234151 | C | 80 | 0.20 | 0.800 | H=80 不稳定 |
| v21-mdqn-H80-kl030/train_20260226_204224 | D old | 80 | 0.30 | 0.700 | 失败 |
| v21-mdqn-H80-kl030/train_20260227_002828 | D new | 80 | 0.30 | 0.700 | 失败 |

## Smoke 推理 Runs (runs=5, H=60)

| Run | 配置 | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v21-smoke-B2-ep150-H60/20260226_063833 | B R2 ep150 含 mid | 1.0/16.66m | 1.0/50.50m |
| v21-smoke-B2-ep120/20260226_063534 | B R2 ep120 (错误 H=15) | 0.4/19.80m | 0.0/- |

## Formal 推理 Runs (runs=20, H=60)

| Run | 配置 | short SR/path | long SR/path |
| --- | --- | --- | --- |
| **v21-formal-B2-ep150-H60/20260226_064458** | **M-DQN 单模型** | **1.00/16.48m** | **0.85/51.03m** |

## Ensemble Runs (negotiate K=20, H=60)

| Run | 配置 | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v21-ens-negotiate-K20-smoke/20260226_065159 | smoke runs=5 | 1.0/16.64m | 1.0/48.38m |
| **v21-ens-negotiate-K20-formal/20260226_065334** | **formal runs=20** | **0.95/16.08m** | **0.90/46.64m** |

## 独立模型目录

| 路径 | 说明 |
| --- | --- |
| v16-C-ep190/models/forest_a/cnn-ddqn.pt | DDQN (V16-C) 主模型, H=45 |
| v21-mdqn-H60-kl030/train_20260226_234042/models/forest_a/cnn-mdqn.pt | M-DQN B R2 ep150, H=60 |

## 结论

单 M-DQN short SR=1.00 首次超越 A*-MPC (0.95)，里程碑进展。
Negotiate ensemble 提升 long SR (0.85→0.90) 但降低 short SR (1.00→0.95)。
§13 仍未通过: long SR < 1.0, path > A*-MPC。
