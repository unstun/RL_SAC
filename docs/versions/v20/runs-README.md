# V20 Runs 目录

## 训练 Runs

| Run | 算法 | 说明 |
| --- | --- | --- |
| v20-dqn/train_20260225_032621 | cnn-dqn | ep200 early-stop, ep130 仅 1 次 all=1.0 |
| v20-mdqn/train_20260225_041530 | cnn-mdqn | ep290 early-stop, 7 次 all=1.0 |

## 独立模型目录

| 路径 | 说明 |
| --- | --- |
| v20-dqn-ep130/models/forest_a/cnn-dqn.pt | DQN ep130 standalone |
| v20-mdqn-ep170/models/forest_a/cnn-mdqn.pt | M-DQN ep170 standalone |
| v16-C-ep190/models/forest_a/cnn-ddqn.pt | DDQN (V16-C) 主模型 |

## Smoke 推理 Runs (runs=5, seed=42)

| Run | 配置 | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v20-dqn-ep130-smoke/20260225_045920 | DQN 单模型 | 0.8/18.80m | 0.6/43.07m |
| v20-mdqn-ep170-smoke/20260225_050025 | M-DQN 单模型 | 1.0/17.10m | 1.0/46.23m |
| v20-ens3-smoke/20260225_050145 | 3-ens (DQN+DDQN+M-DQN) | 0.8/14.33m | 1.0/44.27m |
| v20-ens2-ddqn-mdqn-smoke/20260225_050258 | 2-ens (DDQN+M-DQN) | 0.8/12.71m | 1.0/47.72m |

## Formal 推理 Runs (runs=20, seed=77)

| Run | 配置 | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v20-ens3-conf-r20/20260225_054741 | 3-ens conf | 0.80/14.34m | 0.90/48.35m |
| v20-ens2-pavg-r20/20260225_055316 | 2-ens pavg | 0.80/14.59m | 0.95/47.08m |
| v20-ens3-qavg-r20/20260225_055823 | 3-ens qavg | 0.80/15.30m | 0.90/46.81m |
| v20-ens2-qavg-r20/20260225_060445 | 2-ens qavg | 0.90/14.75m | 0.90/47.06m |
| **v20-ens2-conf-r20/20260225_061006** | **2-ens conf** | **0.90/14.47m** | **0.95/46.43m** |

## Phase 2 Smoke (runs=5, seed=42)

| Run | 配置 | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v20-ens2-rank-smoke/20260225_065442 | 2-ens rank | 1.0/12.80m | 1.0/45.17m |
| v20-ens2-qgap-smoke/20260225_065532 | 2-ens qgap | 0.8/12.24m | 1.0/45.43m |
| v20-ens2-veto-smoke/20260225_065627 | 2-ens veto | 1.0/13.50m | 0.8/44.56m |
| v20-ens2-topk-smoke/20260225_065713 | 2-ens topk | 1.0/13.52m | 0.8/43.85m |
| v20-ens2-negotiate-smoke2/20260225_071431 | 2-ens negotiate K=20 | 1.0/13.90m | 1.0/45.38m |

## Phase 2 Formal (runs=20, seed=77)

| Run | 配置 | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v20-single-ddqn-r20/20260225_070535 | V16-C 单模型 | 0.85/14.43m | 1.00/46.00m |
| v20-ens2-rank-r20/20260225_065838 | 2-ens rank | 0.90/14.93m | 0.90/45.48m |
| v20-ens2-negotiate-k10-r20/20260225_072535 | negotiate K=10 | 0.95/14.34m | 0.95/45.49m |
| v20-ens2-negotiate-k11-r20/20260225_073808 | negotiate K=11 | 0.90/14.69m | 1.00/45.12m |
| v20-ens2-negotiate-k15-r20/20260225_073001 | negotiate K=15 | 0.85/14.70m | 1.00/45.65m |
| **v20-ens2-negotiate-r20/20260225_071539** | **negotiate K=20** | **0.95/14.55m** | **0.95/46.09m** |
| v20-ens2-negotiate-k30-r20/20260225_072140 | negotiate K=30 | 0.85/14.82m | 0.95/45.61m |

## 结论

negotiate K=20 为全局最优（short SR=0.95 历史最高，总失败仅 2 个）。
§13 仍未通过，但差距大幅缩小（short SR: 0.85->0.95, path gap 仅 0.35m）。
