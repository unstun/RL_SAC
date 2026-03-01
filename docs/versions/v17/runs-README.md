# V17 Runs 目录

## 训练 Runs

| Run | 配置 | 说明 |
| --- | --- | --- |
| v17/train_20260224_194911 | H45, k_p=12, k_len=0.10 | R1: 默认 early-stop, ep210 停 |
| v17/train_20260224_223429 | 同上 + patience=999 | R2: 无 early-stop, 跑满 300 ep |

## R1 Smoke 推理 Runs (runs=5, seed=42)

| Run | ckpt | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v17-infer-best | best | 0.8/15.22m | 0.0/- |
| v17-infer-ep040 | ep40 | 0.6/14.58m | 0.6/50.95m |
| v17-infer-ep050 | ep50 | 0.8/13.71m | 0.8/52.34m |
| v17-infer-ep060 | ep60 | 1.0/15.16m | 0.6/48.89m |
| v17-infer-ep070 | ep70 | 1.0/14.26m | 0.8/55.71m |
| **v17-infer-ep080** | **ep80** | **0.8/12.08m** | **0.8/50.48m** |
| v17-infer-ep090 | ep90 | 1.0/14.74m | 0.4/49.94m |
| v17-infer-ep100 | ep100 | 1.0/13.19m | 0.6/55.22m |
| v17-infer-ep120 | ep120 | 1.0/14.35m | 0.4/52.19m |
| v17-infer-ep150 | ep150 | 1.0/12.91m | 0.6/51.33m |
| v17-infer-ep190 | ep190 | 0.8/13.26m | 0.6/51.71m |
| v17-infer-ep210 | ep210 | 1.0/13.22m | 0.6/51.00m |

## R2 Smoke 推理 Runs (runs=5, seed=42, 无 early-stop)

| Run | ckpt | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v17r2-infer-ep080 | ep80 | 1.0/14.05m | 0.6/50.65m |
| **v17r2-infer-ep200** | **ep200** | **1.0/22.40m** | **1.0/47.17m** |
| v17r2-infer-ep260 | ep260 | 1.0/14.05m | 0.6/62.31m |
| v17r2-infer-ep300 | ep300 | 1.0/14.54m | 0.8/52.22m |

## 结论

未进入 runs=20 正式评测。V16-C ep190 仍为最优。
