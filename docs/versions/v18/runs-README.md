# V18 Runs 目录

## 训练 Run

| Run | 配置 | 说明 |
| --- | --- | --- |
| v16-C-r3/train_20260224_235346 | H45, k_p=12, k_len=0.10 | 本地 GPU, ep250 early-stop, ~34min |

## Smoke 推理 Runs (runs=5, seed=42)

| Run | ckpt | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v16-C-r3-infer-ep120 | ep120 | - | - |
| v16-C-r3-infer-ep140 | ep140 | 1.0/13.40m | 1.0/46.35m |
| v16-C-r3-infer-ep220 | ep220 | - | - |
| v16-C-r3-infer-ep230 | ep230 | - | - |
| v16-C-r3-infer-ep240 | ep240 | 1.0/13.09m | 1.0/49.42m |
| v16-C-r3-infer-ep250 | ep250 | - | - |

## Runs=10 验证 (seed=42)

| Run | ckpt | short SR/path | long SR/path |
| --- | --- | --- | --- |
| v16-C-r3-infer-ep140-r10 | ep140 | 1.0/20.26m | 1.0/46.65m |
| v16-C-r3-infer-ep240-r10 | ep240 | 1.0/16.26m | 1.0/47.89m |
| v16-C-r3-infer-ep250-r10 | ep250 | 0.8/16.14m | 1.0/46.52m |

## Formal 评测 (runs=20, seed=77)

| Run | ckpt | short SR/path | long SR/path |
| --- | --- | --- | --- |
| **v16-C-r3-formal-ep240** | **ep240** | **0.85/15.34m** | **1.00/48.17m** |

## 独立模型目录

| 路径 | 说明 |
| --- | --- |
| v16-C-r3-ep240/models/forest_a/cnn-ddqn.pt | ep240 standalone 模型 |

## 结论

R3 成功复现 long SR=1.00，但路径长度略差于 V16-C ep190。
V16-C ep190（远端训练）仍为全局最优模型。
