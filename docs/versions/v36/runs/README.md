# V36 Run 路径

## 训练 Runs

| 实验 | 路径 | 说明 |
| ---- | ---- | ---- |
| V36-B train | `runs/v36-B-dual-smoke/train_20260227_211404` | cnn-dual-ddqn 150ep（demo禁用版） |
| V36-B2 train | `runs/v36-B2-dual-smoke/train_20260227_220457` | cnn-dual-ddqn 150ep（218-dim demo修复版，8m57s） |
| V36-B3 train | `runs/v36-B3-dual-smoke/train_20260227_231634` | cnn-dual-ddqn 150ep（16x16+GAP，obs_dim=410，8m06s） |
| V36-A train | `runs/v36-A-drqn-smoke/train_20260227_212056` | cnn-drqn 150ep |
| V36-C train | `runs/v36-C-histformer-smoke/train_20260227_212639` | cnn-histformer 150ep |

## 推理 Runs

| 实验 | 路径 | 说明 |
| ---- | ---- | ---- |
| V36-B infer | `runs/v36-B-dual-infer/20260227_211942` | cnn-dual-ddqn smoke runs=3 |
| V36-B2 infer | `runs/v36-B2-dual-infer/20260227_221508` | cnn-dual-ddqn smoke runs=3（demo修复版） |
| V36-B3 infer | `runs/v36-B3-dual-infer/20260227_232519` | cnn-dual-ddqn smoke runs=3（16x16+GAP，SR=0） |
| V36-A infer | `runs/v36-A-drqn-infer/20260227_212545` | cnn-drqn smoke runs=3 |
| V36-C infer | `runs/v36-C-histformer-infer/20260227_212945` | cnn-histformer smoke runs=3 |

## 评测口径

- smoke 阶段：`--rand-two-suites --runs 3 --seed 77`
- 三种架构均在 forest_a 单环境评测
- 基线 Hybrid A*-MPC 作为对比
