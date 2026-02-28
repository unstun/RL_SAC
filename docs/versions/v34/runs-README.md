# V34 Runs 索引

## 模型来源

| 模型 | 路径 |
|------|------|
| V16-C DDQN | `runs/v16-C-ep190/` |
| V32-B M-DQN+shield | `runs/v32-B-mdqn-shield/train_20260227_225149/` |

## V34-C：V32-B 单模型 formal

| 目录 | 配置 | 结论 |
|------|------|------|
| `v34-C-formal-h45/20260227_191458` | V32-B cnn-mdqn, H=45, runs=20 | short 0.90/16.51m, long 1.00/46.22m |

## V34-D：negotiate K=10 formal

| 目录 | 配置 | 结论 |
|------|------|------|
| `v34-D-neg-k10-formal/20260227_191721` | V16-C+V32-B negotiate K=10, H=45, runs=20 | short **0.80**/15.00m, long 1.00/**45.28m** |

## 注意

- V34-C 第一次跑（v34-C-formal/）误用 H=30，结果无效（short SR=0.85），已废弃
- 正确结果均使用 `--forest-adm-horizon 45`（与训练时一致）
