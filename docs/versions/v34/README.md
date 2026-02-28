# V34 — V32-B M-DQN+Shield 消融（方向 C + D）

## 动机

V32 实验在 smoke (runs=3) 时发现：
- V32-B M-DQN+shield: inad 从 30% 降到 16%（M-DQN 支持训练时动作约束）
- V32 ensemble (V16-C + V32-B, negotiate K=10): short SR=1.00（优于 V22-D 的 0.95）
- V32 当时判定为"负面结果"（路径无改善），未进 formal

V34 补充 formal runs=20 以：
1. 确认 V32-D ensemble 的 short SR 提升是否稳健
2. 彻底量化 M-DQN+shield 贡献（单模型 + ensemble）

## 消融矩阵

| 实验 | 模型 | 模式 | 目标 |
|------|------|------|------|
| V34-C | V32-B M-DQN+shield | 单模型 | 量化 shield 单模型性能 |
| V34-D | V16-C DDQN + V32-B M-DQN+shield | negotiate K=10 | 确认 short SR=1.00 是否稳健 |

## 模型路径

| 模型 | 路径 |
|------|------|
| V16-C DDQN | `runs/v16-C-ep190/` |
| V32-B M-DQN+shield | `runs/v32-B-mdqn-shield/train_20260227_225149/` |

## 评测口径

- formal：runs=20，rand-two-suites（short+long），forest_a
- seed=42，device=cuda

## 结论

见 [RESULTS.md](RESULTS.md)
