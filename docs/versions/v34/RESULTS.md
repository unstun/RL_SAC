# V34 RESULTS — V32-B M-DQN+Shield 消融

## 对标（V22-D 全局最优，formal runs=20）

| 配置 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| V22-D negotiate K=10 | 0.95 | 15.75m | 1.00 | 46.11m |
| A*-MPC | 0.95 | 15.77m | 1.00 | 42.93m |
| §13 目标 | ≥0.95 | <15.77m | =1.00 | <42.93m |

## V34-C: V32-B M-DQN+shield 单模型（formal runs=20, H=45）

| 指标 | V34-C | V22-D K=10 | A\*-MPC |
|------|-------|-----------|---------|
| short SR | 0.90 | **0.95** | 0.95 |
| short path | 16.51m | **15.75m** | 15.44m |
| long SR | **1.00** | 1.00 | 1.00 |
| long path | 46.22m | **46.11m** | 42.91m |
| short inad | 0.312 | 0.263 | — |
| long inad | **0.228** | 0.316 | — |

- short SR=0.90 < 0.95，§13 不达标
- long inad 从 31.6% → 22.8%（shield 有效降低 inad）
- 单模型不如 V22-D ensemble

## V34-D: V16-C DDQN + V32-B M-DQN+shield negotiate K=10（formal runs=20, H=45）

| 指标 | V34-D | V22-D K=10 | A\*-MPC |
|------|-------|-----------|---------|
| short SR | 0.80 | **0.95** | 0.95 |
| **short path** | **15.00m** | 15.75m | 15.44m |
| long SR | **1.00** | 1.00 | 1.00 |
| **long path** | **45.28m** | 46.11m | 42.91m |

- 路径显著缩短：short -0.75m, long -0.83m
- short SR 劣化到 0.80，§13 不达标
- trade-off：shield M-DQN approver 选更短路但安全裕量不足

## 总结

V34-D 路径质量优于 V22-D，但 short SR 代价过高（0.80）。
V22-D negotiate K=10 仍是最优综合配置（§13 通过 5/6 条件）。

K sweep（更小 K）可能改善 SR，值得进一步探索。
