# V35 RESULTS — 进一步缩短路径消融

## 对标

| 配置 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| V22-D K=10（全局最优） | 0.95 | 15.75m | 1.00 | 46.11m |
| V34-D K=10（shield approver） | 0.80 | 15.00m | 1.00 | 45.28m |
| A*-MPC | 0.95 | 15.44m | 1.00 | 42.93m |
| §13 目标 | ≥0.95 | <15.77m | =1.00 | <42.93m |

## V35-A：K 上扫（V16-C DDQN + V32-B shield）smoke runs=3, seed=77

| K | short SR | short path | long SR | long path | 备注 |
|---|----------|-----------|---------|-----------|------|
| 15 | 1.00 | 15.11m | 1.00 | 47.61m | long 恶化 |
| **20** | **1.00** | **14.02m** | **1.00** | **47.48m** | short 极短！long 仍恶化 |

- K=20 short path=14.02m 为所有配置最短（含 A\*-MPC 15.44m）
- long path 47.5m 劣于 V34-D K=10 (45.28m)：K 增大后退化为 DDQN 的 long path 特性
- Argmax inad rate: K=15 约 0.328，K=20 约 0.394（DDQN 不变）

## V35-B：角色互换（V22-D M-DQN 提议 + V32-B shield 审批）smoke runs=3, seed=77

| K | short SR | short path | long SR | long path | 备注 |
|---|----------|-----------|---------|-----------|------|
| 10 | 1.00 | 12.88m | 0.333 | 53.56m | long 崩溃 |
| 15 | 0.667 | 13.30m | 0.000 | NaN | 完全失败 |

**V35-B 完全失败**：V22-D M-DQN inad_rate=0.75（long）作为提议者质量差，
提议的候选集多为 inadmissible，V32-B 审批无法修复 → long SR 崩溃。

## Formal

V35-A K=20 待定（short path 亮眼但 long path 恶化，是否进 formal 需判断）。

## 总结

- **V35-A K=20 smoke**：short path 14.02m 创历史新低，但 long path 47.5m 劣于所有基准
- **V35-B 完全失败**：V22-D M-DQN 不适合作 proposer（inad 率过高）
- K 上扫对 short 路径有效，但对 long 路径负面效应明显
- V34-D K=10（formal）仍是 long path 最优配置（45.28m）
- **§13 长路径缺口（42.93m 目标）需要训练层面改动**，纯推理消融已接近上限
