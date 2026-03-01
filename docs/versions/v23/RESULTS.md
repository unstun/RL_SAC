# V23 RESULTS — Negotiate 融合模式消融 (FAILED)

## Smoke (runs=5)

| 配置 | mode | 参数 | short SR | short path | long SR | long path |
|------|------|------|----------|-----------|---------|-----------|
| R1 | negrelax | K=10/20 | 1.0 | 16.43m | 1.0 | 45.53m |
| R2 | negrelax | K=10/30 | 1.0 | 16.43m | 1.0 | 45.53m |
| R3 | negrelax | K=8/25 | 1.0 | 16.85m | 1.0 | 45.99m |
| S1 | negsoft | α=0.7,τ=0.1 | 1.0 | 17.25m | 1.0 | 45.80m |
| S2 | negsoft | α=0.5,τ=0.1 | 0.8 | — | 1.0 | — |
| S3 | negsoft | α=0.7,τ=0.03 | 0.8 | — | 0.8 | — |

R1/R2 与 V22-D negotiate K=10 smoke 完全相同。
S2/S3 short SR < 0.90 不达标。

## Formal (runs=20)

| 配置 | short SR | short path | long SR | long path | inadm |
|------|----------|-----------|---------|-----------|-------|
| V22-D negotiate K=10 | 0.95 | **15.75m** | 1.00 | 46.11m | 0.316 |
| R2 negrelax K=10/30 | 0.95 | 15.75m | 1.00 | 46.11m | 0.316 |
| S1 negsoft α=0.7 | 0.95 | 15.85m | 1.00 | 45.96m | 0.324 |
| A*-MPC | 0.95 | 15.77m | 1.00 | 42.93m | — |

## 分析

### negrelax 为何无效

negotiate K=10 在 225 动作空间中，DDQN top-10 和 M-DQN top-10 **总有交集**。
20 runs × 200+ steps/run = 4000+ 步，每步都能在 K=10 找到共识。
松弛根本不触发 → 结果与 negotiate K=10 完全相同。

### negsoft 为何无效

软投票消除了硬否决但引入新问题：
- M-DQN softmax 高度集中 → 少数动作得分极高
- 加权后 argmax 偏向 M-DQN 的极端偏好 → 路径拐角 0→36
- long path 微减 0.15m 不足以补偿 short path 退化

### 真正的瓶颈

**argmax_inadmissible_rate ≈ 30%** — 融合后的最优动作仍有 30% 时间违反
admissibility 约束（碰撞/停滞预测），被迫 top-K/mask 替代 → 绕路。

这不是融合算法能解决的：无论 negotiate/negrelax/negsoft，
融合后的动作都要过同一个 admissibility gate。

## 最终结论

1. **negrelax 完全无效**: K=10 在 225 动作空间内总能达成共识
2. **negsoft 微弱改善 long (-0.15m) 但 short 退化 (+0.10m)**
3. **融合算法不是 long path 的瓶颈**
4. **瓶颈在 admissibility fallback**: 需改进 fallback 质量或模型本身
5. V22-D negotiate K=10 仍是最优配置
