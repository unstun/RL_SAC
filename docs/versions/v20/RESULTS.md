# V20 RESULTS — 异构 DQN Ensemble + 融合模式消融

## 训练结果

| 算法 | early-stop | all=1.0 epochs | 选用 ckpt |
|------|-----------|---------------|----------|
| DQN | ep200 (patience=6) | ep130 (仅1次) | ep130 |
| M-DQN | ep290 (patience=6) | ep30/50/110/150/170/210/270 (7次) | ep170 |
| DDQN | V16-C 已有 | - | ep190 |

M-DQN ep170 选择依据: ratio(short=9.606, long=10.590) 最低 + inad=0.132 最低。

## 单模型 Smoke (runs=5, seed=42)

| 模型 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| DDQN V16-C ep190 | 1.0 | 13.24m | 1.0 | 45.99m |
| DQN ep130 | 0.8 | 18.80m | 0.6 | 43.07m |
| M-DQN ep170 | 1.0 | 17.10m | 1.0 | 46.23m |
| A*-MPC 基线 | 1.0 | 13.46m | 1.0 | 42.82m |

DQN 模型质量最弱（long SR=0.6），M-DQN 路径较长但 SR=1.0。

## Ensemble Smoke (runs=5, seed=42)

| 配置 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| 3-ens (DQN+DDQN+M-DQN) | 0.8 | 14.33m | 1.0 | **44.27m** |
| 2-ens (DDQN+M-DQN) | 0.8 | **12.71m** | 1.0 | 47.72m |
| V19 同构 2-ens (DDQN×2) | 1.0 | 17.68m | 1.0 | 53.22m |

## 关键发现

1. **异构 ensemble 远好于同构**: V20 3-ens long path=44.27m vs V19 同构 53.22m
2. **2-ens (DDQN+M-DQN) short path=12.71m 首次短于 A* (13.46m)**
3. **Ensemble 降低 short SR**: 两种 ensemble 的 short SR 均为 0.8，DDQN 单模型为 1.0
4. **DQN 模型太弱**: long SR=0.6，拖累 3-model ensemble 但对 long path 有正面贡献
5. **M-DQN 训练稳定性好**: 7 次 all=1.0（vs DQN 仅 1 次），但路径较长

## §13 门槛检查（Smoke 口径，非 formal）

| 条件 | 要求 | 3-ens | 2-ens | 通过? |
|------|------|-------|-------|-------|
| SR(short) ≥ A* | ≥1.00 | 0.8 | 0.8 | **否** |
| SR(long) ≥ A* | ≥1.00 | 1.0 | 1.0 | 是 |
| path(short) < A* | <13.46m | 14.33m | 12.71m | 2-ens **是** |
| path(long) < A* | <42.82m | 44.27m | 47.72m | **否** |

**结论**: Smoke 未通过 §13，不进入 formal 评测。方向正确但模型质量不足。

## Ensemble 融合模式消融 Smoke (runs=5, seed=42)

| 配置 | 模式 | short SR | short path | long SR | long path |
|------|------|----------|-----------|---------|-----------|
| 3-ens | qavg | 0.8 | 14.33m | 1.0 | 44.27m |
| 3-ens | pavg | 0.8 | 13.81m | 0.8 | 47.81m |
| 3-ens | vote | 0.8 | 12.31m | 0.0 | - |
| 3-ens | conf | 1.0 | 14.28m | 0.8 | 46.54m |
| 2-ens | qavg | 0.8 | 12.71m | 1.0 | 47.72m |
| 2-ens | pavg | 1.0 | 14.45m | 0.8 | 45.04m |
| 2-ens | conf | 0.8 | 12.27m | 1.0 | 46.39m |

## Formal 评测 (runs=20, seed=77)

| 配置 | 模式 | short SR | short path | long SR | long path |
|------|------|----------|-----------|---------|-----------|
| DDQN 单模型 V16-C | - | 0.85 | 14.43m | 1.00 | 46.00m |
| 3-ens | conf | 0.80 | 14.34m | 0.90 | 48.35m |
| 2-ens | pavg | 0.80 | 14.59m | 0.95 | 47.08m |
| 3-ens | qavg | 0.80 | 15.30m | 0.90 | 46.81m |
| 2-ens | qavg | 0.90 | 14.75m | 0.90 | 47.06m |
| **2-ens** | **conf** | **0.90** | **14.47m** | **0.95** | **46.43m** |
| A*-MPC | - | 1.00 | 14.20m | 1.00 | 42.98m |

## §13 门槛检查（Formal, 2-ens conf — ensemble 最优）

| 条件 | 要求 | 2-ens conf | 通过? |
|------|------|-----------|-------|
| SR(short) ≥ A* | ≥1.00 | 0.90 | **否** |
| SR(long) ≥ A* | ≥1.00 | 0.95 | **否** |
| path(short) < A* | <14.20m | 14.47m | **否** |
| path(long) < A* | <42.98m | 46.43m | **否** |

## 结论（Phase 1: 对称融合）

1. **2-ens conf 是对称融合最优**: short SR=0.90 首次超过 V16-C 单模型 (0.85)
2. 但 long SR 从 V16-C 的 1.00 降到 0.95，long path +0.9%
3. Smoke SR=1.0 在 runs=20 中全部没保住（方差效应）
4. 含弱 DQN 的 3-ens 全部劣于 2-ens，弱模型是负担

---

## Phase 2: 非对称融合消融

### 新增 5 种模式

| 模式 | 策略 | 核心思想 |
|------|------|----------|
| rank | Borda 投票 | 尺度无关排名融合 |
| qgap | Q-gap 加权 | max-2nd 差值作为置信度 |
| veto | 主模型+否决权 | DDQN 优先，M-DQN 否决时回退 conf |
| topk | 约束选择 | DDQN top-K 中由 M-DQN 选最好 |
| negotiate | 谈判式 | DDQN 按优先级提议，M-DQN 审批 |

### Phase 2 Smoke (2-ens DDQN+M-DQN, runs=5, seed=42)

| 模式 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| rank | **1.0** | **12.80m** | **1.0** | 45.17m |
| qgap | 0.8 | 12.24m | 1.0 | 45.43m |
| veto | 1.0 | 13.50m | 0.8 | 44.56m |
| topk | 1.0 | 13.52m | 0.8 | 43.85m |
| negotiate(K=20) | **1.0** | 13.90m | **1.0** | 45.38m |

rank 和 negotiate 是唯二双 SR=1.0 的模式。

### Phase 2 Formal (2-ens, runs=20, seed=77)

| 模式 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| V16-C 单模型 | 0.85 | 14.43m | 1.00 | 46.00m |
| rank | 0.90 | 14.93m | 0.90 | 45.48m |
| **negotiate(K=20)** | **0.95** | 14.55m | **0.95** | 46.09m |
| A\*-MPC | 1.00 | 14.20m | 1.00 | 42.98m |

### negotiate K 值消融 (runs=20, seed=77)

| K | short SR | short path | long SR | long path | 失败 runs |
|---|----------|-----------|---------|-----------|----------|
| 10 | **0.95** | **14.34m** | 0.95 | 45.49m | s10, L9 |
| 11 | 0.90 | 14.69m | **1.00** | **45.12m** | s8,s10 |
| 12 | 0.85 | 14.61m | 0.95 | 44.96m | s8,s10,s16,L18 |
| 15 | 0.85 | 14.70m | **1.00** | 45.65m | s8,s10,s16 |
| **20** | **0.95** | 14.55m | 0.95 | 46.09m | s10, L3 |
| 30 | 0.85 | 14.82m | 0.95 | 45.61m | s8,s10,s16,L13 |

### 失败场景诊断 (seed=77)

| 方法 | short 失败 | long 失败 | 总失败 |
|------|-----------|----------|--------|
| V16-C 单模型 | 0, 8, 10 | 无 | 3 |
| 2-ens conf | 8, 10 | 18 | 3 |
| 2-ens rank | 8, 10 | 0, 9 | 4 |
| **2-ens negotiate(K=20)** | **10** | **3** | **2** |

negotiate 修复了 short run 8（rank/conf 均无法修复）。
short run 10 是所有方法的共同死穴（模型固有弱点）。

## §13 门槛检查（negotiate K=20 — 全局最优）

| 条件 | 要求 | negotiate K=20 | 通过? |
|------|------|---------------|-------|
| SR(short) >= A* | >=1.00 | 0.95 | **否** |
| SR(long) >= A* | >=1.00 | 0.95 | **否** |
| path(short) < A* | <14.20m | 14.55m | **否** |
| path(long) < A* | <42.98m | 46.09m | **否** |

## 最终结论

1. **negotiate(K=20) 是全局最优**: short SR=0.95 为历史最高
2. negotiate K=10 path 最短 (14.34m)，仅差 A* 0.14m
3. negotiate K=11 long SR=1.00，同时 short SR 提升到 0.90
4. **核心创新**: 谈判式融合让 DDQN 主导路径质量、M-DQN 做安全审查
5. **瓶颈**: short run 10 collision 无法被任何融合策略修复
6. §13 仍未通过，但差距大幅缩小（short SR 0.85->0.95）
