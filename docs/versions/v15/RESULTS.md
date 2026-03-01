# v15 RESULTS — 路径长度优化实验结果

## 实验总览

### 路径 1: Epsilon 调度修复

| 实验 | eps_decay | episodes | ckpt | prog | 结果概述 |
|------|-----------|----------|------|------|----------|
| E1-fix | 100 | 150 | ep110 | 0.01 | short 0.8, long 1.0 (best ckpt 差) |
| E1-long | 100 | 300 | ep220 | 0.0 | short 0.8, long 1.0, path=50.81m |

### 路径 2: Reward Shaping (基于 eps_decay=100)

| 实验 | k_len | ckpt | prog | 结果概述 |
|------|-------|------|------|----------|
| R1 | 0.05 | ep130 | 0.0 | **最优**: long SR=1.0, path=46.14m |
| R2 | 0.10 | ep170 | 0.0 | 失败: long SR=0.2, path=66.3m |

### 路径 3: min_progress_m 放松 (v11 模型, 零训练成本)

| 实验 | prog | 结果概述 |
|------|------|----------|
| M2 | 0.005 | long SR +0.2 |
| M4 | 0.0 | long SR +0.2, 无副作用 → **采纳** |

### 多 Seed 实验 (R1 配置)

| seed | 训练结果 | 推理结果 |
|------|----------|----------|
| 21 | ep130 SR=1.0/1.0 | **最优**: runs=20 long SR=1.00 |
| 42 | 早停 ep170, best_sr=0.5 | 完全失败 |
| 7 | ep150 SR=1.0/1.0 | best: short 0.9/long 0.8 |

---

## 正式评测（runs=20, seed=77）

配置: R1 (k_len=0.05, eps_decay=100, seed=21), ckpt=ep130, prog=0.0
Run: `v15-formal-r1-ep130-prog0/20260224_074827`

### Success Rate

| Suite | CNN-DDQN | A\*-MPC | 差异 |
|-------|----------|---------|------|
| short | 0.80 | 1.00 | -20% |
| long | **1.00** | 1.00 | = |

### Average Path Length (m, 仅成功 runs)

| Suite | CNN-DDQN | A\*-MPC | 差异 |
|-------|----------|---------|------|
| short | 15.19 | 14.20 | +7.0% |
| long | 46.14 | 42.98 | +7.4% |

### Path Time (s, 仅成功 runs)

| Suite | CNN-DDQN | A\*-MPC | 差异 |
|-------|----------|---------|------|
| short | 10.35 | 8.67 | +19.4% |
| long | 27.72 | 22.79 | +21.6% |

### Compute Time (s)

| Suite | CNN-DDQN | A\*-MPC | 加速比 |
|-------|----------|---------|--------|
| short | 0.39 | 1.36 | **3.5x** |
| long | 0.86 | 5.19 | **6.0x** |

### 其他指标

| Suite | 指标 | CNN-DDQN | A\*-MPC |
|-------|------|----------|---------|
| short | curvature (1/m) | 0.113 | 0.062 |
| short | corners | 0 | 0 |
| short | max corner (°) | 45 | 46 |
| short | inadmissible rate | 0.161 | N/A |
| long | curvature (1/m) | 0.133 | 0.068 |
| long | corners | 2 | 0 |
| long | max corner (°) | 28 | 3 |
| long | inadmissible rate | 0.171 | N/A |

---

## 辅助评测（runs=10）

### R1 ep130+prog0, seed=42 (runs=10)

Run: `v15-infer-r1-ep130-prog0-r10/20260224_074257`

| Suite | CNN-DDQN SR | path (m) | A\*-MPC SR | A\* path (m) |
|-------|-------------|----------|------------|--------------|
| short | 0.80 | 15.53 | 0.90 | 15.31 |
| long | **1.00** | **44.78** | 1.00 | 42.87 |

### R1 seed=7, best ckpt+prog0 (runs=10)

Run: `v15-infer-r1-seed7-best-prog0-r10/20260224_093638`

| Suite | CNN-DDQN SR | path (m) | A\*-MPC SR | A\* path (m) |
|-------|-------------|----------|------------|--------------|
| short | 0.90 | 16.47 | 0.90 | 15.31 |
| long | 0.80 | 52.30 | 1.00 | 42.87 |

### R1 seed=7, ep150+prog0 (runs=10)

Run: `v15-infer-r1-seed7-ep150-prog0-r10/20260224_093632`

| Suite | CNN-DDQN SR | path (m) | A\*-MPC SR | A\* path (m) |
|-------|-------------|----------|------------|--------------|
| short | 0.70 | 16.91 | 0.90 | 15.31 |
| long | 0.70 | 55.06 | 1.00 | 42.87 |

### R2 (k_len=0.10) ep170+prog0 (runs=5)

Run: `v15-infer-r2-ep170-prog0/20260224_084120`

| Suite | CNN-DDQN SR | path (m) |
|-------|-------------|----------|
| short | 0.60 | 19.12 |
| long | 0.20 | 66.31 |

**结论**: k_len=0.10 过于激进，SR 严重崩溃，排除。

---

## 版本对比（v11 → v15 正式评测, runs=20）

| 指标 | v11 | v15 | 变化 |
|------|-----|-----|------|
| short SR | 0.95 | 0.80 | -15% ❌ |
| short path (m) | 16.41 | 15.19 | -7.4% ✅ |
| long SR | 0.95 | **1.00** | +5% ✅ |
| long path (m) | 47.34 | **46.14** | -2.5% ✅ |

## 门槛判定 (CLAUDE.md §13)

| 条件 | short | long | 通过? |
|------|-------|------|-------|
| SR(CNN) >= SR(A\*) | 0.80 < 1.00 ❌ | 1.00 = 1.00 ✅ | ❌ |
| path < A\* path | 15.19 > 14.20 ❌ | 46.14 > 42.98 ❌ | ❌ |
| time < A\* time | 10.35 > 8.67 ❌ | 27.72 > 22.79 ❌ | ❌ |

**结论: 未通过最终门槛。** Long SR 首次达标，但 path/time 仍超 A\*-MPC。
