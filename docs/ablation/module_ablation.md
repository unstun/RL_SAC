# 模块消融实验 (V11 Ablation Study)

> 来源版本: V11 — 即插即用模块化 CNN-DDQN
> 原始留档: `docs/versions/v11/RESULTS.md`

## 实验设计

- **基础架构**: CNN-DDQN + DQfD, profile=v9p2
- **训练**: episodes=150, GPU
- **推理 (smoke)**: runs=5, short/mid/long, `--rand-two-suites`
- **推理 (正式)**: runs=20, short/long
- **方法**: 从最佳初始配置 (Dueling+CBAM+MHA) 逐一移除/加入模块

## 消融矩阵

| # | 配置 | Dueling | CBAM | MHA | NoisyNet | QR-DQN |
|---|------|---------|------|-----|----------|--------|
| A | Full (D+C+M) | Y | Y | Y | - | - |
| B | w/o MHA | Y | Y | - | - | - |
| C | w/o CBAM | Y | - | Y | - | - |
| D | w/o Dueling | - | Y | Y | - | - |
| E | +NoisyNet | Y | Y | Y | Y | - |
| F | +QR-DQN(32) | Y | Y | Y | - | Y |
| G | Baseline v9p2 | - | - | - | - | - |

## Smoke 推理结果 (runs=5)

### Success Rate

| # | 配置 | short | mid | long | 平均 |
|---|------|-------|-----|------|------|
| A | Full (D+C+M) | 0.8 | 0.8 | 0.8 | 0.80 |
| B | w/o MHA | 0.8 | 0.4 | 0.2 | 0.47 |
| **C** | **w/o CBAM** | **1.0** | **0.8** | **1.0** | **0.93** |
| D | w/o Dueling | 0.8 | 0.8 | 1.0 | 0.87 |
| E | +NoisyNet | 1.0 | 0.4 | 0.0 | 0.47 |
| F | +QR-DQN(32) | 0.6 | 0.4 | 0.2 | 0.40 |
| G | Baseline v9p2 | 0.6 | 0.4 | 0.0 | 0.33 |
| - | Hybrid A\*-MPC | 1.0 | 1.0 | 1.0 | 1.00 |

### Average Path Length (m, 仅成功 runs)

| # | 配置 | short | mid | long |
|---|------|-------|-----|------|
| A | Full (D+C+M) | 19.56 | 32.49 | 53.23 |
| B | w/o MHA | 19.53 | 34.58 | 50.41 |
| **C** | **w/o CBAM** | **17.88** | **26.85** | **47.47** |
| D | w/o Dueling | 16.46 | 28.03 | 45.43 |
| E | +NoisyNet | 19.91 | 28.31 | N/A |
| F | +QR-DQN(32) | 31.51 | 44.51 | 67.31 |
| G | Baseline v9p2 | 15.29 | 31.03 | N/A |
| - | Hybrid A\*-MPC | 16.87 | 25.15 | 43.02 |

### Path Time (s, 仅成功 runs)

| # | 配置 | short | mid | long |
|---|------|-------|-----|------|
| A | Full (D+C+M) | 13.71 | 23.09 | 33.10 |
| B | w/o MHA | 17.96 | 25.73 | 35.55 |
| **C** | **w/o CBAM** | **11.32** | **16.20** | **26.54** |
| D | w/o Dueling | 10.51 | 17.26 | 27.10 |
| E | +NoisyNet | 13.42 | 17.88 | N/A |
| F | +QR-DQN(32) | 20.47 | 27.10 | 38.15 |
| G | Baseline v9p2 | 9.68 | 21.70 | N/A |
| - | Hybrid A\*-MPC | 10.00 | 13.87 | 22.82 |

## 模块贡献分析

### 1. MHA (Spatial Multi-Head Attention) — 最关键模块

- 移除 MHA (A→B): long SR 0.8→0.2, mid SR 0.8→0.4
- 含 MHA 的配置 (A/C/D) 平均 SR=0.87; 不含 MHA 的 (B/G) 平均 SR=0.40
- **结论: 必须保留**

### 2. CBAM — 有负面作用

- 移除 CBAM (A→C): short SR 0.8→1.0, long SR 0.8→1.0（全面提升）
- C (Dueling+MHA) 是所有配置中综合最优
- **结论: 禁止默认启用**

### 3. Dueling — 作用不明确

- 移除 Dueling (A→D): long SR 0.8→1.0（反而提升）
- D (CBAM+MHA) 与 C (Dueling+MHA) 表现接近
- **结论: 保留无害，贡献不确定**

### 4. NoisyNet — 有害

- 加入 NoisyNet (A→E): long SR 0.8→0.0, mid SR 0.8→0.4
- 探索噪声在本任务中导致长距离导航严重退化
- **结论: 禁止使用**

### 5. QR-DQN(32) — 有害

- 加入 QR-DQN (A→F): 全面下降, short SR 0.8→0.6
- 150ep 训练不足以收敛 32 分位数的分布式 Q 值
- **结论: 禁止使用（除非大幅增加训练量）**

## 最优配置正式评测 (Config C, runs=20)

| 指标 | short | long | vs A\*-MPC |
|------|-------|------|-----------|
| SR | 0.95 | 0.95 | =/−5% |
| Path (m) | 16.41 | 47.34 | +4.1%/+10.3% |
| Time (s) | 10.54 | 26.61 | +13.9%/+16.9% |
| Compute (s) | 0.142 | 0.328 | **3.5x/5.9x 加速** |

## Run 目录

| # | 训练 run | 推理 run |
|---|----------|----------|
| A | v11-smoke-dcbam-mha | v11-abl-infer-full/ |
| B | v11-smoke-dcbam | v11-abl-infer-dcbam/ |
| C | v11-abl-duel-mha | v11-abl-infer-duel-mha/ |
| D | v11-abl-cbam-mha | v11-abl-infer-cbam-mha/ |
| E | v11-abl-full-noisy | v11-abl-infer-full-noisy/ |
| F | v11-abl-full-qr32 | v11-abl-infer-full-qr32/ |
| G | v11-abl-baseline | v11-abl-infer-baseline/ |
| C(正式) | v11-abl-duel-mha | v11-formal-duel-mha/ |
