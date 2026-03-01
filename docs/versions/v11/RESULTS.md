# v11 RESULTS — 功能验证 Smoke 结果

## 验证环境

- 远端: ubuntu-zt, CUDA, conda ros2py310
- 基线 profile: v9p2
- episodes: 10 (功能验证级别)

## 各模块 Smoke 结果 (10 episodes)

| 配置 | exit_code | pretrain SR | ep1 SR(s/l) | ep10 SR(s/l) | 速度 |
|------|-----------|------------|-------------|--------------|------|
| CBAM only | 0 | - | - | - | ~5s/ep |
| NoisyNet only | 0 | 0.750 | 1.0/0.4 | 0.6/0.0 | ~5s/ep |
| MHA only | 0 | 0.625 | 0.6/0.2 | 1.0/0.2 | ~7s/ep |
| QR-DQN(32) only | 0 | 0.750 | 0.0/0.0 | 0.6/0.0 | ~6s/ep |
| 全组合 | 0 | 0.500 | 0.0/0.0 | 0.0/0.0 | ~10s/ep |

## 结论

- 所有 5 个模块单独和全组合均训练不崩
- 全组合速度约为基线 2 倍（10s/ep vs 5s/ep），符合预期
- 10ep 仅为功能验证，SR 不作为效果评估依据

## 正式 Smoke 训练结果 (episodes=150)

| 配置 | 最佳 SR(s/l) | 最佳 epoch | ep150 SR(s/l) | ratio(s/l) |
|------|-------------|-----------|---------------|------------|
| Dueling+CBAM | 1.0/1.0 | ep80 | early-stop@140 | 9.5/13.2 |
| **Dueling+CBAM+MHA** | **1.0/1.0** | ep80,90,110,150 | **1.0/1.0** | **9.6/10.6** |
| Dueling+CBAM+NoisyNet | 1.0/0.6 | ep100 | 1.0/0.2 | 10.2/14.6 |

最佳配置: Dueling+CBAM+MHA（多次达到双套件 SR=100%，路径比最短）

## 正式 Smoke 推理结果 (runs=3, Dueling+CBAM+MHA)

| Suite | CNN-DDQN SR | Path(m) | Baseline SR | Baseline Path(m) |
|-------|------------|---------|-------------|-------------------|
| short | 0.667 | 20.16 | 1.0 | 17.03 |
| mid | 0.667 | 28.20 | 1.0 | 24.08 |
| long | 0.667 | 52.41 | 1.0 | 43.01 |

注: runs=3 方差大（2/3成功=0.667），训练 eval 多次 1.0/1.0。正式结论需 runs=20。

---

## 消融实验 (Ablation Study)

### 实验设计

- 训练: episodes=150, 基线 profile=v9p2
- 推理: runs=5, short/mid/long 三套件, `--profile v9p2` (随机起终点)
- 基准: 从最佳配置 (Dueling+CBAM+MHA) 逐一移除模块 + 加入 NoisyNet/QR-DQN + 纯基线

### 消融矩阵

| # | 配置 | Dueling | CBAM | MHA | NoisyNet | QR-DQN |
|---|------|---------|------|-----|----------|--------|
| A | Full (D+C+M) | Y | Y | Y | - | - |
| B | w/o MHA | Y | Y | - | - | - |
| C | w/o CBAM | Y | - | Y | - | - |
| D | w/o Dueling | - | Y | Y | - | - |
| E | +NoisyNet | Y | Y | Y | Y | - |
| F | +QR-DQN(32) | Y | Y | Y | - | Y |
| G | Baseline v9p2 | - | - | - | - | - |

### 推理结果: Success Rate

| # | 配置 | short | mid | long | 平均 |
| --- | --- | --- | --- | --- | --- |
| A | Full (D+C+M) | 0.8 | 0.8 | 0.8 | 0.80 |
| B | w/o MHA | 0.8 | 0.4 | 0.2 | 0.47 |
| **C** | **w/o CBAM** | **1.0** | **0.8** | **1.0** | **0.93** |
| D | w/o Dueling | 0.8 | 0.8 | 1.0 | 0.87 |
| E | +NoisyNet | 1.0 | 0.4 | 0.0 | 0.47 |
| F | +QR-DQN(32) | 0.6 | 0.4 | 0.2 | 0.40 |
| G | Baseline v9p2 | 0.6 | 0.4 | 0.0 | 0.33 |
| - | Hybrid A\*-MPC | 1.0 | 1.0 | 1.0 | 1.00 |

### 推理结果: Average Path Length (m, 仅成功 runs)

| # | 配置 | short | mid | long |
| --- | --- | --- | --- | --- |
| A | Full (D+C+M) | 19.56 | 32.49 | 53.23 |
| B | w/o MHA | 19.53 | 34.58 | 50.41 |
| **C** | **w/o CBAM** | **17.88** | **26.85** | **47.47** |
| D | w/o Dueling | 16.46 | 28.03 | 45.43 |
| E | +NoisyNet | 19.91 | 28.31 | N/A |
| F | +QR-DQN(32) | 31.51 | 44.51 | 67.31 |
| G | Baseline v9p2 | 15.29 | 31.03 | N/A |
| - | Hybrid A\*-MPC | 16.87 | 25.15 | 43.02 |

### 推理结果: Path Time (s, 仅成功 runs)

| # | 配置 | short | mid | long |
| --- | --- | --- | --- | --- |
| A | Full (D+C+M) | 13.71 | 23.09 | 33.10 |
| B | w/o MHA | 17.96 | 25.73 | 35.55 |
| **C** | **w/o CBAM** | **11.32** | **16.20** | **26.54** |
| D | w/o Dueling | 10.51 | 17.26 | 27.10 |
| E | +NoisyNet | 13.42 | 17.88 | N/A |
| F | +QR-DQN(32) | 20.47 | 27.10 | 38.15 |
| G | Baseline v9p2 | 9.68 | 21.70 | N/A |
| - | Hybrid A\*-MPC | 10.00 | 13.87 | 22.82 |

### 消融分析

**模块贡献排序** (按 SR 提升幅度):

1. **MHA (Spatial Multi-Head Attention)** — 最关键模块

   - 移除 MHA (A→B): long SR 0.8→0.2, mid SR 0.8→0.4
   - 含 MHA 的配置 (A/C/D) 平均 SR=0.87; 不含 MHA 的 (B/G) 平均 SR=0.40

2. **Dueling** — 作用不明确

   - 移除 Dueling (A→D): long SR 0.8→1.0 (反而提升)
   - D (CBAM+MHA) 与 C (Dueling+MHA) 表现接近

3. **CBAM** — 可能有负面作用

   - 移除 CBAM (A→C): short SR 0.8→1.0, long SR 0.8→1.0 (全面提升)
   - C (Dueling+MHA) 是所有配置中综合最优

4. **NoisyNet** — 有害

   - 加入 NoisyNet (A→E): long SR 0.8→0.0, mid SR 0.8→0.4
   - NoisyNet 的探索噪声在本任务中导致长距离导航严重退化

5. **QR-DQN(32)** — 有害

   - 加入 QR-DQN (A→F): 全面下降, short SR 0.8→0.6
   - 150ep 训练不足以收敛 32 分位数的分布式 Q 值

**最优配置**: C (Dueling+MHA, 不含 CBAM)
- 综合 SR=0.93 (1.0/0.8/1.0), 全配置最高
- Path length 仅比 Hybrid A*-MPC 长 6-10%
- Path time 比 Hybrid A*-MPC 长 13-17%

**vs Baseline v9p2 (G)**:
- G: SR=0.33 (0.6/0.4/0.0) → C: SR=0.93 — 提升 +181%
- 模块组合显著提升了 CNN-DDQN 在随机起终点场景的泛化能力

### Run 目录映射

| # | 训练 run | 推理 run |
| --- | --- | --- |
| A | v11-smoke-dcbam-mha | v11-abl-infer-full/20260224_154426 |
| B | v11-smoke-dcbam | v11-abl-infer-dcbam/20260224_154428 |
| C | v11-abl-duel-mha | v11-abl-infer-duel-mha/20260224_154430 |
| D | v11-abl-cbam-mha | v11-abl-infer-cbam-mha/20260224_154653 |
| E | v11-abl-full-noisy | v11-abl-infer-full-noisy/20260224_154653 |
| F | v11-abl-full-qr32 | v11-abl-infer-full-qr32/20260224_154702 |
| G | v11-abl-baseline | v11-abl-infer-baseline/20260224_154704 |

---

## 正式评测 (runs=20, short/long 双套件)

配置: **C (Dueling+MHA)** — 消融实验最优配置
训练 run: `v11-abl-duel-mha`
推理 run: `v11-formal-duel-mha/20260224_160845`

### Success Rate

| Suite | CNN-DDQN | Hybrid A\*-MPC | 差异 |
| --- | --- | --- | --- |
| short | 0.95 | 0.95 | = |
| long | 0.95 | 1.0 | -5% |

### Average Path Length (m, 仅成功 runs)

| Suite | CNN-DDQN | Hybrid A\*-MPC | 差异 |
| --- | --- | --- | --- |
| short | 16.41 | 15.77 | +4.1% |
| long | 47.34 | 42.93 | +10.3% |

### Path Time (s, 仅成功 runs)

| Suite | CNN-DDQN | Hybrid A\*-MPC | 差异 |
| --- | --- | --- | --- |
| short | 10.54 | 9.25 | +13.9% |
| long | 26.61 | 22.76 | +16.9% |

### Compute Time (s)

| Suite | CNN-DDQN | Hybrid A\*-MPC | 加速比 |
| --- | --- | --- | --- |
| short | 0.142 | 0.504 | **3.5x** |
| long | 0.328 | 1.944 | **5.9x** |

### 最终门槛判定 (CLAUDE.md §13)

| 条件 | short | long | 通过? |
| --- | --- | --- | --- |
| SR(CNN) >= SR(A\*-MPC) | 0.95 = 0.95 ✅ | 0.95 < 1.0 ❌ | ❌ |
| path_length(CNN) < path_length(A\*-MPC) | 16.41 > 15.77 ❌ | 47.34 > 42.93 ❌ | ❌ |
| path_time(CNN) < path_time(A\*-MPC) | 10.54 > 9.25 ❌ | 26.61 > 22.76 ❌ | ❌ |

**结论: 未通过最终门槛。**

### 积极面

- SR=0.95 相比 baseline v9p2 (SR=0.33) 提升 +188%
- 计算时间 CNN-DDQN 快 3.5-5.9x（无需 A\* 搜索+MPC 跟踪）
- 路径长度仅长 4-10%，路径时间仅慢 14-17%
- V11 模块组合显著提升了泛化能力，但尚不足以超越传统规划
