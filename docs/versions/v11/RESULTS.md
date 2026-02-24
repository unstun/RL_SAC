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
