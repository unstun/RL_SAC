# v11 — 即插即用模块化 CNN-DDQN

## 概述

在 v9p2 (CNN-DDQN + DQfD) 基础上，实现 5 个即插即用增强模块。
每个模块通过独立 CLI flag 开关，可自由组合。

## 新增模块

| 模块 | CLI Flag | 论文来源 |
|------|----------|----------|
| Dueling DQN | `--dueling` | Wang et al., ICML 2016 |
| CBAM | `--cbam` | Woo et al., ECCV 2018 |
| NoisyNet | `--noisy-net` | Fortunato et al., ICLR 2018 |
| Spatial MHA | `--mha [--mha-heads N]` | Vaswani et al., 2017 |
| QR-DQN | `--n-quantiles N` | Dabney et al., AAAI 2018 |

## 数据流（全模块启用）

```
obs → split(scalars, map)
    → conv(map)           # 3层CNN → (B, 64, 3, 3)
    → CBAM(conv_out)      # 通道+空间注意力
    → SpatialMHA(conv_out) # 9个空间token自注意力
    → flatten → concat(scalars, flat) = feats
    → NoisyLinear/Linear  # 噪声线性层
    → Dueling(V+A)        # 价值/优势分流
    → reshape(n_quantiles) # QR-DQN分位数
```

## 状态

**功能验证通过**（各模块单独 + 全组合 10ep smoke 不崩）。
待正式 smoke (episodes=150, runs=3) 评估效果。
