# V32 — 训练时动作掩码消融（Training-time Action Shield）

## 版本摘要

消融实验：在训练时启用 `--forest-action-shield`，
强制 ε-greedy 只从 admissible 动作中采样（当前 V22-D 训练时关闭）。
测试训练时约束探索空间是否改善推理时的 inadmissible rate 和路径质量。

## 方法

**零代码改动**：`--forest-action-shield` flag 已存在于 v9p2 profile（默认 false）。
启用后：ε-greedy 随机动作只从 admissible 子集采样，greedy argmax 也只在
admissible 动作上取最大值。TD target 计算不变（已有 next_mask）。

## 消融矩阵

| 实验 | 模型 | shield | 其余参数 | episodes |
|------|------|--------|----------|----------|
| V32-A | DDQN | **true** | V16-C: eps100, k_len=0.10, H=45 | 300 |
| V32-B | M-DQN | **true** | V22-D M-DQN: eps100, k_len=0.30, H=45 | 300 |
| V32 ens | V16-C + V32-B | negotiate K=10 | — | — |

## 主要结果（smoke runs=3）

见 [RESULTS.md](RESULTS.md)。

## 结论

**负面结果，不进 formal。**
- DDQN + shield = 灾难（inadmissible 22%→62%，SR 崩塌）
- M-DQN + shield = 中性（inadmissible 降到 16%，但 ensemble 无改善）
- V22-D negotiate K=10 仍是全局最优
