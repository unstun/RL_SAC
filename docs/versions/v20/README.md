# V20 — 异构 DQN Ensemble（DQN + DDQN + M-DQN）

## 动机

V16-C ep190（cnn-ddqn + Dueling + MHA）是全局最优单模型但 §13 未通过。
V19 同构 ensemble（两个 DDQN 模型）因质量不匹配导致负面结果（path +33%）。

**新方案**: 异构 ensemble — 三种不同 TD target 算法 × 相同 Dueling+MHA 网络架构，推理时 Q-value averaging。

## 学术依据

- Averaged-DQN (Anschel et al., ICML 2017) — Q 值平均理论
- Munchausen RL (Vieillard et al., NeurIPS 2020) — M-DQN 算法
- DM-DQN (Complex & Intelligent Systems, 2022) — Dueling+Munchausen 路径规划先例

## 核心改动

1. `agents.py`: 新增 M-DQN 训练算法（AlgoBase 扩展、TD target 分支）
2. `infer.py`: ensemble 加载改为跨 algo（自动搜索 cnn-dqn/ddqn/mdqn/*.pt）
3. 训练 DQN 和 M-DQN 两个新模型

## Smoke 结论

- 3-model ensemble long path=44.27m 是所有配置中最短
- 2-model ensemble (DDQN+M-DQN) short path=12.71m 首次短于 A*
- 但 ensemble short SR=0.8，低于 DDQN 单模型的 1.0
- **未进入 formal 评测**（short SR 回退是 §13 blocker）
