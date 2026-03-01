# V36 — 架构消融实验（DRQN / Dual-Scale / HistTransformer）

## 背景

V35 系列（M-DQN K sweep + 角色互换）补训无效，结论为"补训没有用，必须换架构"。
V36 在 V22-D 稳定基线（negotiate K=10）基础上，对三种新架构做消融实验，
验证架构变化能否突破 long path 瓶颈（46m vs A* 42.9m）。

## 三种实验设计

| 版本 | 架构 | 算法名 | 核心思路 |
|------|------|--------|---------|
| V36-A | DRQN | `cnn-drqn` | CNN → LSTM(256,256) → Q，BPTT 序列训练 |
| V36-B | Dual-Scale CNN | `cnn-dual-ddqn` | 局部 12×12 + 全局 8×8 双路 CNN，obs_dim=218 |
| V36-C | Hist-Transformer | `cnn-histformer` | CNN → proj → TransformerEncoder(T=8) → Q |

## Smoke 结果（150ep，runs=3）

| 架构 | short SR | long SR | 结论 |
|------|----------|---------|------|
| V36-A DRQN | 0 | 0 | 失败：150ep 无 demo 未收敛 |
| V36-B Dual-Scale | 0 | 0 | 失败：demo 被禁用（obs_dim 不匹配），无法预训练 |
| V36-C HistFormer | 0 | 0 | 失败：150ep 无 demo 未收敛 |

**全部架构 smoke 失败，按规则 12.2 不进入 full 评测。**

## 失败根因分析

1. **Demo 预训练缺失**：V36-A/C 的 `train_one_drqn/histformer` 无 DQfD demo 预训练；
   V36-B 的 DualScaleWrapper 使 obs_dim 从 154→218，已有 demo（154-dim）无法复用。
2. **Episodes 不足**：无 demo 引导时，新架构需要 300ep+ 才能看到初步学习信号
   （参考 V15 无 demo 时 150ep 也基本 SR=0）。
3. **训练信号稀疏**：LSTM/Transformer 的初始化期更长，且当前 seq_len=8 对长距导航偏短。

## 代码状态

三种架构代码完整实现，self-check 和 3ep 测试全部通过：
- `networks.py`: DualScaleCNNQNetwork, DRQNNetwork, HistTransformerNetwork
- `agents.py`: DQNFamilyAgent（cnn-dual 分支），EpisodeReplayBuffer，DRQNAgent，HistFormerAgent
- `train.py`: train_one_drqn(), train_one_histformer()，cnn-dual-ddqn 复用 train_one()
- `infer.py`: rollout_drqn(), rollout_histformer()，resolve_model_path fallback

## 下一步建议

1. **为 V36-A/C 加 demo 预训练**：在 train_one_drqn/histformer 中实现 BC 预训练（行为克隆）
2. **延长训练 300ep**：加 demo 后重跑 smoke
3. **或回退 V22-D**：继续在 V22-D 基础上探索其他方向（如 geodesic + larger map）
