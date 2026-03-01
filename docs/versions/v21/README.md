# V21 — M-DQN H×k_len 消融 + Negotiate Ensemble

## 动机

V20 negotiate(K=20) 达到历史最优 short SR=0.95，但 §13 仍未通过。
用户提出假设：**大胆增大视距 H 和路径惩罚 k_len，M-DQN 应能同时提升 SR 和缩短路径**。

## 方法

在 V16-C 基线上，固定 Dueling+MHA+M-DQN，消融 H∈{60,80} × k_len∈{0.20,0.30}。
最优单模型再与 V16-C DDQN ep190 组 negotiate ensemble。

## 训练参数（相对 V16-C 变更）

| 参数 | V16-C | V21 |
|------|-------|-----|
| `forest_adm_horizon` | 45 | **60 / 80** |
| `forest_reward_k_len` | 0.10 | **0.20 / 0.30** |
| `rl_algo` | cnn-ddqn | **cnn-mdqn** |

其余参数不变：`k_p=12.0, eps_decay=100, episodes=300, --dueling --mha`

## 关键发现

1. **H=60 远优于 H=80**: H=80 long SR 持续不稳定，最佳仅 0.7-0.8
2. **k_len=0.30 优于 k_len=0.20**: 更强路径惩罚让 M-DQN 找到更短路径
3. **B R2 ep150 = 历史首个 sr_all=1.0 训练 checkpoint**
4. **单 M-DQN short SR=1.00 首次超越 A*-MPC (0.95)**
5. **关键 Bug**: 推理时 H 必须匹配训练时 H（v9p2 profile 默认 H=15）
