# V22 — Path Length Reduction Ablation (K sweep + PBRS + DQfD + H45/k_len0.30)

## 动机

V21 single M-DQN 达到 short SR=1.00（首次超越 A*-MPC），但路径仍长。
§13 剩余 blockers：long SR < 1.0、path > A*-MPC。
V22 尝试三个方向降低路径长度：K sweep、PBRS 奖励塑形、DQfD 增强。

## 方法

### V22-A: Negotiate K sweep（零代码，推理消融）
- 模型：V16-C DDQN + V20 M-DQN（both H=45）
- K ∈ {5,8,10,12,15,25,30}，smoke runs=5 → top-3 formal runs=20

### V22-B: PBRS 训练（c_prog 参数，零代码）
- B1: Linear PBRS (`c_prog=12.0, potential_base=0`)
- B2: Exponential PBRS (`c_prog=12.0, potential_base=3.0`)

### V22-C: DQfD 增强（demo_lambda 参数）
- C1: `demo_lambda=16.0`（默认 8.0 的 2 倍）

### V22-D: M-DQN H=45 + k_len=0.30（legacy k_p）
- 组合 V20 的 H=45（短路径）+ V21 的 k_len=0.30（高 SR）

## 关键发现

1. **V22-D negotiate K=10 是历史最佳**：首次 long SR=1.00 + short path < A*
2. **PBRS 全面失败**：c_prog 替换 k_p 后泛化能力显著下降
3. **DQfD 增强失败**：demo_lambda=16 过强干扰 TD 学习
4. **K=10 是最优 K**：比 K=20 更严格的审批 → 更高 long SR
5. **V22-D 训练最稳定**：完成全 300ep，多次 sr_all=1.0
