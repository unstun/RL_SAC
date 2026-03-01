# V36 Smoke 结果

## V36-B (Dual-Scale CNN, cnn-dual-ddqn) — 150ep

训练：`runs/v36-B-dual-smoke/train_20260227_211404`（本地 GPU，4m26s）
推理：`runs/v36-B-dual-infer/20260227_211942`

| 套件  | SR | path(m) | compute(s) |
| ----- | -- | ------- | ---------- |
| short | 0  | —       | —          |
| long  | 0  | —       | —          |

**失败原因**：demo 禁用（obs_dim 154→218 不匹配），无预训练 → 150ep 随机探索无法收敛

---

## V36-B2 (Dual-Scale CNN, cnn-dual-ddqn, 修复 demo 采集) — 150ep

训练：`runs/v36-B2-dual-smoke/train_20260227_220457`（本地 GPU，8m57s 总 / RL 阶段 1m04s）
推理：`runs/v36-B2-dual-infer/20260227_221508`

| 套件  | SR | path(m) | compute(s) |
| ----- | -- | ------- | ---------- |
| short | 0  | —       | —          |
| long  | 0  | —       | —          |

**修复内容**：train.py 改为用 DualScaleWrapper 重新采集 218-dim demo，demo 已成功采集（size=19864），DQfD 预训练正常运行。

**仍然失败，根因（深层架构问题）**：

| 对比项              | V36-B2 (DualScale)    | 标准 DDQN  |
| ------------------- | --------------------- | ---------- |
| fc_in               | 1098（10+576+512）    | 586        |
| 平均 episode 长度   | 43 步                 | 115 步     |
| train_steps (150ep) | 6505 (38%)            | 17287      |
| 正收益首次出现      | 从未                  | ep20: +491 |

根因：

- `d_global=512`：8×8 地图（每格约 6m）经 2 层 Conv → 32 通道 × 4×4 = 512 维，信噪比极低
- fc_in 是标准 DDQN 的 1.87 倍 → 网络复杂度过高，150ep 远不够收敛
- 结论：8×8 全局地图分辨率不足，需改为 16×16 + 轻量全局分支（GAP 压缩至 32 维）

---

## V36-B3 (Dual-Scale CNN, cnn-dual-ddqn, 16x16 + GAP) — 150ep

训练：`runs/v36-B3-dual-smoke/train_20260227_231634`（本地 GPU，8m06s 总 / RL 阶段 1m04s）
推理：`runs/v36-B3-dual-infer/20260227_232519`

| 套件  | SR | path(m) | compute(s) |
| ----- | -- | ------- | ---------- |
| short | 0  | —       | —          |
| long  | 0  | —       | —          |

改动内容：

- `DualScaleWrapper.GLOBAL_SIZE`: 8 → 16，obs_dim: 218 → 410
- `global_conv`: 2-layer Conv + GAP → d_global=32，fc_in=618（约等于标准 DDQN 586，+5%）

**仍然失败，三版对比**：

| 对比项              | V36-B3 (16x16 GAP) | V36-B2 (8x8) | 标准 DDQN  |
| ------------------- | ------------------ | ------------ | ---------- |
| obs_dim             | 410                | 218          | 154        |
| fc_in               | 618 (+5%)          | 1098 (+87%)  | 586        |
| train_steps (150ep) | 5462               | 6505         | 17287      |
| 平均 ep 长度        | 36 步              | 43 步        | 115 步     |
| 正收益首次出现      | 从未               | 从未         | ep20: +491 |
| short: 1/3 timeout  | 是（能存活）       | 否           | —          |

根因：

- fc_in 降至 618 有改善（short 2/3 次 timeout 而非全碰），但 long 仍即刻碰撞（63-66步）
- ep=30 出现 return=-113118（数值不稳定，Q-值爆炸）
- obs_dim 410 vs 154，扩展 2.7 倍，DQfD 30000步预训练不足以填补差距
- 结论：DualScale 路线在 150ep 下不可行，需至少 300-500ep 或更强的 demo 初始化

---

## V36-A (DRQN, cnn-drqn) — 150ep

训练：`runs/v36-A-drqn-smoke/train_20260227_212056`（本地 GPU，3m05s）
推理：`runs/v36-A-drqn-infer/20260227_212545`

| 套件  | SR | path(m) | compute(s) |
| ----- | -- | ------- | ---------- |
| short | 0  | —       | —          |
| long  | 0  | —       | —          |

**失败原因**：无 BC/DQfD demo 预训练实现 → 150ep 从零探索无法收敛

---

## V36-C (HistFormer, cnn-histformer) — 150ep

训练：`runs/v36-C-histformer-smoke/train_20260227_212639`（本地 GPU，2m50s）
推理：`runs/v36-C-histformer-infer/20260227_212945`

| 套件  | SR | path(m) | compute(s) |
| ----- | -- | ------- | ---------- |
| short | 0  | —       | —          |
| long  | 0  | —       | —          |

**失败原因**：无 BC/DQfD demo 预训练实现 → 150ep 从零探索无法收敛

---

## 参考基线 (V22-D negotiate K=10, runs=20)

| 套件  | SR   | path(m) | compute(s) |
| ----- | ---- | ------- | ---------- |
| short | 0.95 | 15.75m  | 0.71s      |
| long  | 1.00 | 46.11m  | 1.69s      |

A\*-MPC: short 0.95/15.77m, long 1.00/42.93m

---

## 结论

所有新架构 smoke 全部失败（SR=0），按规则 12.2 不进入 full 评测。
V22-D negotiate K=10 仍是全局最优基线。
