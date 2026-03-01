# v12 CHANGES — 相对 v11 的改动

## 版本意图

冻结 V11 消融实验最优配置（Config C: Dueling+MHA），进行正式评测并探索 horizon 参数空间。

## 相对 v11 的具体变更

**代码层面：无新增代码改动。** V12 完全复用 V11 代码，仅固定 CLI flags 组合。

### 配置变更

- 启用模块：`--dueling --mha`（V11 Config C）
- 禁用模块：CBAM（有负面作用）、NoisyNet（有害）、QR-DQN（有害）
- 基础 profile：`v9p2`（与 V11 一致）

### Horizon 参数实验（附加）

- `forest_adm_horizon`（动作掩码前瞻步数）：30 → 15 / 10（实验性）
- 结果：horizon=15 SR=0.0, horizon=10 SR=0.2 — 因 DQfD 专家 demo 与 horizon=30 耦合导致崩塌
- 结论：不可直接改 horizon，需重新生成专家 demo

## 变更文件

无代码文件变更（纯配置/评测版本）。

## 关键参数快照

| 参数 | 值 |
|------|-----|
| `--dueling` | True |
| `--mha` | True |
| `--mha-heads` | 4 (default) |
| `--cbam` | False |
| `--noisy-net` | False |
| `--n-quantiles` | 1 (default, 即不启用 QR-DQN) |
| `--profile` | v9p2 |
| `forest_adm_horizon` | 30 (训练/推理默认) |
| `episodes` | 150 |
| `runs` (正式) | 20 |
