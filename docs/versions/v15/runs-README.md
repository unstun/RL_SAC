# v15 Runs 目录

## 训练 Runs

| Run | 配置 | seed | 说明 |
|-----|------|------|------|
| v15-baseline-h30 | v9p2 原始 (eps_decay=4500) | 21 | 基线对照 |
| v15-e1-fix | eps_decay=100, ep150 | 21 | 路径1: ε修复 |
| v15-e1-long | eps_decay=100, ep300 | 21 | 路径1: ε修复+延长 |
| **v15-r1-klen005** | eps_decay=100, k_len=0.05, ep300 | 21 | **路径2: 最优配置** |
| v15-r2-klen010 | eps_decay=100, k_len=0.10, ep300 | 21 | 路径2: k_len过激(失败) |
| v15-r1-seed42 | 同R1, seed=42 | 42 | 多seed: 完全失败 |
| v15-r1-seed7 | 同R1, seed=7 | 7 | 多seed: 部分成功 |

## Checkpoint 副本 (从训练 run 复制的 periodic ckpt)

| Run | 来源 | 说明 |
|-----|------|------|
| v15-e1-fix-ep110 | v15-e1-fix ep110 | E1-fix 最佳推理 ckpt |
| v15-e1-long-ep200 | v15-e1-long ep200 | |
| v15-e1-long-ep210 | v15-e1-long ep210 | |
| v15-e1-long-ep220 | v15-e1-long ep220 | E1-long 最佳推理 ckpt |
| v15-e1-long-ep230 | v15-e1-long ep230 | |
| **v15-r1-ep130** | v15-r1-klen005 ep130 | **正式评测用 ckpt** |
| v15-r1-ep150 | v15-r1-klen005 ep150 | |
| v15-r1-ep200 | v15-r1-klen005 ep200 | |
| v15-r1-seed7-ep150 | v15-r1-seed7 ep150 | |
| v15-r2-ep170 | v15-r2-klen010 ep170 | |

## 推理 Runs

### 正式评测 (runs=20)

| Run | 模型 | prog | seed | 说明 |
|-----|------|------|------|------|
| **v15-formal-r1-ep130-prog0** | v15-r1-ep130 | 0.0 | 77 | **最终正式评测** |
| v15-formal-e1long-ep220-prog0 | v15-e1-long-ep220 | 0.0 | 77 | E1-long 正式 |

### Smoke 推理 (runs=5 或 runs=10)

| Run | 模型 | prog | runs | 说明 |
|-----|------|------|------|------|
| v15-infer-baseline | v15-baseline-h30 | 0.01 | 5 | 基线 |
| v15-infer-e1-fix | v15-e1-fix (best) | 0.01 | 5 | best ckpt 差 |
| v15-infer-e1-fix-ep110 | v15-e1-fix-ep110 | 0.01 | 5 | ep110 好 |
| v15-infer-e1-long | v15-e1-long (best) | 0.01 | 5 | |
| v15-infer-e1long-ep220-prog0 | v15-e1-long-ep220 | 0.0 | 5 | |
| v15-infer-e1long-ep220-prog0-r10 | v15-e1-long-ep220 | 0.0 | 10 | |
| v15-infer-r1-ep130-prog0 | v15-r1-ep130 | 0.0 | 5 | R1 smoke |
| v15-infer-r1-ep130-prog0-r10 | v15-r1-ep130 | 0.0 | 10 | R1 扩大 |
| v15-infer-r2-ep170-prog0 | v15-r2-ep170 | 0.0 | 5 | R2 失败 |
| v15-infer-r1-seed7-best-prog0-r10 | v15-r1-seed7 best | 0.0 | 10 | seed7 best |
| v15-infer-r1-seed7-ep150-prog0-r10 | v15-r1-seed7-ep150 | 0.0 | 10 | seed7 ep150 |

## 评测口径

- **short/long 双套件**: `--envs forest_a --rand-two-suites`
- **随机起终点**: 由 `--seed` 控制推理随机性
- **runs=20 为正式口径**, runs=5/10 为 smoke
- **prog=0.0**: 推理时 `--forest-min-progress-m 0.0` (放松进度门控)
- **A\*-MPC 对照**: 每次推理自动生成, 无需单独 run
