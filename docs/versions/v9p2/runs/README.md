# v9p2 Runs 目录

## 训练

| Run | 路径 | 说明 |
|-----|------|------|
| v9p2-smoke1 | `runs/v9p2-smoke1/train_20260222_134346/` | 主训练（150ep, seed=21） |

## 推理

| Run | 路径 | 说明 |
|-----|------|------|
| v9p2-smoke1 (r3) | `runs/v9p2-smoke1/train_20260222_134346/infer/` | smoke runs=3 |
| v9p2-smoke1-r5 | `runs/v9p2-smoke1-r5/20260222_151219/` | smoke runs=5 |
| v9p2-full | `runs/v9p2-full/20260222_154658/` | **full runs=20** |

## 失败变体 runs（仅远端保留）

- `runs/v9p2p1-smoke1/` — k_len=0.04, k_t=0.25, 300ep
- `runs/v9p2p2-smoke1/` — k_len=0.02, k_t=0.20, 300ep
- `runs/v9p2p3-smoke1/` — k_len=0.02, k_t=0.15, 300ep
- `runs/v9p2p4-smoke1/` — k_len=0.025, k_t=0.15, 150ep
- `runs/v9p2p5-smoke1/` — k_len=0.02, k_t=0.16, 150ep
- `runs/v9p2-seed42/` — seed=42, 150ep

## 口径

- 训练：profile=v7p1, episodes=150, seed=21, save-ckpt=best
- 推理：profile=v7p1, random-start-goal, two-suites (short/mid/long)
- Baseline：Hybrid A*-MPC（同 seed 同 start/goal pairs）
