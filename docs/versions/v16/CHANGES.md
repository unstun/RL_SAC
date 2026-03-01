# V16 CHANGES — 相对 V15 的改动

## 改动性质

**零代码改动。** 所有优化通过 CLI 参数 / 配置调整实现。

## 参数变化

| 参数 | V15 值 | V16 值 | 理由 |
|------|--------|--------|------|
| `reward_k_len` | 0.05 | **0.10** | 加大路径惩罚，V15-R2 在 H30 下失败 |
| `forest_adm_horizon` | 30 | **45** | 更长视距掩码，防止走进无法逃脱的窄道 |
| `reward_k_p` | 12.0 | 12.0 (不变) | 消融实验发现 k_p=24 过大 |
| `eps_decay` | 100 | 100 (不变) | V15 已验证 |
| `episodes` | 300 | 300 (不变) | V15 已验证 |
| `min_progress_m`(推理) | 0.0 | 0.0 (不变) | V15 已验证 |

## 消融矩阵（2×2, k_len=0.10 固定）

| 实验 | H | k_p | 最佳 ckpt | short SR/path | long SR/path |
|------|---|-----|----------|--------------|-------------|
| A (V15-R2) | 30 | 12 | ep170 | 0.60/19.12m | 0.20/66.31m |
| B | 30 | 24 | ep260 | 0.80/16.46m | 0.80/49.67m |
| **C** | **45** | **12** | **ep190** | **1.00/13.24m** | **1.00/45.99m** |
| D | 45 | 24 | ep160 | 1.00/17.28m | 1.00/48.16m |

（以上为 smoke runs=5 结果）

## 关键发现

- k_len=0.10 在 H30 下失败（V15-R2），但在 H45 下正常工作
- H45 的更长视距让模型提前"看到"窄道末端，避免走入死胡同
- k_p=24 的双倍进度奖励反而让 short 路径更长（模型过于追求进度）

## 训练命令（最佳配置 C）

```bash
conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.10 --forest-reward-k-p 12.0 \
  --forest-adm-horizon 45 \
  --out v16-h45-kp12 --device cuda
```

## 推理命令（最佳配置）

```bash
conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python infer.py --profile v9p2 \
  --models runs/v16-C-ep190/models \
  --envs forest_a --rand-two-suites --runs 20 \
  --forest-min-progress-m 0.0 \
  --forest-adm-horizon 45 \
  --out v16-formal-C-ep190 --device cuda --seed 77
```
