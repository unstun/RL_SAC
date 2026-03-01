# v9p9 — save-ckpt final（修正 v9p8 checkpoint 选择偏差）

## 版本依据

v9p8 发现问题：`--save-ckpt best` 在 5 个固定训练地图上选 checkpoint，
导致模型对固定地图过拟合，推理碰到新地图泛化能力差。

本版本改用 `--save-ckpt final`，直接保存 ep=150 末尾权重，
消除对固定评测地图的选择偏差。

## 参数（相对 v7p1）

| 参数 | v7p1 | v9p8 | v9p9 |
|------|------|------|------|
| forest_reward_k_len | 0.0 | 0.0 | **0.0** |
| forest_reward_k_t | 0.0 | 0.15 | **0.15** |
| save_ckpt | auto | best | **final** |

## 命令

```bash
# 训练
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python train.py \
  --profile v7p1 --episodes 150 --seed 21 --out v9p9-train \
  --device cuda --progress --save-ckpt final \
  --forest-reward-k-len 0.0 --forest-reward-k-t 0.15"

# full（跳过 smoke，runs=3 方差不可信）
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python infer.py \
  --profile v7p1 --models v9p9-train --out v9p9-full \
  --runs 20 --device cuda --progress"
```

## 状态

- [ ] 训练
- [ ] Full 推理 runs=20
