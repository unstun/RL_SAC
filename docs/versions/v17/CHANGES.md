# V17 CHANGES — 相对 V16 的改动

## 改动性质

**零代码改动，零参数改动。** 纯复训实验。

## 参数（与 V16-C 完全相同）

| 参数 | 值 |
|------|----|
| `reward_k_len` | 0.10 |
| `reward_k_p` | 12.0 |
| `forest_adm_horizon` | 45 |
| `eps_decay` | 100 |
| `episodes` | 300 |
| `--dueling --mha` | 启用 |

## 训练命令

### R1（默认 early-stop）

```bash
conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.10 --forest-reward-k-p 12.0 \
  --forest-adm-horizon 45 \
  --out v17 --device cuda
```

### R2（禁用 RL early-stop）

```bash
conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.10 --forest-reward-k-p 12.0 \
  --forest-adm-horizon 45 \
  --rl-early-stop-patience-points 999 \
  --out v17 --device cuda
```

## 训练环境

- **本地 GPU**（RTX 3070 Ti），远端 ubuntu-zt 不可达
- R1: 26min16s, early-stop at ep210（patience=6）
- R2: 42min01s, 跑满 300 episodes
- 两轮 pretrain 均 2000/30000 步 early-stop（正常）
