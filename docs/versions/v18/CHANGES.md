# V18 CHANGES — 相对 V16 的改动

## 改动性质

**零代码改动，零参数改动。** 纯复训实验（V16-C 配置第三次独立训练）。

## 参数（与 V16-C 完全相同）

| 参数 | 值 |
|------|-----|
| `reward_k_len` | 0.10 |
| `reward_k_p` | 12.0 |
| `forest_adm_horizon` | 45 |
| `eps_decay` | 100 |
| `episodes` | 300 |
| `--dueling --mha` | 启用 |

## 训练命令

```bash
conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.10 --forest-reward-k-p 12.0 \
  --forest-adm-horizon 45 \
  --out v16-C-r3 --device cuda
```

## 训练环境

- **本地 GPU**（RTX 3070 Ti），远端 ubuntu-zt 不可达
- 训练时长: ~34min, early-stop at ep250（patience=6）
- 训练中 6 次 all=1.0（ep120, ep140, ep220, ep230, ep240, ep250）
- Pretrain: 2000/30000 步 early-stop（val_sr≥0.50，正常）

## 推理命令（runs=20 正式评测）

```bash
conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python infer.py --profile v9p2 \
  --models runs/v16-C-r3-ep240/models \
  --envs forest_a --rand-two-suites --runs 20 \
  --forest-min-progress-m 0.0 --forest-adm-horizon 45 \
  --out v16-C-r3-formal-ep240 --device cuda --seed 77
```
