# V30 CHANGES

## 改动性质

**零代码改动。** 仅通过 `--obs-map-size N` CLI flag 控制分辨率。

## CNN 参数变化

| N | conv_out (3×3/4×4/6×6) | fc_in_dim | 总参数量 |
|---|------------------------|-----------|---------|
| 12 | 64×3×3=576 | 586 | ~0.41M |
| 16 | 64×4×4=1024 | 1034 | ~0.52M |
| 24 | 64×6×6=2304 | 2314 | ~0.84M |

## 训练命令

```bash
# B: N=16 DDQN
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 --forest-reward-k-len 0.10 \
  --forest-reward-k-p 12.0 --forest-adm-horizon 45 \
  --obs-map-size 16 --out v30-B-ddqn-n16 --device cuda"

# B: N=16 M-DQN
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha --rl-algos cnn-mdqn \
  --eps-decay 100 --episodes 300 --forest-reward-k-len 0.30 \
  --forest-reward-k-p 12.0 --forest-adm-horizon 45 \
  --obs-map-size 16 --out v30-B-mdqn-n16 --device cuda"

# C: N=24 DDQN
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 --forest-reward-k-len 0.10 \
  --forest-reward-k-p 12.0 --forest-adm-horizon 45 \
  --obs-map-size 24 --out v30-C-ddqn-n24 --device cuda"

# C: N=24 M-DQN
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha --rl-algos cnn-mdqn \
  --eps-decay 100 --episodes 300 --forest-reward-k-len 0.30 \
  --forest-reward-k-p 12.0 --forest-adm-horizon 45 \
  --obs-map-size 24 --out v30-C-mdqn-n24 --device cuda"
```
