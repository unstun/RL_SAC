# V27 CHANGES

## 改动性质

**零代码改动。** 所有基础设施（Dijkstra、obs 通道、CNN 2ch 支持）
在 V28（geodesic reward）实验时已实现。V27 仅通过 CLI flag 启用 obs 通道。

## 训练命令

```bash
# V27-A: DDQN + geodesic obs（V16-C 同参数）
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 --forest-reward-k-len 0.10 \
  --forest-reward-k-p 12.0 --forest-adm-horizon 45 \
  --obs-geodesic-goal-dist \
  --out v27-A-geodist --device cuda"

# V27-B: M-DQN + geodesic obs（如 A 成功）
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha --rl-algos cnn-mdqn \
  --eps-decay 100 --episodes 300 --forest-reward-k-len 0.30 \
  --forest-reward-k-p 12.0 --forest-adm-horizon 45 \
  --obs-geodesic-goal-dist \
  --out v27-B-mdqn-geodist --device cuda"
```

## Smoke 评测命令

```bash
# 单模型 smoke (runs=3)
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python infer.py --profile v9p2 \
  --models runs/v27-A-geodist/<train_dir> \
  --envs forest_a --rand-two-suites --runs 3 \
  --forest-adm-horizon 45 --obs-geodesic-goal-dist \
  --out v27-A-smoke --device cuda --seed 77"
```

**关键**: 推理时必须传 `--obs-geodesic-goal-dist`，否则 obs_dim 不匹配。
