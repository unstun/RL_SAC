# V32 CHANGES

## 改动性质

**零代码改动。** 仅通过 `--forest-action-shield` CLI flag 控制。

## 训练命令

```bash
# V32-A: DDQN + shield
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 --forest-reward-k-len 0.10 \
  --forest-reward-k-p 12.0 --forest-adm-horizon 45 \
  --forest-action-shield \
  --out v32-A-ddqn-shield --device cuda"

# V32-B: M-DQN + shield
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha --rl-algos cnn-mdqn \
  --eps-decay 100 --episodes 300 --forest-reward-k-len 0.30 \
  --forest-reward-k-p 12.0 --forest-adm-horizon 45 \
  --forest-action-shield \
  --out v32-B-mdqn-shield --device cuda"
```

## Ensemble smoke 命令

```bash
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python infer.py --profile v9p2 \
  --models runs/v16-C-ep190 \
  --ensemble-models runs/v32-B-mdqn-shield/train_20260227_225149 \
  --envs forest_a --rand-two-suites --runs 3 \
  --forest-adm-horizon 45 --ensemble-mode negotiate --negotiate-k 10 \
  --out v32-ens-smoke --device cuda --seed 77"
```
