# V29 CHANGES

## 改动性质

**零代码改动。** 复用 V28 的 `--reward-geodesic-progress` flag。

## 训练命令

```bash
# V29-A: M-DQN + geodesic reward（300ep，V22-D 参数）
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --rl-algos cnn-mdqn --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.30 --forest-reward-k-p 12.0 \
  --forest-adm-horizon 45 --reward-geodesic-progress \
  --out v29-A-mdqn-geo --device cuda"
```

## 推理命令（smoke）

```bash
conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python infer.py --profile v9p2 --rl-algos cnn-ddqn \
  --models runs/v16-C-ep190 \
  --ensemble-models runs/v29-A-mdqn-geo/train_20260227_191102 \
  --ensemble-mode negotiate --negotiate-k 10 \
  --out v29-A-ens-K10-smoke --runs 3 --rand-two-suites \
  --envs forest_a --forest-adm-horizon 45 --device cuda --seed 77
```

## 与 V22-D 的差异

| 项目 | V22-D | V29-A |
|------|-------|-------|
| Proposer | V16-C DDQN | V16-C DDQN（不变） |
| Approver | V22-D M-DQN（欧式奖励） | V29-A M-DQN（测地奖励） |
| Approver k_len | 0.30 | 0.30 |
| Approver 奖励 | euclidean progress | geodesic progress |
| Ensemble mode | negotiate K=10 | negotiate K=10 |
