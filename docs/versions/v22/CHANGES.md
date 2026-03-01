# V22 CHANGES — 相对 V21/V20 的改动

## 代码改动

**无代码改动** — 全部通过 CLI 参数实现。

## V22-A: K sweep 命令（推理）
```bash
conda run -n ros2py310 python -m forest_vehicle_dqn.cli.infer \
  --profile v9p2 --rl-algos cnn-ddqn \
  --models runs/v16-C-ep190 \
  --ensemble-models runs/v20-mdqn/train_20260225_041530 \
  --ensemble-mode negotiate --negotiate-k <K> \
  --out v22-A-K<K>-<smoke|formal> --runs <5|20> \
  --rand-two-suites --envs forest_a \
  --forest-adm-horizon 45 --device cuda
```

## V22-B: PBRS 训练命令
```bash
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python train.py --profile v9p2 --dueling --mha \
  --rl-algos cnn-mdqn --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.30 \
  --forest-reward-c-prog 12.0 --reward-potential-base <0|3.0> \
  --forest-adm-horizon 45 --out v22-B<N> --device cuda"
```

## V22-C: DQfD 增强训练命令
```bash
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python train.py --profile v9p2 --dueling --mha \
  --rl-algos cnn-mdqn --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.30 --demo-lambda 16.0 \
  --forest-adm-horizon 45 --out v22-C1 --device cuda"
```

## V22-D: M-DQN H=45 k_len=0.30 训练 + ensemble 推理
```bash
# 训练
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python train.py --profile v9p2 --dueling --mha \
  --rl-algos cnn-mdqn --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.30 --forest-reward-k-p 12.0 \
  --forest-adm-horizon 45 --out v22-D-mdqn-H45-kl030 --device cuda"

# Negotiate ensemble 推理
conda run -n ros2py310 python -m forest_vehicle_dqn.cli.infer \
  --profile v9p2 --rl-algos cnn-ddqn \
  --models runs/v16-C-ep190 \
  --ensemble-models runs/v22-D-mdqn-H45-kl030/train_20260227_023501 \
  --ensemble-mode negotiate --negotiate-k 10 \
  --out v22-D-ens-K10-formal --runs 20 --rand-two-suites \
  --envs forest_a --forest-adm-horizon 45 --device cuda
```
