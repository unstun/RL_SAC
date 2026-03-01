# V21 CHANGES — 相对 V20/V16-C 的改动

## 代码改动

**无代码改动** — 全部通过 CLI 参数实现。

## 关键参数变更

### 训练命令模板
```bash
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha --rl-algos cnn-mdqn \
  --eps-decay 100 --episodes 300 \
  --forest-reward-k-len <0.20|0.30> \
  --forest-reward-k-p 12.0 \
  --forest-adm-horizon <60|80> \
  --out v21-mdqn-H<H>-kl0<kl> --device cuda"
```

### 推理命令（关键：必须传 --forest-adm-horizon）
```bash
conda run -n ros2py310 python -m forest_vehicle_dqn.cli.infer \
  --profile v9p2 --rl-algos cnn-mdqn \
  --models <train_dir> --out <name> \
  --runs 20 --rand-two-suites --envs forest_a \
  --forest-adm-horizon 60 --device cuda
```

### Negotiate Ensemble 命令
```bash
conda run -n ros2py310 python -m forest_vehicle_dqn.cli.infer \
  --profile v9p2 --rl-algos cnn-ddqn \
  --models runs/v16-C-ep190 \
  --ensemble-models runs/v21-mdqn-H60-kl030/train_20260226_234042 \
  --ensemble-mode negotiate --negotiate-k 20 \
  --out <name> --runs 20 --rand-two-suites --envs forest_a \
  --forest-adm-horizon 60 --device cuda
```

## 踩坑记录

### H 不匹配 Bug（严重）
- v9p2 profile 不设置 `forest_adm_horizon`，argparse 默认值=15
- 模型训练用 H=60，推理不传 `--forest-adm-horizon 60` 时用 H=15
- 表现：short SR=0.4, long SR=0.0（灾难性结果）
- 修复后：short SR=1.0, long SR=1.0（smoke）
