# v9p2 CHANGES

## 代码改动

无代码改动。仅通过 CLI 参数覆盖 v7p1 profile 的默认值。

## 参数变更

```
--forest-reward-k-len 0.02   # 路径长度惩罚（v7p1 默认 0.0）
--forest-reward-k-t   0.15   # 每步时间惩罚（v7p1 默认 0.0）
```

## 训练命令

```bash
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python train.py \
  --profile v7p1 --episodes 150 --out v9p2-smoke1 \
  --device cuda --progress --save-ckpt best \
  --forest-reward-k-len 0.02 --forest-reward-k-t 0.15"
```

## 推理命令

```bash
# smoke (runs=5)
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python infer.py \
  --profile v7p1 --models v9p2-smoke1 --out v9p2-smoke1-r5 \
  --runs 5 --device cuda --progress"

# full (runs=20)
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python infer.py \
  --profile v7p1 --models v9p2-smoke1 --out v9p2-full \
  --runs 20 --device cuda --progress"
```

## 失败变体记录

| 变体 | k_len | k_t | ep | short SR | long SR | 失败原因 |
|------|-------|-----|----|----------|---------|---------|
| v9p2p1 | 0.04 | 0.25 | 300 | 66.7% | 33.3% | 惩罚过重 |
| v9p2p2 | 0.02 | 0.20 | 300 | 100% | 0% | k_t 过重 |
| v9p2p3 | 0.02 | 0.15 | 300 | 33.3% | 0% | early-stop 选差 ckpt |
| v9p2p4 | 0.025 | 0.15 | 150 | 66.7% | 33.3% | k_len 微增即崩 |
| v9p2p5 | 0.02 | 0.16 | 150 | 66.7% | 100% | k_t 微增 short 崩 |
| seed=42 | 0.02 | 0.15 | 150 | 100% | 40% | 种子敏感 |
