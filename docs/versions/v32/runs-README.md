# V32 Runs 索引

| Run 目录 | 描述 | 关键结果 |
|----------|------|---------|
| `runs/v32-A-ddqn-shield/train_20260227_223513` | DDQN+shield 300ep | best ckpt |
| `runs/v32-B-mdqn-shield/train_20260227_225149` | M-DQN+shield 300ep | best ckpt |
| `runs/v32-A-smoke/20260227_225217` | V32-A 单模型 smoke | short 0.333, long 0.333 |
| `runs/v32-B-smoke/20260227_230555` | V32-B 单模型 smoke | short 0.667, long 1.00/47.71m |
| `runs/v32-ens-smoke/20260227_230811` | V16-C+V32-B negotiate K=10 | short 1.00/15.53m, long 1.00/46.10m |

## 评测口径

- smoke：runs=3，rand-two-suites（short+long），forest_a，H=45
- 训练在远端 ubuntu-zt，本地回传

## 结论

负面结果，不进 formal。V22-D 仍是全局最优。
