# V27 Runs 索引

| Run 目录 | 描述 | 关键结果 |
|----------|------|---------|
| `runs/v27-A-geodist/train_20260228_003702` | DDQN+geodist 300ep | ep170 early-stop, short 1.00/15.15m, long 0.667/49.89m |
| `runs/v27-B-mdqn-geodist/train_20260228_005335` | M-DQN+geodist 300ep | ep260 early-stop, short 0.333, long 0.667/65.48m |
| `runs/v27-A-smoke/20260228_005215` | V27-A 单模型 smoke | 见上 |
| `runs/v27-B-smoke/20260228_011411` | V27-B 单模型 smoke | 见上 |

## 评测口径

- smoke：runs=3，rand-two-suites（short+long），forest_a，H=45
- 训练在远端 ubuntu-zt

## 结论

负面结果，不进 formal。V22-D 仍是全局最优。
