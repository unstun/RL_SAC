# V29 Runs 索引

| Run 目录 | 描述 | 关键结果 |
|----------|------|---------|
| `runs/v29-A-mdqn-geo/train_20260227_191102` | V29-A M-DQN+geodesic 300ep 训练 | best ckpt |
| `runs/v29-A-ens-K10-smoke/20260227_193153` | V16-C+V29-A negotiate K=10 smoke | short 1.00/13.40m, long 1.00/48.50m |

## 评测口径

- smoke：runs=3，rand-two-suites（short+long），forest_a，H=45
- 全部在远端 ubuntu-zt 训练，本地回传

## 结论

负面结果，不进 formal。V22-D 仍是全局最优。
