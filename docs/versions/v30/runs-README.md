# V30 Runs 索引

| Run 目录 | 描述 | 关键结果 |
|----------|------|---------|
| `runs/v30-B-ddqn-n16/train_20260227_200621` | N=16 DDQN 300ep | best ckpt |
| `runs/v30-B-mdqn-n16/train_20260227_204255` | N=16 M-DQN 300ep | best ckpt |
| `runs/v30-C-ddqn-n24/train_20260227_200621` | N=24 DDQN 300ep | best ckpt |
| `runs/v30-C-mdqn-n24/train_20260227_204255` | N=24 M-DQN 300ep | best ckpt |
| `runs/v30-B-ens-n16-smoke/` | N=16 ensemble smoke | short 0.667/13.18m, long 1.00/65.22m |
| `runs/v30-C-ens-n24-smoke/` | N=24 ensemble smoke | short 0.667/13.37m, long 0.667/51.82m |

## 评测口径

- smoke：runs=3，rand-two-suites（short+long），forest_a，H=45
- 全部在远端 ubuntu-zt 训练，本地回传

## 结论

负面结果，不进 formal。obs_map_size 扩大方向代价高，暂不继续。
