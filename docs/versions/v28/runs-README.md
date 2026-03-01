# V28 Runs 索引

| Run 目录 | 描述 | 关键结果 |
|----------|------|---------|
| `runs/v28-A-baseline/train_20260227_162338` | baseline DDQN 150ep | best ckpt |
| `runs/v28-B-geodesic/train_20260227_162342` | geodesic reward 150ep | best ckpt |
| `runs/v28-B-geo-ep300/train_20260227_165922` | geodesic reward 300ep（主要） | best ckpt |
| `runs/v28-A-smoke/20260227_165606` | A baseline smoke runs=3 | short 1.00/22.41m, long 0.667/57.75m |
| `runs/v28-B-smoke/20260227_165634` | B geodesic 150ep smoke runs=3 | short 1.00/19.03m, long 0.667/45.41m |
| `runs/v28-B-ep300-smoke/` | B2 geodesic 300ep smoke runs=3 | short 0.667/15.89m, long 1.00/47.93m |

## 评测口径

- smoke：runs=3，rand-two-suites（short+long），forest_a，H=45
- 全部在远端 ubuntu-zt 训练，本地回传
