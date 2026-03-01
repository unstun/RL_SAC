# v9p8 RESULTS（待填写）

## Smoke（runs=3）

| 指标 | Short RL | Short BL | Long RL | Long BL |
|------|----------|----------|---------|---------|
| SR | **33%** | 100% | **0%** | 100% |
| Path (m) | 13.13m | 17.03m | — | 43.01m |
| Time (s) | 8.20s | 10.27s | — | 22.82s |

## Full（runs=20）

未执行（smoke 未通过门槛）。

## 结论

**smoke 失败，v9p8 标记为失败版本。不进入 full 评测。**

失败原因：训练随机性——同参数（k_len=0.0, k_t=0.15）在 v9p7 消融 kt15 中
short/long SR=100%，但本次训练模型未收敛到同样水平。

根因分析：seed=21 + DQfD demo 采集的随机性导致本次训练质量差。
建议下一步：换 seed 或增加 episodes 重新训练。
