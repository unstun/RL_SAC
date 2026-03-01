# V30 RESULTS — obs_map_size 消融

## 全配置对比

| 模型 | short SR | short path | long SR | long path | long corners |
|------|----------|------------|---------|-----------|-------------|
| V22-D N=12 (formal) | 0.95 | 15.75m | **1.00** | **46.11m** | — |
| V30-B N=16 (smoke) | 0.667 | 13.18m | 1.00 | 65.22m | **82** |
| V30-C N=24 (smoke) | 0.667 | 13.37m | 0.667 | 51.82m | 3 |
| A*-MPC (formal) | 0.95 | 15.77m | 1.00 | 42.93m | — |

## 关键发现

1. **N=16 灾难性失败**：long path=65.22m，82个拐角，agent 绕圈
   - inadmissible rate=32.1%，negotiate 无法消化如此多的无效动作
2. **N=24 部分失败**：long SR 降为 0.667，path=51.82m
   - inadmissible rate(short)=8.8%（比 N=12 低很多），但模型未充分收敛
3. **300ep 不够**：N=12 经历多轮调参才收敛，N=16/24 首次单次训练无法比拟
4. **obs_map_size 提升需配套更多 episodes 和超参调优**

## inadmissible rate 对比

| 配置 | short inad | long inad |
|------|-----------|----------|
| V22-D N=12 | ~22% | ~30% |
| V30-B N=16 | 28.2% | 32.1% |
| V30-C N=24 | **8.8%** | 40.5% |

N=24 short inadmissible rate 仅 8.8% 是好信号，
说明更高分辨率确实帮助 agent 避开障碍，但 long suite 不稳定。

## 结论

- 负面结果，不进 formal
- obs_map_size 扩大方向可能有效，但需要更多训练（500ep+）和超参重调
- 短期内改善 long path 的性价比低
- V22-D negotiate K=10 仍是全局最优
