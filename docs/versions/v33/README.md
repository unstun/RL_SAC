# V33 — TD3 连续动作空间消融实验

## 动机

V22-D (DQN ensemble, negotiate K=10) 是全局最优：
- short 0.95/15.75m, long 1.00/46.11m
- long path vs A*-MPC (42.93m) 差距 +7.4%

V23-V32 共 10 轮优化全部失败，根因分析指向：
- DQN 的 15×15=225 离散动作空间是根本限制
- 车只能走 225 个预设方向，无法输出最优连续控制量

## 方法

用 TD3 (Twin Delayed DDPG) 替换 DQN，输出连续动作 `(delta_dot, acceleration)`。

### TD3 vs SAC 关键差异

| 特性 | SAC | TD3 |
|------|-----|-----|
| 策略 | 随机高斯 + tanh | 确定性 tanh |
| 探索 | 熵驱动 | 加性高斯噪声 (σ=0.1) |
| Actor 更新 | 每步 | 延迟 (每 D=2 步) |
| 目标 Actor | 无 | 有 (慢速 polyak) |
| 目标平滑 | 无 | 有 (σ=0.2, clip=0.5) |

### 代码复用

| 组件 | 来源 | 复用方式 |
|------|------|---------|
| GlobalCNNEncoder | sac_networks.py | 直接 import |
| SACCritic (Twin Q) | sac_networks.py | 直接 import |
| SACReplayBuffer | sac_agent.py | 直接 import |
| 训练循环 | train_one_sac() | agent_override 参数注入 |
| 观测 | observe_sac() | 直接复用 (48×48 + 12标量) |
| 动力学 | step_continuous() | 直接复用 |

## 实验

### V33-A: TD3 基线 (150ep)

```bash
python train.py --profile v9p2 --rl-algos cnn-td3 \
  --episodes 150 --forest-reward-k-p 12.0 \
  --out v33-A-td3 --device cuda
```

## 结果

short SR=0%, long SR=0%，完全失败。详见 [RESULTS.md](RESULTS.md)。

## 结论

**负面结果。** TD3 在 150ep 训练预算下无法学会森林避障导航。
V22-D negotiate K=10 仍为全局最优。
