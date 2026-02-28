# V33 RESULTS — TD3 连续动作空间消融

## 对标

| 配置 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| V16-C (单 DDQN) | 0.85 | 14.43m | 1.00 | 46.00m |
| V22-D (ensemble K=10) | 0.95 | 15.75m | 1.00 | 46.11m |
| A*-MPC | 0.95 | 15.77m | 1.00 | 42.93m |

## V33-A: TD3 基线 (smoke runs=3)

| 模型 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| V33-A TD3 | **0.00** | N/A | **0.00** | N/A |
| A*-MPC | 1.00 | 12.95m | 1.00 | 42.95m |

**完全失败**——所有 6 次 rollout 均碰撞退出。

### 训练日志分析

- 150ep, 13m42s, best_return=-228.3
- Q 值: q1 从 -0.39 → -2.01（持续下降，未收敛）
- eval: ep50=-1002, ep100=-322960, ep150=-145372
- best ckpt 在 ep49（ret=-228），之后未改善

### 失败原因

1. **样本不足**: 150ep ≈ 90K 步，TD3 一般需 100K+ 步收敛
2. **探索不足**: σ=0.1 高斯噪声在复杂森林中效率低
3. **Q 值发散**: eval_return 暴跌至 -322960
4. **无 demo 引导**: DQN 有 DQfD，TD3 从零探索

## V33-B: TD3-Local (V16-C obs + BC 预训练, smoke runs=3)

| 模型 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| V33-B TD3-Local | **0.00** | N/A | **0.00** | N/A |
| A*-MPC | 1.00 | 12.95m | 1.00 | 42.95m |

**完全失败**——short: collision=2/timeout=1; long: collision=2/timeout=1。

### 训练日志分析

- 300ep, 11m16s, BC 预训练 5000步, Critic warmup 2000步
- BC 采集: 50ep → 11981 demo transitions
- best_return=-146.4 (ep249), feat_dim=586 (9×64 conv + 10 scalars)
- Q 值: q1 从 0→-4.89（持续下降，后期大量 catastrophic returns）
- 后期大量 -10000~-100000 级别回合（Q-landscape 发散）

### 失败原因

1. **TD3 样本效率低**: 连续动作空间需要更多训练步才能收敛
2. **Q 值发散**: 后期训练不稳定，catastrophic returns 持续增加
3. **BC 不足以引导**: 5000步 BC 后，critic warmup 后仍无法维持稳定学习
4. **根本问题**: V16-C 局部观测已被证明有效，但离散 DQN 比连续 TD3 在此任务上样本效率高得多

## 总结

**V33 完全失败（A+B 均 SR=0）。**
V22-D negotiate K=10 仍为全局最优（short 0.95/15.75m, long 1.00/46.11m）。

TD3 连续动作空间在本任务（复杂森林导航）中：
- 全局观测（V33-A）：样本不足 + 无 demo，SR=0
- 局部观测+BC（V33-B）：即使有 demo 引导，300ep 仍不收敛，SR=0
- 结论：**离散动作 DQN 在本任务优于连续 TD3**，后续不再追求 TD3 方向
