# V23 — Negotiate 融合模式消融 (negrelax + negsoft)

## 目标

改进 negotiate 融合算法降低 long path 方差和均值。
V22-D negotiate K=10 已达 long SR=1.00, short path < A*，但 long path=46.11m > A* 42.93m (+7.4%)。

## 方法

### negrelax（渐进松弛 negotiate）
无共识时逐步放宽 K（10→15→20→25→30），避免直接回退纯 DDQN。

### negsoft（软投票 negotiate）
DDQN 按排名给分 + M-DQN 按 softmax 概率给分，加权求和后 argmax。消除硬否决。

## 结果

**负面结果：两个新模式均未改善 long path。**

- negrelax: 与 negotiate K=10 完全相同（K=10 在 225 动作空间内总能达成共识，松弛从未触发）
- negsoft: long path 微减 0.15m，但 short path 反而 +0.10m，路径拐角从 0 飙升到 36

## 关键发现

**瓶颈不在融合算法，而在 admissibility fallback。**
融合后 argmax 动作 ~30% inadmissible → top-K/mask 替代产生绕路。

## 代码改动

`forest_vehicle_dqn/cli/infer.py`: 新增 `negrelax`/`negsoft` 两个 ensemble mode（~50 行）。
改动保留在代码中供后续研究使用。
