# V27 — 测地目标距离场观测通道（Geodesic Goal Distance Field）

## 版本摘要

给 observation 新增一个"测地目标距离场"通道：每个栅格格子的值 =
该点沿自由空间到目标的 Dijkstra 最短路径距离，归一化到 [-1, +1]。
CNN 输入从 1 通道变 2 通道，其余架构不变。

## 动机

V22-D long path=46.11m vs A*-MPC=42.93m，差距 +7.4%。
agent 的 12×12 占据栅格只告诉它"哪里有树"，不告诉它"绕左边还是绕右边更近"。
A* 有全局最短路信息，RL agent 没有 → 在关键岔路口选了次优方向 → 绕路。

## 方法

- Ch0（现有）：占据栅格，-1=自由, +1=障碍
- Ch1（新增）：测地距离场，-1=目标处, +1=最远/不可达
- CNN Conv2d(1→32) 变为 Conv2d(2→32)，其余不变
- 仅在 episode reset 时计算一次 Dijkstra（~10ms），step 中零开销

## 消融矩阵

见 [CHANGES.md](CHANGES.md)。

## 主要结果

见 [RESULTS.md](RESULTS.md)。

## 对标

| 配置 | short SR | short path | long SR | long path |
|------|----------|------------|---------|-----------|
| V16-C (单 DDQN) | 0.85 | 14.43m | 1.00 | 46.00m |
| V22-D (ensemble) | 0.95 | 15.75m | 1.00 | 46.11m |
| A*-MPC | 0.95 | 15.77m | 1.00 | 42.93m |
| **V27 目标** | ≥0.85 | — | ≥0.80 | **<44m** |
