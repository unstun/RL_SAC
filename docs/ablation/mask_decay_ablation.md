# V14: Decaying Action Rollout 消融 (Mask Decay Ablation)

> **结论：负面结果**。衰减 rollout 在所有配置下均劣于 H30 无衰减基线（SR 下降且路径更长）。

## 1. 动机

V13 Horizon 消融发现：

- H30 常值 rollout 对极端转向画出 1.5s 巨弧，误杀近障碍物的紧凑操作 → 路径过长
- H10 成功时 path=15.40m < A\*-MPC 16.87m，证明 mask 是路径长的根因
- 但 H10 的 long SR=0.2，安全性不可接受

**假设**：在 rollout 中引入动作衰减（先保持 P 步，再线性衰减到 0），
可在保持安全性的同时允许更紧凑的操作。

## 2. 方法

### 2.1 Decaying Action Rollout

在 `_rollout_constant_actions_end_state()` 中：

```text
步骤 k < P:  action_k = (delta_dot, accel)          # 保持原动作
步骤 k >= P: action_k = (delta_dot × α, accel × α)  # 线性衰减
  α = max(0, 1 - (k-P)/(h-P))
```

- `action_persistence` (P): 动作保持步数，默认 0 (= 无衰减，等价旧行为)
- 代码改动: `env.py` (核心衰减逻辑), `train.py` / `infer.py` (CLI `--forest-adm-persistence`)

### 2.2 实验矩阵

两类实验：

1. **交叉推理诊断**: 用 V13-H15 已训练模型，推理时改变 horizon/persistence
2. **端到端训练**: 用新 mask 从头训练 150ep + 推理评测

## 3. 结果

### 3.1 交叉推理（模型固定 = V13-H15，仅改推理 mask）

| 推理 mask 配置 | short SR | short path (m) | long SR | long path (m) | inadmissible rate |
| --- | --- | --- | --- | --- | --- |
| **H30 无衰减** (baseline) | **1.0** | **19.45** | **1.0** | **55.24** | 0.14 / 0.28 |
| H20 无衰减 | 1.0 | 20.27 | 0.8 | 55.16 | 0.12 / 0.22 |
| P5-H30 (衰减) | 1.0 | 21.84 | 0.8 | 58.18 | 0.15 / 0.33 |
| P10-H30 (衰减) | 1.0 | 19.67 | 0.8 | 60.79 | 0.18 / 0.38 |
| H10 无衰减 | 0.6 | 15.88 | 0.2 | 54.45 | 0.04 / 0.15 |

### 3.2 端到端训练 + 推理

| 训练配置 | 推理配置 | short SR | short path (m) | long SR | long path (m) |
| --- | --- | --- | --- | --- | --- |
| P5-H30 | P5-H30 | 0.6 | 23.83 | 0.2 | 52.15 |

### 3.3 参考基线

| 算法 | short SR | short path (m) | long SR | long path (m) |
| --- | --- | --- | --- | --- |
| Hybrid A\*-MPC | 1.0 | 16.87 | 1.0 | 43.02 |

## 4. 分析

### 4.1 为什么衰减 rollout 不起作用

1. **rollout 轨迹不真实**：真实策略每 0.05s 决策一次，既不保持常量也不线性衰减。
   衰减假设（"转向后回正"）与实际策略行为不匹配，导致 mask 判断不可靠。
2. **安全裕度下降**：衰减后 rollout 轨迹更短/更温和 → 危险动作通过 mask →
   碰撞增加，long SR 下降。
3. **训练-推理一致性差**：端到端训练（P5-H30）在 150ep 内无法适应过于宽松的 mask，
   表现远差于交叉推理。

### 4.2 关键对比：各方案综合性能

| 方案 | short SR | short path (m) | long SR | long path (m) | 备注 |
| --- | --- | --- | --- | --- | --- |
| **H30 训练 + H30 推理** (V12 runs=20) | 0.95 | **16.41** | 0.95 | **47.34** | 路径最短 |
| H15 训练 + H30 推理 (runs=5) | 1.0 | 19.45 | 1.0 | 55.24 | SR 最高但路径长 +18% |
| P5-H30 衰减 (交叉推理) | 1.0 | 21.84 | 0.8 | 58.18 | SR 下降且路径更长 |
| P10-H30 衰减 (交叉推理) | 1.0 | 19.67 | 0.8 | 60.79 | long SR 下降 |
| P5-H30 端到端训练 | 0.6 | 23.83 | 0.2 | 52.15 | 全面最差 |
| Hybrid A\*-MPC | 1.0 | 16.87 | 1.0 | 43.02 | 对照基线 |

**结论**：
- 衰减 rollout 全面劣于无衰减，方案排除。
- H15→H30 解耦 SR 更高（1.0 vs 0.95），但路径显著更长（+18%）；
  考虑到路径长度是首要优化目标（CLAUDE.md §13），**H30→H30 仍为综合最优基线**。
- H15→H30 的 SR=1.0 基于 runs=5，统计置信度不如 H30→H30 的 runs=20。

## 5. Run 路径

| 实验 | 路径 |
| --- | --- |
| Cross H15→H30 (baseline) | `runs/v14-cross-h15trainH30infer/` |
| Cross H15→H10 | `runs/v14-cross-h15trainH10infer/` |
| Cross H15→P5-H30 | `runs/v14-cross-h15-p5h30/` |
| Cross H15→P10-H30 | `runs/v14-cross-h15-p10h30/` |
| Cross H15→H20 | `runs/v14-cross-h15-infH20/` |
| P5-H30 训练 | `runs/v14-smoke-p5h30/` |
| P5-H30 推理 | `runs/v14-infer-p5h30/` |
| 向后兼容检查 | `runs/v14-compat-check/` |

## 6. 下一步方向

衰减 rollout 方案已排除。保留 `--forest-adm-persistence` CLI 参数（default=0，向后兼容）。

后续路径长度优化应转向：

1. **H30→H30 (Dueling+MHA)** 保持为综合最优基线
2. **reward shaping** — 增加路径长度惩罚
3. **增加训练量** — 从 150ep 扩大到 500-1000ep
4. **连续动作空间** (SAC) — 避免离散化导致的路径锯齿
