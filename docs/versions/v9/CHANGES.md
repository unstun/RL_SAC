# v9 改动清单（相对 v7p1）

## 变更类型

- 本版为**奖励参数调优**，回归 CNN-DDQN 主线，不涉及算法/架构变更。
- 目标：通过增大曲率与转向惩罚，降低路径曲率，缩短路径长度和耗时。

## 关键差异（old → new）

| 参数 | v7p1 | v9 | 说明 |
|------|------|----|------|
| `forest_reward_k_kappa` | 0.2 | 0.5 | 曲率惩罚（`k_kappa * tan(δ)²`），2.5x |
| `forest_reward_k_delta` | 0.8 | 1.5 | 转向变化惩罚（`k_delta * (δ_next - δ_before)²`），~2x |
| `forest_reward_k_len` | 0.0 | 0.05 | 路径长度惩罚（每步距离 × k_len），新增 |
| profile 名 | `v7p1` | `v9` | — |
| 训练/推理输出目录 | `v7p1` | `v9` | — |

## 保持不变（关键参数口径）

- 训练/推理策略口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 奖励主参数：`forest_reward_k_t=0.10`
- gating 参数：`forest_topk=10`、`forest_adm_horizon=30`、`forest_min_progress_m=0.01`、`forest_min_od_m=0.02`
- 终止策略：`no_terminate_on_stuck=true`
- 网络结构：`hidden_layers=3`、`hidden_dim=256`
- DQfD 参数：全部保持 v7p1 口径

## 受影响文件

- 新增：`configs/v9.json`
- 新增：`docs/versions/v9/`（四件套）
- 更新：`docs/versions/README.md`（版本索引）

## 代码实现影响

- 无代码变更。所有改动均为配置参数调整。
