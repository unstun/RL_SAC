# v7p1 改动清单（相对 v6p2p3）

## 变更类型
- 本版为**主线命名收敛**，不是算法/实现重构。
- 目标是将稳定配置从 `v6p2p3` 统一为主线 profile `v7p1`。

## 关键差异（old -> new）
- profile 名：`v6p2p3` -> `v7p1`
- 训练输出目录默认名：`train.out: v6p2p3 -> v7p1`
- 推理输出目录默认名：`infer.out: v6p2p3 -> v7p1`
- 推理模型目录默认名：`infer.models: v6p2p3 -> v7p1`

## 保持不变（关键参数口径）
- 训练/推理策略口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 奖励主参数：`forest_reward_k_t=0.10`、`forest_reward_k_delta=0.8`
- gating 参数：`forest_topk=10`、`forest_adm_horizon=30`、`forest_min_progress_m=0.01`、`forest_min_od_m=0.02`
- 终止策略：`no_terminate_on_stuck=true`

## 受影响文件
- `configs/v7p1.json`（主配置入口）
- `configs/v6p2p3.json`（对照配置，确认参数一致性）

## 代码实现影响
- 本版归档未记录到相对 `v6p2p3` 的算法代码变更（`forest_vehicle_dqn/**` 行为口径保持一致）。
