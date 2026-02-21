# v7p1 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v6p2p3`
- 本版口径：`shielded/hybrid`（训练与推理一致）
- 状态：**已运行（train=300 + short/mid/long 各 runs=5），待 full20**

## 本版目标
- 将 `v6p2p3` 的稳定策略参数切换为主线 profile 名 `v7p1`，作为后续迭代回退基线。
- 保持算法实现与关键参数不变，避免引入额外行为漂移。

## 方法摘要
- 主配置：`configs/v7p1.json`
- 对照配置：`configs/v6p2p3.json`
- 配置对比结论：除命名字段外（`train.out`、`infer.out`、`infer.models`），训练与推理参数保持一致。
  - 关键一致项：`forest_reward_k_t=0.10`、`forest_reward_k_delta=0.8`、`forest_no_fallback=false`、`forest_topk=10`、`forest_adm_horizon=30`、`forest_min_progress_m=0.01`、`forest_min_od_m=0.02`、`no_terminate_on_stuck=true`。

## 本轮关键命令
- 训练：
  - `conda run -n ros2py310 python train.py --profile v7p1 --episodes 300 --out v7p1_train300_esbest --device cuda --progress --save-ckpt best`
- 推理：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models v7p1_train300_esbest --out v7p1_train300_esbest --progress`
  - 说明：该次推理使用 profile 默认 `runs=5`（非最终门槛口径）。

## 代表 run
- 训练：`runs/v7p1_train300_esbest/train_20260221_010743`
- 推理：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927`
- KPI（均值）：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_raw.csv`

## 核心结果摘要（runs=5）
- short：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=16.7849`，`path_time_s=11.60`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=16.8724`，`path_time_s=10.00`
- mid：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=26.6106`，`path_time_s=16.60`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=25.1525`，`path_time_s=13.87`
- long：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=51.9081`，`path_time_s=34.01`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=43.0247`，`path_time_s=22.82`
- `failure_reason`（CNN / Hybrid）：均为 `reached=15`（short/mid/long 各 `reached=5`）。

## 结论与下一步
- `v7p1` 可作为稳定主线入口与回退基线；但在当前 `runs=5` 对比下，尚未满足相对 `Hybrid A*-MPC` 的路径时间优势门槛。
- 按最终研究门槛，仍需 short/long 各 `runs=20` 的独立证据作为最终结论。
