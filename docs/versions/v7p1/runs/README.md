# v7p1 runs 追溯

## 1. 本轮命令
- 训练：
  - `conda run -n ros2py310 python train.py --profile v7p1 --episodes 300 --out v7p1_train300_esbest --device cuda --progress --save-ckpt best`
- 推理：
  - `conda run -n ros2py310 python infer.py --profile v7p1 --models v7p1_train300_esbest --out v7p1_train300_esbest --progress`
  - 说明：profile 默认 `runs=5`。

## 2. run 路径登记
- train：
  - `run_dir`：`runs/v7p1_train300_esbest/train_20260221_010743`
  - `run_json`：`runs/v7p1_train300_esbest/train_20260221_010743/configs/run.json`
  - `train_meta`：`runs/v7p1_train300_esbest/train_20260221_010743/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p1_train300_esbest/train_20260221_010743/train_flow.log`
- infer：
  - `run_dir`：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927`
  - `run_json`：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_raw.csv`

## 3. 关键参数快照（run.json）
- train：
  - `profile=v7p1`，`episodes=300`，`save_ckpt=best`，`device=cuda`
  - `forest_no_fallback=false`，`forest_topk=10`，`forest_adm_horizon=30`
  - `forest_reward_k_t=0.10`，`forest_reward_k_delta=0.8`
- infer：
  - `profile=v7p1`，`models=v7p1_train300_esbest`，`runs=5`
  - `envs=[forest_a::short, forest_a::mid, forest_a::long]`
  - `baselines=[hybrid_astar_mpc]`
  - `forest_no_fallback=false`，`forest_topk=10`，`forest_adm_horizon=30`

## 4. short/long 关键指标
- short：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=16.7849`，`path_time_s=11.60`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=16.8724`，`path_time_s=10.00`
- long：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=51.9081`，`path_time_s=34.01`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=43.0247`，`path_time_s=22.82`

## 5. `failure_reason` 分布
- CNN-DDQN：
  - short：`reached=5`
  - mid：`reached=5`
  - long：`reached=5`
- Hybrid A*-MPC：
  - short：`reached=5`
  - mid：`reached=5`
  - long：`reached=5`
