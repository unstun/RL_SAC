# v7p2 runs 追溯

## 1. 本轮命令
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke（micro-smoke）训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2 --episodes 40 --out v7p2_smoke --device cuda --progress"`
- 远端 smoke（micro-smoke）推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2 --models v7p2_smoke --runs 3 --out v7p2_smoke --progress"`
- 结果回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2_smoke/`

## 2. run 路径登记
- self-check：
  - `run_dir`: `N/A`（自检不产出 run 目录）
  - `run.json`: `N/A`
- smoke（micro-smoke）train：
  - `run_dir`：`runs/v7p2_smoke/train_20260220_211732`
  - `run.json`：`runs/v7p2_smoke/train_20260220_211732/configs/run.json`
  - `train_meta`：`runs/v7p2_smoke/train_20260220_211732/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2_smoke/train_20260220_211732/train_flow.log`
- smoke（micro-smoke）infer：
  - `run_dir`：`runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137`
  - `run.json`：`runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137/configs/run.json`
  - KPI（均值）：`runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137/table2_kpis_mean_raw.csv`
  - KPI（逐回合）：`runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137/table2_kpis_raw.csv`

## 3. 结果登记
- self-check：
  - `train.py --self-check`：通过（CUDA 可用）
  - `infer.py --self-check`：通过（CUDA 架构支持）
- smoke（micro-smoke）train：
  - `episodes=40/40`，`stop_reason=completed`
  - 训练完成并保存模型：`runs/v7p2_smoke/train_20260220_211732/models/forest_a/cnn-ddqn.pt`
- smoke（micro-smoke，runs=3）：
  - short（CNN）：`success_rate=1.00`，`avg_path_length=20.0492`，`path_time_s=11.6833`
  - mid（CNN）：`success_rate=0.333`，`avg_path_length=26.9989`，`path_time_s=14.8000`
  - long（CNN）：`success_rate=1.00`，`avg_path_length=65.6659`，`path_time_s=44.5333`
  - short（Hybrid）：`success_rate=1.00`，`avg_path_length=17.0342`，`path_time_s=10.2667`
  - mid（Hybrid）：`success_rate=1.00`，`avg_path_length=24.0814`，`path_time_s=13.3333`
  - long（Hybrid）：`success_rate=1.00`，`avg_path_length=43.0107`，`path_time_s=22.8167`
  - `failure_reason`（CNN 汇总）：`reached=7`, `collision=1`, `timeout=1`
  - `failure_reason`（Hybrid 汇总）：`reached=9`

## 4. 备注
- 本轮遵循 `self-check -> smoke（micro-smoke）`，尚未进入 full（`runs=20`）门槛评测。
- `v7p2` 仅修复观测一致性，训练/推理策略口径保持 `shielded/hybrid`。

## 5. 追加运行（2026-02-20，full300 best vs final）

### 5.1 本轮命令
- 远端 full300 训练（best）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2 --episodes 300 --out v7p2_full300 --device cuda --progress --rl-early-stop-patience-points 9999 --rl-early-stop-warmup-episodes 9999"`
- 远端 full300 推理（best）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2 --models v7p2_full300 --runs 20 --out v7p2_full300 --progress"`
- 远端 full300 训练（final）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2 --save-ckpt final --episodes 300 --out v7p2_final300 --device cuda --progress --rl-early-stop-patience-points 9999 --rl-early-stop-warmup-episodes 9999"`
- 远端 full300 推理（final）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2 --models v7p2_final300 --runs 20 --out v7p2_final300 --progress"`
- 结果回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2_full300/ /home/sun/phdproject/dqn/dqn/runs/v7p2_full300/`
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2_final300/ /home/sun/phdproject/dqn/dqn/runs/v7p2_final300/`

### 5.2 run 路径登记
- full300(best) train：
  - `run_dir`：`runs/v7p2_full300/train_20260220_213003`
  - `run.json`：`runs/v7p2_full300/train_20260220_213003/configs/run.json`
  - `train_meta`：`runs/v7p2_full300/train_20260220_213003/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2_full300/train_20260220_213003/train_flow.log`
- full300(best) infer：
  - `run_dir`：`runs/v7p2_full300/train_20260220_213003/infer/20260220_214341`
  - `run.json`：`runs/v7p2_full300/train_20260220_213003/infer/20260220_214341/configs/run.json`
  - KPI（均值）：`runs/v7p2_full300/train_20260220_213003/infer/20260220_214341/table2_kpis_mean_raw.csv`
  - KPI（逐回合）：`runs/v7p2_full300/train_20260220_213003/infer/20260220_214341/table2_kpis_raw.csv`
- full300(final) train：
  - `run_dir`：`runs/v7p2_final300/train_20260220_215145`
  - `run.json`：`runs/v7p2_final300/train_20260220_215145/configs/run.json`
  - `train_meta`：`runs/v7p2_final300/train_20260220_215145/configs/train_meta_forest_a.json`
  - `train_flow_log`：`runs/v7p2_final300/train_20260220_215145/train_flow.log`
- full300(final) infer：
  - `run_dir`：`runs/v7p2_final300/train_20260220_215145/infer/20260220_220346`
  - `run.json`：`runs/v7p2_final300/train_20260220_215145/infer/20260220_220346/configs/run.json`
  - KPI（均值）：`runs/v7p2_final300/train_20260220_215145/infer/20260220_220346/table2_kpis_mean_raw.csv`
  - KPI（逐回合）：`runs/v7p2_final300/train_20260220_215145/infer/20260220_220346/table2_kpis_raw.csv`

### 5.3 结果登记（CNN-DDQN）
- full300(best)：
  - short：`success_rate=0.85`，`avg_path_length=20.2527`，`path_time_s=13.2912`
  - mid：`success_rate=0.65`，`avg_path_length=33.1484`，`path_time_s=20.4538`
  - long：`success_rate=0.75`，`avg_path_length=62.5007`，`path_time_s=39.0567`
  - `failure_reason`：short=`reached=17, timeout=1, collision=2`；mid=`reached=13, collision=4, timeout=3`；long=`reached=15, timeout=5`
- full300(final)：
  - short：`success_rate=0.80`，`avg_path_length=19.8434`，`path_time_s=12.7531`
  - mid：`success_rate=0.85`，`avg_path_length=28.0888`，`path_time_s=17.2441`
  - long：`success_rate=0.75`，`avg_path_length=66.5743`，`path_time_s=42.6033`
  - `failure_reason`：short=`reached=16, collision=4`；mid=`reached=17, collision=3`；long=`reached=15, timeout=4, collision=1`

### 5.4 备注
- `train_meta_forest_a.json` 显示：
  - full300(best) 的 `chosen_ckpt=best`
  - full300(final) 的 `chosen_ckpt=final`
- 结论口径：`final` 作为对照结果保留，`best` 仍是当前更稳妥的统一主结果。
