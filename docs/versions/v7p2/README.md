# v7p2 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p1`
- 本版口径：`shielded/hybrid`（训练与推理规则保持不变）
- 对外表述（文档/论文）：`CNN-DDQN (shielded/hybrid inference)`，不宣称 `strict-argmax`
- 状态：**已运行 smoke（micro-smoke，episodes=40）+ full300（best/final）并完成 short/mid/long 各 runs=20 对比**

## 本版目标
- 修复 `AMRBicycleEnv`（Ackermann 车辆环境）中的部分马尔可夫性缺口：
  - 奖励含 `reward_k_a * (a - prev_a)^2 * v_scale`（加速度平滑惩罚）
  - 旧版观测不含 `prev_a`（上一时刻加速度命令）
- 保持奖励参数与策略规则不变，只补齐观测信息，避免引入额外行为漂移。

## 方法概要
- 环境观测最小修复：
  - 在 `forest_vehicle_dqn/env.py` 的 `_observe()` 中新增 `prev_a_n`（`prev_a` 归一化到 `[-1,1]`）
  - `AMRBicycleEnv` 观测维度从 `10 + N^2` 变为 `11 + N^2`
- 网络输入布局兼容：
  - 在 `forest_vehicle_dqn/networks.py` 的 `infer_flat_obs_cnn_layout(...)` 中将 bicycle 布局识别从 `10 + N^2` 更新为 `11 + N^2`
- 新增轻量回归测试（标准库 `unittest`）：
  - `tests/test_v7p2_markov_obs_prev_a.py`

## 本轮执行记录（2026-02-20）
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 smoke（micro-smoke）训练：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2 --episodes 40 --out v7p2_smoke --device cuda --progress"`
- 远端 smoke（micro-smoke）推理：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2 --models v7p2_smoke --runs 3 --out v7p2_smoke --progress"`
- 结果回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p2_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p2_smoke/`

## 关键命令
- 自检：
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- smoke（micro-smoke）：
  - `conda run -n ros2py310 python train.py --profile v7p2 --episodes 40 --out v7p2_smoke --device cuda --progress`
  - `conda run -n ros2py310 python infer.py --profile v7p2 --models v7p2_smoke --runs 3 --out v7p2_smoke`

## 代表 run
- 训练：`runs/v7p2_smoke/train_20260220_211732`
- 推理：`runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137`
- full300(best) 训练：`runs/v7p2_full300/train_20260220_213003`
- full300(best) 推理：`runs/v7p2_full300/train_20260220_213003/infer/20260220_214341`
- full300(final) 训练：`runs/v7p2_final300/train_20260220_215145`
- full300(final) 推理：`runs/v7p2_final300/train_20260220_215145/infer/20260220_220346`

## 本轮结果摘要（runs=3）
- short：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=20.0492`，`path_time_s=11.6833`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=17.0342`，`path_time_s=10.2667`
- mid：
  - CNN-DDQN：`success_rate=0.333`，`avg_path_length=26.9989`，`path_time_s=14.8000`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=24.0814`，`path_time_s=13.3333`
- long：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=65.6659`，`path_time_s=44.5333`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=43.0107`，`path_time_s=22.8167`

## 追加执行记录（2026-02-20，full300）
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

## full300 对比摘要（best -> final，runs=20）
- short：
  - `success_rate: 0.85 -> 0.80`（下降）
  - `avg_path_length: 20.2527 -> 19.8434`（变短）
  - `path_time_s: 13.2912 -> 12.7531`（变快）
- mid：
  - `success_rate: 0.65 -> 0.85`（显著提升）
  - `avg_path_length: 33.1484 -> 28.0888`（变短）
  - `path_time_s: 20.4538 -> 17.2441`（变快）
- long：
  - `success_rate: 0.75 -> 0.75`（持平）
  - `avg_path_length: 62.5007 -> 66.5743`（变长）
  - `path_time_s: 39.0567 -> 42.6033`（变慢）
  - `planning_time_s: 0.55093 -> 2.8190`（显著变慢）
- 结论：
  - `final` 不是全局更优，只在 mid 套件明显更好；
  - 如要稳定综合表现，当前仍应优先使用 `best` 作为主汇报模型，`final` 作为对照结果保留。

## 下一步
1. 以 `best` 作为 `v7p2` 主结果继续后续版本迭代，`final` 作为 ablation（消融对照）保留。
2. 定位 `final` 在 long 套件 `planning_time_s` 激增的触发条件（优先排查不可行动作率上升与重规划频率）。
3. 若下一版继续尝试 `final`，建议先做小规模 smoke（micro-smoke）对 long 进行定向验证再上 full。
