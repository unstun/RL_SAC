# v13 — Horizon 参数消融（Dueling+MHA 固定）

- 版本类型：**Major (v12→v13)**
- 研究路线：`CNN-DDQN` + Dueling + MHA，消融 `forest_adm_horizon`
- 状态：**未通过最终门槛**
- 上一版本：`v12`

## 概要

- 目标：系统性消融 horizon=25/20/15/10，验证缩短掩码前瞻能否缩短路径长度。
- 结果：H15 short SR=1.0 最优但 long SR 仅 0.4；H10 成功时路径最短（15.40m < A\*-MPC 16.87m）但 SR 过低；H30 仍为 SR 综合最优。
- 决策：无配置全面优于 H30。Horizon 参数调优不能同时兼顾 SR 和路径长度。

## 方法意图

- V12 路径长 4-10%，用户肉眼观察到 RL 在障碍物附近过度避让（大转弯）。
- 根因是 `forest_adm_horizon=30`（常值动作前瞻 1.5s）过于保守。
- 本版系统性缩短 horizon 验证能否放宽掩码、减少绕行。

## 复现命令

```bash
# 训练 (以 H15 为例)
conda run -n ros2py310 python train.py --profile v9p2 --dueling --mha \
  --forest-adm-horizon 15 --out v13-h15
# 推理 (runs=5 smoke)
conda run -n ros2py310 python infer.py --profile v9p2 --models v13-h15 \
  --forest-adm-horizon 15 --envs forest_a --rand-two-suites --runs 5 --out v13-infer-h15
```

## 结论 / 下一步

- Horizon 缩短能改善路径长度（H10 成功时 15.40m < A\*-MPC 16.87m），但严重损害 long SR。
- H30 仍为最优综合配置（SR=0.95/0.95）。
- 下一步：需结构性改进而非参数调优（如连续动作空间 SAC、或动态 horizon 机制）。
