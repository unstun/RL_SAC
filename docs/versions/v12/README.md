# v12 — Dueling+MHA 冻结配置（V11 消融最优）

- 版本类型：**Major (v11→v12)**
- 研究路线：`CNN-DDQN` + Dueling + Spatial MHA
- 状态：**未通过最终门槛**
- 上一版本：`v11`

## 概要

- 目标：冻结 V11 消融实验最优配置 (Config C: Dueling+MHA)，进行 runs=20 正式评测。
- 结果：short SR=0.95 (=A\*-MPC), long SR=0.95 (<1.0 A\*-MPC)；路径长度长 4-10%，路径时间慢 14-17%；计算速度快 3.5-5.9x。
- 决策：未通过门槛。需根本性改进（如连续动作空间/SAC）才能超越 A\*-MPC。

## 方法意图

- V11 实现了 5 个即插即用模块，通过 7 组消融实验确定 Dueling+MHA 为最优组合。
- V12 冻结该配置，进行正式评测并探索 horizon 参数调优。
- Dueling（价值/优势分流）+ MHA（空间多头自注意力）仍为标准 CNN-DDQN 架构增强，满足学术定义。

## 复现实验配置 / 命令

- 配置文件：`configs/v9p2.json`（基础 profile）
- 训练：
  ```bash
  ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
    python train.py --profile v9p2 --dueling --mha --episodes 150 --run-name v12-duel-mha"
  ```
- 推理（runs=20, short/long）：
  ```bash
  ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
    python infer.py --profile v9p2 --run-name v11-abl-duel-mha \
    --envs forest_a --rand-two-suites --runs 20 --out-dir v12-formal"
  ```

## 代表性运行

- 训练 run: `runs/v11-abl-duel-mha/`（复用 V11 消融训练）
- 推理 run: `runs/v11-formal-duel-mha/20260224_160845/`
- KPI: `runs/v11-formal-duel-mha/20260224_160845/table2_kpis_mean.csv`

## 结论 / 下一步

- V12 (Dueling+MHA) 相比 baseline v9p2 提升 +188% SR，但仍未超越 A\*-MPC。
- Horizon 参数调优（15/10）因 DQfD 专家耦合导致 SR 崩塌，证明参数微调无法突破瓶颈。
- 下一步建议：(1) 用短 horizon 重新生成专家 demo 再训练，(2) 增加训练至 500-1000ep，(3) 转向连续动作空间（SAC）。
