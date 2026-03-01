# v15 CHANGES — 相对 v11 的改动

## 改动性质

**零代码改动。** 所有优化通过 CLI 参数 / 配置调整实现。

## 路径 1: Epsilon 调度修复

| 参数 | v11 值 | v15 值 | 理由 |
|------|--------|--------|------|
| `eps_decay` | 4500 | 100 | v11 训 150ep 时 ε≈0.194，全程 ~20% 随机动作 |
| `episodes` | 150 | 300 | 更多利用阶段（实际 ep130 早停最优） |

**诊断**：`eps_decay=4500` 从旧配置继承，与 150ep 训练严重错配。
修复后 ep100 时 ε=0.02，模型进入真正的利用阶段。

## 路径 2: Reward Shaping

| 参数 | v11 值 | v15 值 | 理由 |
|------|--------|--------|------|
| `reward_k_len` | 0.02 | 0.05 | 原值下路径惩罚相对进度奖励微乎其微 |

**测试范围**：k_len ∈ {0.05, 0.10}
- k_len=0.05: long path -8%，SR 不降 → **采纳**
- k_len=0.10: SR 崩溃至 0.2 → **排除**

## 路径 3: 推理时 min_progress_m 放松

| 参数 | v11 值 | v15 值 | 理由 |
|------|--------|--------|------|
| `min_progress_m`(推理) | 0.01 | 0.0 | 原值导致 15.6% inadmissible rate |

**效果**：prog=0.0 一致提升 long SR +0.1~0.2，无副作用。

## 训练命令（最佳配置 R1）

```bash
conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python train.py --profile v9p2 --dueling --mha \
  --eps-decay 100 --episodes 300 \
  --forest-reward-k-len 0.05 \
  --out v15-r1-klen005 --device cuda
```

## 推理命令（最佳配置）

```bash
conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 \
  python infer.py --profile v9p2 \
  --models runs/v15-r1-ep130/models \
  --envs forest_a --rand-two-suites --runs 20 \
  --forest-min-progress-m 0.0 \
  --out v15-formal-r1-ep130-prog0 --device cuda --seed 77
```

## 关键发现

- Checkpoint ep130 (非训练自动选择的 "best") 是最优推理检查点
- 必须手动扫描 periodic checkpoints，"best" 选择机制不可靠
- 训练结果高度依赖 seed（seed=21 好, seed=42 完全失败）
