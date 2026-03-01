# v9p8 — 消融驱动：去 k_len，保 k_t=0.15

## 版本依据

基于 v9p7 消融实验结论（2026-02-23）直接定版：

1. **去掉 k_len**：消融组 klen（k_len=0.02, k_t=0.0）mid/long SR=0%，
   证明路径惩罚是 mid/long 场景崩溃的根因，必须移除。
2. **保持 k_t=0.15**：kt15 组（k_len=0.0, k_t=0.15）在 short/long 双场景 SR=100%，
   符合最终门槛口径（只看 short/long），是消融中唯一满足门槛 SR 的方向。
3. **smoke 路径偏差可接受**：smoke runs=3 方差大，full runs=20 有统计意义。

## 参数（相对 v7p1）

| 参数 | v7p1 | v9p2 | v9p8 |
|------|------|------|------|
| forest_reward_k_len | 0.0 | 0.02 | **0.0** |
| forest_reward_k_t | 0.0 | 0.15 | **0.15** |

## 命令

```bash
# 训练
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python train.py \
  --profile v7p1 --episodes 150 --seed 21 --out v9p8-train \
  --device cuda --progress --save-ckpt best \
  --forest-reward-k-len 0.0 --forest-reward-k-t 0.15"

# smoke
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python infer.py \
  --profile v7p1 --models v9p8-train --out v9p8-r3 \
  --runs 3 --device cuda --progress"

# full
ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac \
  -n ros2py310 python infer.py \
  --profile v7p1 --models v9p8-train --out v9p8-full \
  --runs 20 --device cuda --progress"
```

## 状态

- [ ] 训练
- [ ] Smoke 推理 runs=3
- [ ] Full 推理 runs=20
