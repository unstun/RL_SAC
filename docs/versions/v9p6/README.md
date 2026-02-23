# V9P6: Cost-to-Go 引导 + Expert Exploration + CE Loss

## 版本定位

基于 V9P2，引入 dqn4 仓库分析得出的 5 项关键改进，目标是缩小 long 场景与 baseline 的差距。

## 核心改动

1. **Cost-to-go 观测通道**：地图从 1 通道(occ) → 2 通道(occ + Dijkstra cost-to-go)，标量从 10 → 11 维(新增 cost_n)
2. **Dijkstra progress 奖励**：progress 项从欧氏距离改为 cost-to-go 差（障碍物感知）
3. **Expert exploration**：训练早期以概率 0.7→0.0 混合专家动作（衰减到 60% episodes）
4. **DQfD CE loss**：新增 cross-entropy 行为克隆损失（demo_ce_lambda=1.0）
5. **Stuck 检测**：启用终端 stuck 检测（20 步位移<0.02m → 终止 + -300 惩罚）

## 预期效果

- Long 场景路径长度和时间缩短（cost-to-go 引导绕障更高效）
- 训练早期收敛更快（expert exploration 提供成功轨迹）
- 更强的模仿信号（CE loss 补充 margin loss）

## 命令

```bash
# Smoke
conda run -n ros2py310 python train.py --profile v9p6 --out v9p6-smoke1
conda run -n ros2py310 python infer.py --profile v9p6 --models v9p6-smoke1 --out v9p6-r3 --runs 3
# Full
conda run -n ros2py310 python infer.py --profile v9p6 --models v9p6-smoke1 --out v9p6-full --runs 20
```

## 状态

- [ ] Smoke 训练
- [ ] Smoke 推理 (runs=3)
- [ ] Full 评测 (runs=20)
