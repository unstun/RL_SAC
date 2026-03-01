# V29 RESULTS — Geodesic Reward M-DQN Ensemble

## V29-A 训练进度（ep=240 中间快照）

| 指标 | 值 |
|------|----|
| sr(short/long/all) | 1.00/1.00/1.00 |
| argmax_inad(short/long/all) | 0.142/0.174/0.158 |
| 训练时长 | 20m36s |

训练时 inadmissible rate 仅 14-17%，看似健康。

## V29-A Ensemble Smoke（runs=3, negotiate K=10）

| 套件 | SR | path | inadmissible rate |
|------|----|------|-------------------|
| short | 1.00 | 13.40m | 43% |
| long | 1.00 | 48.50m | **51%** |

**推理时 inadmissible rate 暴涨到 43-51%**（训练时仅 14-17%）。
说明 geodesic reward 训练的 M-DQN 在推理分布下偏好走训练时未见过的方向。

## 全配置对比

| 模型 | short SR | short path | long SR | long path |
|------|----------|------------|---------|-----------|
| V29-A ensemble (smoke) | 1.00 | 13.40m | 1.00 | 48.50m |
| V22-D K=10 (formal) | 0.95 | 15.75m | **1.00** | **46.11m** |
| A*-MPC (formal) | 0.95 | 15.77m | 1.00 | 42.93m |

## 关键发现

1. **short path 13.40m 看似亮眼**：但 smoke 噪声大（runs=3），不可靠
2. **long path 48.50m 劣于 V22-D 46.11m**：训练→推理 inadmissible 率跳变是根因
3. **Geodesic reward 对 M-DQN approver 有害**：approver 的职责是"在提议列表中选好的"，
   不是"自主导航"；geodesic reward 训练使其偏好 Q-landscape 与 proposer 不兼容
4. **不进 formal 评测**

## §13 门槛检查（smoke，非正式）

| 条件 | 要求 | V29-A smoke | 通过? |
|------|------|-------------|-------|
| SR(short) >= A* | >=0.95 | 1.00 | 是（smoke 噪声） |
| SR(long) >= A* | >=1.00 | 1.00 | 是（smoke 噪声） |
| path(short) < A* | <15.77m | 13.40m | 是（smoke 噪声） |
| path(long) < A* | <42.93m | 48.50m | **否（+13%）** |

## 结论

- V29-A 是负面结果，long path 退步
- V22-D negotiate K=10 仍是全局最优
- Long path gap（46.11m vs 42.93m，+7.4%）可能是局部观测架构的固有上限
- 后续方向：扩大观测视野 或 引入全局路径提示 或 转入写作
