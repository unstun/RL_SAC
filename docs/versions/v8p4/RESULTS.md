# v8p4 RESULTS

## 数据来源

- KPI 文件：`runs/v8p4-smoke1/train_20260222_090846/infer/20260222_101831/table2_kpis_mean_raw.csv`

## Smoke 推理结果（1000ep 训练，runs=3）

| Suite | Algorithm | SR | avg_path_length | path_time_s |
| --- | --- | --- | --- | --- |
| short | CNN-SAC | 0.0 | N/A | N/A |
| short | Hybrid A*-MPC | 1.0 | 17.03 | 10.27 |
| mid | CNN-SAC | 0.0 | N/A | N/A |
| mid | Hybrid A*-MPC | 1.0 | 24.08 | 13.33 |
| long | CNN-SAC | 0.0 | N/A | N/A |
| long | Hybrid A*-MPC | 1.0 | 43.01 | 22.82 |

## 训练关键指标

| 指标 | 值 |
| --- | --- |
| best_return | -230.6 |
| alpha（最终） | 0.010（alpha_min 下界生效） |
| q1（最终） | -1.36 |
| tq（最终） | -1.36 |
| 训练耗时 | 1h04m |
| 训练 episodes | 1000 |

## 门槛检查

- SR(CNN-SAC) = 0% < SR(Hybrid A*-MPC) = 100% → **未通过**

## failure_reason 分布

所有 CNN-SAC 推理均以 timeout（超时未到达目标）结束。

## 与 v8p3 对比

| 指标 | v8p3 | v8p4 | 变化 |
| --- | --- | --- | --- |
| alpha（最终） | 0.003 | 0.010 | alpha_min 下界生效 |
| SR | 0% | 0% | 无变化 |
| best_return | N/A（膨胀失真） | -230.6 | base=4 修复膨胀 |
| tq 显示 | 0.00（key 缺失） | -1.36 | log key 对齐 |
