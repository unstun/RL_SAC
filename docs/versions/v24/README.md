# V24 — Admissibility-Aware Fusion Pre-filter (FAILED)

## 目标

V23 发现瓶颈在 admissibility fallback（~30% inadmissible rate），
尝试在 ensemble fusion 前预过滤不可行动作，让两模型只在可行空间内达成共识。

## 方法

新增 `--ensemble-adm-prefilter` 标志。启用后：
1. 每步调用 `admissible_action_mask()`（向量化，便宜）
2. 将不可行动作 Q 值设为 -inf
3. 正常 fusion → argmax 天然可行

## 结果

**负面结果：路径更长、方差更大、计算更慢。**

| 指标 | V22-D baseline | V24 prefilter | 变化 |
|------|---------------|---------------|------|
| short SR | 0.95 | 0.95 | = |
| short path | **15.75m** | 15.88m | +0.13m |
| long SR | 1.00 | 1.00 | = |
| long path | 46.11m | 47.57m | **+1.46m** |
| inadm rate | 0.316 | 0.000 | 技术成功 |
| compute | 1.69s | 2.86s | +69% |

## 关键发现

1. inadmissible rate 0→0% — 技术上预过滤完全生效
2. 但路径长度和方差均恶化 — 预过滤改变了 negotiate 共识动态
3. M-DQN 在限缩空间内的 top-K 偏移 → 共识质量下降
4. 原 DDQN-based fallback 在 30% 步骤上反而优于 M-DQN 参与的选择
5. **消除 inadmissible ≠ 改善路径**

## 代码改动

`forest_vehicle_dqn/cli/infer.py`: +1 CLI flag `--ensemble-adm-prefilter` (~25 行)。
改动保留供后续研究。
