# V24 RESULTS — Admissibility Pre-filter (FAILED)

## Smoke (runs=5)

| 配置 | short SR | short path | long SR | long path | inadm |
|------|----------|-----------|---------|-----------|-------|
| P1 prefilter | 1.0 | 17.52m | 1.0 | 47.63m | 0.000 |

## Formal (runs=20)

| 配置 | short SR | short path | long SR | long path | inadm | compute |
|------|----------|-----------|---------|-----------|-------|---------|
| V22-D baseline | 0.95 | **15.75m** | 1.00 | 46.11m | 0.316 | 1.69s |
| V24 prefilter | 0.95 | 15.88m | 1.00 | 47.57m | **0.000** | 2.86s |
| A*-MPC | 0.95 | 15.77m | 1.00 | 42.93m | — | 1.95s |

## Per-run 分析 (long)

V24 long paths: min=42.28m, max=61.22m, std=5.42m
两个 60m+ 异常值 (run 1: 61.22m, run 6: 61.00m)

## 分析

### 为何 prefilter 反而更差

1. **negotiate 共识动态被改变**: M-DQN top-K 在限缩空间内偏移
   - 原本 inadmissible 但 M-DQN 高排名的动作消失
   - 导致 negotiate 在不同（更差的）动作上达成共识
2. **70% 正常步骤无变化但增加开销**: admissible_action_mask 每步调用
3. **30% fallback 步骤质量下降**: 原 DDQN value-based fallback 反而更优
4. **方差翻倍**: 预过滤限制了决策灵活性，极端环境下路径恶化

### 核心洞察

**inadmissible rate ≠ 路径质量指标**。30% inadmissible 不是"坏"的，
而是 negotiate 在全动作空间找到最优共识后被 admissibility 修正的结果。
修正本身（DDQN-based fallback）质量合理，强行避免修正反而适得其反。

## 最终结论

1. **预过滤使路径更长**: long +1.46m, short +0.13m
2. **消除 inadmissible ≠ 改善路径**: 技术成功但实际效果为负
3. **DDQN value-based fallback 是合理的**: 不需要 M-DQN 参与 fallback
4. **V22-D negotiate K=10 仍是最优配置**
5. long path 的 gap (+7.4% vs A*-MPC) 可能已接近当前架构的能力极限
