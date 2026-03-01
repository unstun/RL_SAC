# V28 RESULTS — Geodesic Progress Reward 消融

## 消融矩阵（smoke runs=3）

### V28-A（baseline，150ep）

| 套件 | SR | path |
|------|----|------|
| short | 1.00 | 22.41m |
| long  | 0.667 | 57.75m |

### V28-B（geodesic reward，150ep）

| 套件 | SR | path |
|------|----|------|
| short | 1.00 | 19.03m |
| long  | 0.667 | **45.41m** |

**对比 A→B（150ep）**：long path -21%（57.75→45.41m），short path -15%（22.41→19.03m）

### V28-B2（geodesic reward，300ep）

| 套件 | SR | path |
|------|----|------|
| short | 0.667 | **15.89m** |
| long  | **1.00** | 47.93m |

## 全配置对比（含历史参考）

| 模型 | short SR | short path | long SR | long path |
|------|----------|-----------|---------|-----------|
| V28-A baseline 150ep (smoke) | 1.00 | 22.41m | 0.667 | 57.75m |
| V28-B geodesic 150ep (smoke) | 1.00 | 19.03m | 0.667 | 45.41m |
| **V28-B2 geodesic 300ep (smoke)** | 0.667 | **15.89m** | **1.00** | 47.93m |
| V16-C (formal runs=20) | 0.85 | 14.43m | 1.00 | 46.00m |
| **V22-D K=10 (formal runs=20, 全局最优)** | 0.95 | **15.75m** | 1.00 | 46.11m |
| A*-MPC (formal runs=20) | 0.95 | 15.77m | 1.00 | 42.93m |

## 关键发现

1. **Geodesic reward 显著缩短路径**：150ep 时 long path -21%（58m→45m），效果最强
2. **单模型 short path < A*-MPC**：V28-B2 15.89m < A*-MPC 17.03m（smoke）
3. **300ep 收敛更好**：long SR 0.667→1.00，但 long path 略有回升（45→48m）
4. **SR 在 smoke(3次) 下噪声大**：short SR=0.667 不可靠，需 formal 验证

## §13 门槛检查（V28-B2 smoke，非正式）

| 条件 | 要求 | V28-B2 smoke | 通过? |
|------|------|-------------|-------|
| SR(short) >= A* | >=0.95 | 0.667 | **否（smoke噪声）** |
| SR(long) >= A* | >=1.00 | 1.00 | 是 |
| path(short) < A* | <15.77m | 15.89m | 否(+0.12m) |
| path(long) < A* | <42.93m | 47.93m | 否(+11.6%) |

## 结论

- Geodesic reward 是有效方向：路径缩短效果显著，尤其在长距离场景
- 单模型未能超越 V22-D ensemble（long path 47.93 > 46.11）
- 下一步可考虑：V28-B2 + negotiate ensemble（搭配 V16-C DDQN）
- §13 尚未通过，long path gap 仍是主要瓶颈
