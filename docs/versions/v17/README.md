# V17 — V16-C 配置复训（失败版本）

## 版本定位

复训 V16 最优配置 C（H45+k_p=12+k_len=0.10），验证可复现性。

**零代码改动。** 配置与 V16-C 完全相同（第二轮增加 `--rl-early-stop-patience-points 999`）。

## 结论：失败

两轮训练均未能复现 V16-C 的结果：

### R1（默认 early-stop, ep210 停止）

- long SR 最高仅 0.8（V16-C = 1.0）
- 最佳 ep80: short 0.8/12.08m, long 0.8/50.48m

### R2（禁用 early-stop, 跑满 300 ep）

- ep200 出现 long SR=1.0 但 short path=22.40m（退化）
- 最佳平衡 ep300: short 1.0/14.54m, long 0.8/52.22m

### 根因

demo 采集随机性 + RL 探索噪声导致训练方差。

## V16-C 对照（ep190, runs=5 smoke）

| Suite | SR | Path |
|-------|----|------|
| short | 1.0 | 13.24m |
| long | 1.0 | 45.99m |

## 未进入 runs=20 正式评测（§12.2 失败版本流程）
