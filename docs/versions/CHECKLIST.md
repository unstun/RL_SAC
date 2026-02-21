# 版本归档执行清单（v1+ 通用）

> 目的：把“创建版本 → 运行实验 → 留档归档 → 索引同步”收敛成固定动作，降低遗漏与口径漂移。

## 0) 术语口径（先对齐）

- `self-check`：导入/设备快速自检，不产出结论。
- `micro-smoke`：超快 sanity（如 `episodes=40`），不等价标准 smoke 门。
- `smoke`：标准筛查门（训练 `episodes=150` + 推理 `runs=3`）。
- `full`：最终结论门（short/long 各 `runs=20`）。

## 1) 开始改版本前（必须完成）

- [ ] `git status` 为 clean（若不 clean，先 `commit` 或 `stash`）。
- [ ] `git push` 成功（远端存在可回退快照）。
- [ ] （推荐）打快照 tag：`<version>-pre`，并 `git push --tags`。
- [ ] 明确版本命名：`vxpx`（大改动 `v+1`，小改动 `p+1`）。

## 2) 运行阶段（每次运行都要留痕）

- [ ] 记录完整命令（train/infer，含 profile 与 argv 覆盖）。
- [ ] 记录 `run_dir` 与 `configs/run.json` 路径。
- [ ] 记录 KPI 工件路径：
  - [ ] `table2_kpis_mean_raw.csv`（门槛对比/表格汇总）
  - [ ] `table2_kpis_raw.csv`（逐回合诊断/`failure_reason`）
- [ ] 若运行失败：指标写 `N/A`，并写明失败原因（报错/超时/人工终止）。

## 3) 版本四件套（`docs/versions/<version>/`）

- [ ] `README.md`：目标、方法、关键命令、代表 run、结论、下一步。
- [ ] `CHANGES.md`：相对上一版 `old -> new` 改动与影响文件。
- [ ] `RESULTS.md`：short/long 指标、基线对比、门槛检查、`failure_reason` 分布。
- [ ] `runs/README.md`：所有代表性 run 的路径索引（命令+run_dir+工件）。

## 4) 索引同步（同轮完成）

- [ ] 更新根目录 `README.md` 的“版本总索引”。
- [ ] 更新 `README.zh-CN.md` 对应索引。
- [ ] 更新 `docs/versions/README.md` 镜像索引。
- [ ] 若是 `v9+`：确认三处索引均包含版本号、目录、主配置、关键 run、short/long 最佳 SR、基线 SR、状态。

## 5) 结果口径检查（发布前）

- [ ] 主线命名与实现一致：`strict-argmax` vs `shielded/hybrid` 不混用。
- [ ] 最终结论只使用 `full`（short/long 各 `runs=20`）证据。
- [ ] `table2_kpis_mean_raw.csv` 用于结论；`table2_kpis_raw.csv` 用于诊断。
- [ ] 若仅有 baseline-only（`--skip-rl`）：在版本文档中显式标注，RL 指标写 `N/A`。

## 6) 快速校验命令（建议每轮执行）

```bash
diff -u AGENTS.md CLAUDE.md
rg -n -- "STABLE_PROFILE|SMOKE_GATE|FINAL_GATE|REPORT_ARTIFACTS" README.md README.zh-CN.md
rg -n -- "episodes=40|micro-smoke|smoke" README.md README.zh-CN.md docs/versions/README.md
```

## 7) 最小通过标准

- [ ] 四件套齐全且路径可点击可追溯。
- [ ] 三处索引已同步，且口径一致无冲突。
- [ ] 本轮新增运行已归档（成功或失败均有记录）。
- [ ] 约束文件一致性通过：`diff -u AGENTS.md CLAUDE.md` 无输出。
