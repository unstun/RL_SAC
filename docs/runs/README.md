# `runs/` 目录整理与使用说明（2026-02-21）

> 目标：先让结果“找得到、讲得清、可追溯”，再做物理清理。

## 1) 当前目录现状（快照）

- 顶层目录约 `285` 个，混合了主线版本、`repro_*` 快照、临时目录（`tmp_*` / `_tmp*`）与历史别名目录。
- 当前推荐主线结果入口：
  - `runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/`
  - 均值 KPI（门槛对比）：`table2_kpis_mean_raw.csv`
  - 逐回合 KPI（失败诊断）：`table2_kpis_raw.csv`

## 2) 顶层目录分类规则（从本轮开始）

1. `v*`：版本链主目录（例如 `v7p1_*`），优先保留。
2. `repro_*`：一次实验一次目录的可复现快照，默认保留到完成版本归档后。
3. `tmp_*` / `_tmp*` / `__tmp*`：临时排查目录，默认候选清理。
4. `outputs*`：汇总导出目录，保留。
5. `forest_*`：历史别名目录，按“是否已在版本文档被引用”决定保留/归档。

## 3) 工件定位（训练/推理）

典型结构：

```text
runs/<out>/<train_ts>/
  configs/run.json
  train_flow.log
  models/
  infer/<infer_ts>/
    table2_kpis_mean_raw.csv
    table2_kpis_raw.csv
```

- `configs/run.json`（运行快照）用于还原命令与参数覆盖。
- `train_flow.log`（训练流程日志）用于排查中断与阶段耗时。
- `table2_kpis_mean_raw.csv`（均值表）用于 short/long 门槛与横向比较。
- `table2_kpis_raw.csv`（明细表）用于 `failure_reason`（失败类型）与逐样本诊断。

## 4) 清理策略（安全版）

先文档化后删除，避免把版本证据删掉：

1. 先在 `docs/versions/<version>/runs/README.md` 标注“代表 run”。
2. 再将无引用目录加入 `docs/runs/CANDIDATES_TO_ARCHIVE_20260221.md`。
3. 每次只处理一小批目录，先 `mv` 到归档区，再观察 1 轮迭代后再删。

> 强约束：任何出现在 `README.md` / `README.zh-CN.md` / `docs/versions/**` 的 `run_dir` 不可直接删除。

## 5) 建议的归档区（可选）

- 建议先建：`runs/_archive_manual/`
- 仅移动“确认不再引用”的目录（优先 `tmp_*`）。

示例（只演示，不在本轮自动执行）：

```bash
mkdir -p runs/_archive_manual
mv runs/tmp_v6p2_smoke runs/_archive_manual/
```

## 6) 与版本文档联动

- 每次运行结束，同轮更新对应版本四件套（尤其 `docs/versions/<version>/runs/README.md`）。
- 若 `run_dir` 发生替换（重跑），需同步更新三处索引：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`
