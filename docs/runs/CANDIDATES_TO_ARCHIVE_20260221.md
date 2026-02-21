# `runs/` 归档候选清单（2026-02-21）

> 说明：本清单只做“候选”登记，不执行删除。  
> 判定方法：目录名在 `README.md`、`README.zh-CN.md`、`docs/versions/**` 中未检索到引用。

## A. 高优先级候选（临时/排查目录，建议先归档）

- `runs/__tmp_parse_check/`
- `runs/_tmp_openpyxl_local/`
- `runs/_smoke_infer_mpc/`
- `runs/_smoke_infer_strict_dqfd/`
- `runs/smoke_strict_dqfd/`
- `runs/tmp_3base_planner_only_easy/`
- `runs/tmp_3base_planner_only_rrt200k/`
- `runs/tmp_6base_edge3/`
- `runs/tmp_6base_mpc_forest_tuned/`
- `runs/tmp_6base_mpc_slow05_h25/`
- `runs/tmp_6base_reject_hybrid/`
- `runs/tmp_6base_try_padding03_h18/`
- `runs/tmp_6base_try_tunedh18_s07/`
- `runs/tmp_check_6base_easy/`
- `runs/tmp_check_all_planners_100/`
- `runs/tmp_check_all_planners_100_iter100k/`
- `runs/tmp_midcover_defaultgating/`
- `runs/tmp_test_nobase/`
- `runs/tmp_v6p2_infer_v6gating/`
- `runs/tmp_v6p2_smoke/`
- `runs/tmp_v6p2p1_smoke/`
- `runs/tmp_v6p2p1_smoke_v6gating/`
- `runs/tmp_v6p2p1_v6gating/`

## B. 处理顺序建议

1. 先移动到人工归档区：`runs/_archive_manual/`。
2. 保留至少 1 个迭代周期（确认无人引用）。
3. 再执行永久删除。

## C. 建议命令（手动执行）

```bash
mkdir -p runs/_archive_manual

mv runs/__tmp_parse_check runs/_archive_manual/
mv runs/_tmp_openpyxl_local runs/_archive_manual/
mv runs/_smoke_infer_mpc runs/_archive_manual/
mv runs/_smoke_infer_strict_dqfd runs/_archive_manual/
```

## D. 删除前二次确认

```bash
rg -n -- "__tmp_parse_check|_tmp_openpyxl_local|tmp_v6p2_smoke" README.md README.zh-CN.md docs/versions
```

若命中为空，再进入下一步处理。
