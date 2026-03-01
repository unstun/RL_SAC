# v13 CHANGES — 相对 v12 的改动

## 版本意图

系统性消融 `forest_adm_horizon` 参数（25/20/15/10），验证缩短掩码前瞻是否能缩短路径。

## 相对 v12 的具体变更

**代码层面：无新增代码改动。** 纯 CLI override 实验。

### 配置变更

- `forest_adm_horizon`: 30(V12) → 25/20/15/10（四组消融）
- 其他参数不变：`--dueling --mha --profile v9p2 --episodes 150`

## 变更文件

无代码文件变更。

## 关键参数快照

| 参数 | H25 | H20 | H15 | H10 |
|------|-----|-----|-----|-----|
| `forest_adm_horizon` | 25 | 20 | 15 | 10 |
| `--dueling` | True | True | True | True |
| `--mha` | True | True | True | True |
| `episodes` | 150 | 150 | 150 | 150 |
| `--profile` | v9p2 | v9p2 | v9p2 | v9p2 |
