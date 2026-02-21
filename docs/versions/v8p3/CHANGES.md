# v8p3 CHANGES（相对 v8p2）

## 新增文件

- `forest_vehicle_dqn/plr_curriculum.py`：Syllabus PLR 课程 wrapper（ForestPLRCurriculum）
- `configs/v8p3.json`：v8p3 训练/推理配置
- `tests/test_tecrl.py`：TECRL 单元测试（3 个）
- `tests/test_plr_curriculum.py`：PLR 单元测试（3 个）

## 修改文件

### `forest_vehicle_dqn/sac_networks.py`
- 新增 `SACEntropyCritic` 类（twin Q 网络，结构同 SACCritic，用于估计累积熵）

### `forest_vehicle_dqn/sac_agent.py`
- `SACConfig`：新增 `use_tecrl`, `lr_entropy_critic`, `entropy_budget_ratio` 字段
- `SACAgent.__init__`：当 `use_tecrl=True` 时初始化 entropy_critic / target / optimizer / budget
- `update()`：入口分发，`use_tecrl=True` 时调用 `update_tecrl()`
- 新增 `update_tecrl()`：5 步更新（reward critic → entropy critic → actor → alpha → soft target）
- `save()`/`load()`：条件保存/加载 entropy critic 状态

### `forest_vehicle_dqn/env.py`
- `__init__`：新增 `reward_potential_base`, `reward_potential_bias` 参数
- `reset()`：记录 `_initial_dist`（初始距离）
- reward 计算：当 `reward_potential_base > 0` 时使用指数 potential `Φ = exp(base * progress)`

### `forest_vehicle_dqn/cli/train.py`
- 新增 8 个 CLI 参数（TECRL / PLR / exp-potential）
- `train_one_sac()` 签名扩展 + SACConfig 构造追加 TECRL 字段
- PLR 初始化 + episode 循环集成（sample → override reset_options → report）
- env 构造传递 `reward_potential_base`/`reward_potential_bias`
- 调用点传递所有新参数

## 配置变更（v8p2 → v8p3）

| 参数 | v8p2 | v8p3 |
|------|------|------|
| `sac_use_tecrl` | N/A | `true` |
| `sac_entropy_budget_ratio` | N/A | `0.6` |
| `sac_lr_entropy_critic` | N/A | `3e-4` |
| `reward_potential_base` | N/A | `32` |
| `reward_potential_bias` | N/A | `0.0` |
| `use_syllabus_plr` | N/A | `true` |
| `plr_levels` | N/A | `"6,10,14,20,30,42"` |
| `forest_curriculum` | `true` | `false` |
| `forest_train_two_suites` | `true` | `false` |
