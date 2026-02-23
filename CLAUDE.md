# RL_sac/AGENTS.md（Ubuntu 24.04 + ros2py310）

> **作用域**：本文件适用于 `/home/sun/phdproject/dqn/RL_sac/**`。仓库通用说明见 `../AGENTS.md`。若冲突，以本文件为准。

## 0. 总原则（必须遵守）

1) **先计划后动手**

   - 写/改任何文件前：必须先输出 3–7 步「实施计划」+「将改动的文件清单」+「风险点」+「验证方式」。
   - 输出计划后：默认等待你明确确认（例如“开始/实现/按计划执行”）。如你在当次任务中明确允许“自检后直接开始”，则在完成约定自检并通过后开始实际修改。
   - 未进入实施前，仅允许做非侵入式探索（读文件、搜索、运行不改代码的检查/测试）。
2) **小步提交与可回滚**

   - 每次改动尽量小、单一目的；避免“顺手重构/顺手格式化全仓库”。
   - 涉及大量文件或大范围行为变化：拆成多个可独立评审的步骤。
3) **最小高置信变更**

   - 修 bug：优先“加失败测试 → 修复 → 全绿”。
   - 重构：必须保证行为不变（见验收标准），并说明如何验证。
5) **称呼约定**

   - 每次回复默认以“`帅哥，`”开头（除非你明确要求不需要）。

5.1) **沟通口吻与标识符释义**

- 默认用“研究生汇报”口吻：先说清楚我对问题的理解、假设与不确定性，再给可执行步骤与验证方式；避免营销式或口号式表达。
- 文本中首次出现不直观标识符（函数/类/变量/参数/CLI 选项/文件名等）时，在其后用括号补一句短解释（不超过 1 句）；重复出现可省略。
- 示例：`epsilon`（ε-greedy 的探索率/随机动作概率）、`gamma`（折扣因子）、`rollout_agent(...)`（用当前策略在环境里采样轨迹/回合）、`--runs 20`（评测重复次数）。
  5.2) **回复语言（硬约束）**
- 默认仅使用中文回复（包含计划、过程更新、结果总结、风险说明与验证结论）。
- 仅当你明确要求英文或中英双语时，才允许切换输出语言。
- 命令行、文件路径、参数名、代码标识符可保留英文原文，不视为违反“仅中文回复”。
  5.3) **`AGENTS.md` 与 `CLAUDE.md` 同步（硬约束）**
- 两个文件必须保持**逐行一致**，不得出现长期漂移。
- 任一文件发生修改时，必须在同一轮次同步修改另一文件。
- 提交前必须执行 `diff -u AGENTS.md CLAUDE.md`；若存在差异，禁止结束该轮修改。

6) **可复现性（配置）**

   - 默认规则：每次完成任何**代码改动**（含重构/性能优化/数值变更）后，都要新增一份“可复现配置”，保证你本人可复现（你可在当次任务中明确豁免）。纯文档改动默认不强制。
   - 推荐做法：在 `configs/` 新增 `repro_YYYYMMDD_<topic>.json`，并在其中记录：
     - 复现实验/自检命令（推荐 `conda run -n ros2py310 ...`）
     - 关键超参/seed、输入数据/地图、评估指标口径
     - 变更点摘要（可用 `_meta` 字段，避免被严格解析为 CLI 参数）
7) **环境约束**

   - 环境默认为 Ubuntu 24.04；Python 默认使用 conda 环境：`ros2py310`。
8) **仓库研究目标**

   - 最终目标：在森林场景车辆运动规划任务中，提出可复现、可验证的强化学习方法；在统一评测口径下相对多类基线（强化学习基线、传统规划+MPC 基线）达到更优综合性能，重点指标为 `success_rate`（越大越好）、`avg_path_length`（越小越好，**首要优化目标**）、`path_time_s`（越小越好）。
   - **曲率不作为独立优化目标**：路径长度缩短后曲率自然改善，不单独追踪 `curvature` 指标。
   - 方法边界：不限制于 `CNN-DDQN`；允许新增/替换模块或引入其他强化学习算法，但必须满足学术定义合规、可复现留档与门槛验证要求（见第 9、13、16 条）。
9) **学术定义合规（硬约束）**

   - 任何算法名/术语（如 DQN/DDQN/Double Q-learning、CNN 架构、Hybrid A*/RRT*、MPC 等）必须满足其学术或原论文定义；禁止“只改风格但不符合定义”的实现。
   - 若对定义或标准模糊：优先下载或整理对应论文 PDF 或官方实现仓库作为参考，并在说明中引用；建议放入 `paper/`（或记录可追溯链接/出处）。GitHub 仓库可单独建立文件夹。
10) **规则可追加**

- 以上规则可随时追加；新增规则以最新说明为准。

11) **README 同步（训练/推理命令）**

- 每次涉及代码或配置行为的改动后，默认同步更新 `README.md` 与 `README.zh-CN.md` 中“最新训练/推理命令”（含对应 profile 名）。
- 若命令未变化，也应明确检查并保持中英文两份 README 一致，避免漂移。

12) **时间优先验证流程**

- 深度强化学习迭代默认采用两阶段流程：`self-check -> smoke -> full runs=20`。
- 未经明确说明，每轮先完成 smoke（短训练+短评测）再决定是否进入 full 评测，避免直接长时全量实验。
- 推荐 smoke 命令使用 `conda run -n ros2py310 ...`，并限制在可快速回路内完成。

12.1) **远端优先执行（ubuntu-zt）**

- 训练/推理（含 `self-check`、`smoke`、`full`）默认先在 `ssh ubuntu-zt` 上执行；仅当远端不可用（如 SSH 失败、远端环境异常、远端资源不足）时，才回落本地执行以避免本机卡顿。
- 每次远端运行前，必须先执行“本地仓库 -> 远端仓库”同步；此处同步口径固定为本地覆盖远端（不包含 `runs/`）。
- 远端运行完成后，必须将远端 `runs/` 对应结果目录回传到本地 `runs/`，再进行后续分析与归档。

12.1.1) **远端 SSH 执行注意事项（已踩坑）**

- `conda run` 不会自动 cd 到项目目录，必须使用 `conda run --cwd <项目绝对路径>` 指定工作目录。
  - 正确：`ssh ubuntu-zt "conda run --cwd /home/sun/phdproject/dqn/RL_sac -n ros2py310 python train.py ..."`
  - 错误：`ssh ubuntu-zt "conda run -n ros2py310 python train.py ..."`（会在 `~` 下找不到 `train.py`）
- 远端 `~/.bashrc` 的 conda init 块必须放在 interactive guard（`case $- in`）之前，否则 SSH 非交互式命令无法找到 conda。
  - 已于 2026-02-21 修复本地与远端（ubuntu-zt）的 `~/.bashrc`，备份为 `~/.bashrc.bak.*`。

12.2) **版本标准工作流（默认）**

- 每次创建新版本（`vxpx`）前，必须先完成 GitHub 快照：`git status` clean、`git add/commit`、`git push` 成功。
- 每个新版本默认先跑固定 smoke 门：
  - 训练：`episodes=150`
  - 推理：`runs=3`（short/mid/long）
- 若 smoke 未显示明确收益（按当前目标：路径更短、计算时间更短），则：
  - 不进入 full 评测；
  - 主线代码/默认命令回退到上一稳定版本（例如回退到 `v7p1`）；
  - 当轮作为失败版本归档（四件套必须记录失败原因与证据路径）。
- 下一轮在失败版本基础上递增命名继续（例如 `v7p2p1` 失败后进入 `v7p2p2`）。

13) **最终研究门槛（硬约束）**

- 最终结论必须在 `short/long` 双套件、各 `runs=20` 条件下汇报。
- 对标 `Hybrid A*-MPC` 时，最终目标模型（当前规划为 `SAC-GlobalCNN`）至少满足：
  - `success_rate(SAC-GlobalCNN) >= success_rate(Hybrid A*-MPC)`
  - `avg_path_length(SAC-GlobalCNN) < avg_path_length(Hybrid A*-MPC)`
  - `path_time_s(SAC-GlobalCNN) < path_time_s(Hybrid A*-MPC)`
- 若任一套件未满足上述三条，视为未通过最终门槛。

15) **版本命名与留档（硬约束）**

- 版本命名统一采用 `vxpx`：**大改动**执行 `v+1`；**小改动**执行 `p+1`。
- 每个版本必须使用独立文件夹：`docs/versions/<version>/`。
- 每版至少包含：`README.md`（版本总结）、`CHANGES.md`（具体改动）、`RESULTS.md`（结果对比）、`runs/README.md`（对应 run 路径与口径）。
- 每个版本必须写 MD 留档（方法、详细修改点、参数、命令、结果、结论、下一步）。
- 推荐在新一轮开始前先读取上一版本留档，防止重复试验与口径漂移。





17) **文件写入限制（Claude Code 客户端问题，硬约束）**

- 单次 Write / Edit 工具调用写入内容不得超过 50 行；超过时必须拆成多次调用（先 Write 前 50 行，再用 Edit 追加后续内容）。
- 原因：Claude Code 客户端在单次写入过大时会静默报错或进入死循环，导致工具调用反复失败。
- 同理适用于生成大段代码、长 Markdown 文档等场景：宁可多调用几次，不要一次性写完。

18) **联网调研注意事项（Claude Code 客户端问题）**

- WebFetch 和 WebSearch 不混在同一批并行调用（WebFetch 403 会级联拖垮同批 WebSearch）。
- 每批并行最多 2 个同类调用。
- 优先 arXiv / GitHub 等开放源。
- PDF 链接大概率解析失败，优先用 HTML 版本（如 `arxiv.org/html/`）。
- **付费墙站点（tandfonline / sciencedirect / springer）**：WebFetch 会 403，改用 Playwright：
  1. `browser_navigate` 打开 URL
  2. `browser_wait_for` 等 5 秒（Cloudflare 自动验证）
  3. `browser_snapshot` 获取页面内容

## 默认环境

- 操作系统：Ubuntu 24.04
- Conda 环境：`ros2py310`
- 命令风格：优先 `conda run -n ros2py310 ...`
- 安装/快速开始/成功判定：优先参考 `../AGENTS.md`（避免重复维护）。

## 最小自检

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

## 验收标准（最小）

- 文档层面
  - `AGENTS.md` 结构清晰、无明显歧义；与 `../AGENTS.md` 不冲突（重复信息尽量引用上层）。
  - `AGENTS.md` 与 `CLAUDE.md` 保持逐行一致；可用 `diff -u AGENTS.md CLAUDE.md` 验证。
  - 引用的关键路径存在：`train.py`、`infer.py`、`configs/`、`paper/`。
- 过程层面
  - 后续任何实现改动均遵守：先计划后动手、小步变更、学术定义合规、可复现性约束（除非你明确豁免）。
  - 默认执行两阶段验证（smoke 优先），最终结论使用 short/long + runs=20 的硬门槛口径。
