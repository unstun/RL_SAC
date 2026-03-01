# V24 CHANGES — Admissibility Pre-filter

## 代码改动

**文件**: `forest_vehicle_dqn/cli/infer.py`（唯一修改文件）

### 新增 `--ensemble-adm-prefilter` flag
- `rollout_agent` 签名 +1 参数: `ensemble_adm_prefilter: bool = False`
- 预过滤逻辑: 在 `_ensemble_fuse` 调用前，用 `admissible_action_mask()` 把不可行动作 Q 设为 -inf
- 新增 debug 指标: `prefilter_active_steps`
- CLI: `--ensemble-adm-prefilter` (store_true)
- main 调用处传参

### 参数传递链
- argparse → main → rollout_agent → prefilter block → _ensemble_fuse

## 训练命令

无训练（推理消融实验）。

## 推理命令

```bash
python -m forest_vehicle_dqn.cli.infer --profile v9p2 --rl-algos cnn-ddqn \
  --models runs/v16-C-ep190 \
  --ensemble-models runs/v22-D-mdqn-H45-kl030/train_20260227_023501 \
  --ensemble-mode negotiate --negotiate-k 10 \
  --ensemble-adm-prefilter \
  --runs 20 --rand-two-suites --envs forest_a --forest-adm-horizon 45
```
