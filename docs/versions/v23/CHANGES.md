# V23 CHANGES — Negotiate 融合模式消融

## 代码改动

**文件**: `forest_vehicle_dqn/cli/infer.py`（唯一修改文件）

### 新增 `negrelax` mode (渐进松弛 negotiate)
- `_ensemble_fuse()` 新增 elif 分支
- 从 K_min 到 K_max 步长 5 逐步松弛
- 新增 CLI: `--negotiate-k-max`（默认 30）

### 新增 `negsoft` mode (软投票 negotiate)
- DDQN rank score + M-DQN softmax 加权求和
- 新增 CLI: `--negsoft-alpha`（默认 0.7）, `--negsoft-tau`（默认 0.1）

### 参数传递链
- `_ensemble_fuse` 签名 +3 参数: negotiate_k_max, negsoft_alpha, negsoft_tau
- `rollout_agent` 签名 +3 参数
- argparse choices 列表 +2 项
- main 调用处 +3 参数传递

## 训练命令

无训练（推理消融实验）。

## 推理命令

```bash
# negrelax
python -m forest_vehicle_dqn.cli.infer --profile v9p2 --rl-algos cnn-ddqn \
  --models runs/v16-C-ep190 \
  --ensemble-models runs/v22-D-mdqn-H45-kl030/train_20260227_023501 \
  --ensemble-mode negrelax --negotiate-k 10 --negotiate-k-max 30 \
  --runs 20 --rand-two-suites --envs forest_a --forest-adm-horizon 45

# negsoft
python -m forest_vehicle_dqn.cli.infer --profile v9p2 --rl-algos cnn-ddqn \
  --models runs/v16-C-ep190 \
  --ensemble-models runs/v22-D-mdqn-H45-kl030/train_20260227_023501 \
  --ensemble-mode negsoft --negsoft-alpha 0.7 --negsoft-tau 0.10 \
  --runs 20 --rand-two-suites --envs forest_a --forest-adm-horizon 45
```
