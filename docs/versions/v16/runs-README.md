# V16 Runs 目录

## 训练 Runs

| Run | 配置 | 说明 |
| --- | --- | --- |
| v16-h30-kp24 | H30, k_p=24, k_len=0.10 | 实验 B |
| v16-h45-kp12 | H45, k_p=12, k_len=0.10 | **实验 C（最优）** |
| v16-h45-kp24 | H45, k_p=24, k_len=0.10 | 实验 D |

注: 实验 A = V15-R2 (v15-r2-klen010), H30+k_p=12+k_len=0.10

## Checkpoint 副本

| Run | 来源 | 说明 |
| --- | --- | --- |
| **v16-C-ep190** | v16-h45-kp12 ep190 | **正式评测用 ckpt** |

## Smoke 推理 Runs (runs=5, seed=42)

| Run | 模型 | H | 说明 |
| --- | --- | --- | --- |
| v16-infer-B-best | v16-h30-kp24 best | 30 | long SR=0 |
| v16-infer-B-ep250 | v16-h30-kp24 ep250 | 30 | long SR=0.6 |
| v16-infer-B-ep260 | v16-h30-kp24 ep260 | 30 | long SR=0.8 |
| v16-infer-B-ep300 | v16-h30-kp24 ep300 | 30 | SR=1.0/1.0, path 长 |
| v16-infer-C-best | v16-h45-kp12 best | 45 | long SR=0.6 |
| v16-infer-C-ep050 | v16-h45-kp12 ep50 | 45 | SR=1.0/1.0 |
| v16-infer-C-ep100 | v16-h45-kp12 ep100 | 45 | short path=13.95m |
| v16-infer-C-ep150 | v16-h45-kp12 ep150 | 45 | long path=44.80m |
| v16-infer-C-ep180 | v16-h45-kp12 ep180 | 45 | long SR=0.2 |
| **v16-infer-C-ep190** | v16-h45-kp12 ep190 | 45 | **综合最优** |
| v16-infer-C-ep200 | v16-h45-kp12 ep200 | 45 | 退化 |
| v16-infer-D-best | v16-h45-kp24 best | 45 | long SR=0.2 |
| v16-infer-D-ep160 | v16-h45-kp24 ep160 | 45 | SR=1.0/1.0 |
| v16-infer-D-ep230 | v16-h45-kp24 ep230 | 45 | SR=0.4/0.6 |

## 扩大推理 (runs=10, seed=42)

| Run | 模型 | 说明 |
| --- | --- | --- |
| v16-infer-C-ep190-r10 | v16-C-ep190 | short SR=1.0, long SR=0.9 |

## 正式评测 (runs=20, seed=77)

| Run | 模型 | 说明 |
| --- | --- | --- |
| **v16-formal-C-ep190** | v16-C-ep190 | **最终正式评测** |

## 评测口径

- **short/long 双套件**: `--envs forest_a --rand-two-suites`
- **H=45**: `--forest-adm-horizon 45`（训练和推理必须一致）
- **prog=0.0**: `--forest-min-progress-m 0.0`
- **runs=20 为正式口径**, runs=5/10 为 smoke
