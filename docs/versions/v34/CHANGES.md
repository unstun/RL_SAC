# V34 CHANGES

## 零代码改动

V34 是纯推理消融实验，不修改任何代码。
利用 V32-B 已训练的 M-DQN+shield 模型做 formal 评测。

## 实验配置

### V34-C
- 算法: cnn-mdqn（V32-B M-DQN+shield）
- 模型: runs/v32-B-mdqn-shield/train_20260227_225149
- 参数: --forest-adm-horizon 45 --runs 20 --seed 42

### V34-D
- 算法: cnn-ddqn（V16-C）+ cnn-mdqn（V32-B shield approver）
- 命令关键参数: --models runs/v16-C-ep190 --ensemble-models runs/v32-B-mdqn-shield/...
- 模式: negotiate K=10, H=45, runs=20, seed=42

## 踩坑记录

V34-C 第一次跑忘记 --forest-adm-horizon 45，用了 v9p2 默认的 H=30。
结果 short SR=0.85（偏低）。重跑 H=45 后 short SR=0.90，符合预期。
结论：评测时必须传与训练一致的 --forest-adm-horizon。
