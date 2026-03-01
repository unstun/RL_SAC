# V35 CHANGES

## 零代码改动

V35 是纯推理消融实验，不修改任何代码。
复用 V22-D M-DQN + V32-B M-DQN+shield 模型。

## 实验配置

### V35-A：K 上扫
- 算法: V16-C cnn-ddqn + V32-B cnn-mdqn(shield) negotiate
- K: 15, 20
- 参数: --forest-adm-horizon 45

### V35-B：角色互换
- 算法: V22-D cnn-mdqn(proposer) + V32-B cnn-mdqn+shield(approver) negotiate
- K: 10, 15
- 参数: --forest-adm-horizon 45

## 与 V34-D 的区别

| 配置 | 提议者 | 审批者 | K |
|------|--------|--------|---|
| V34-D | V16-C DDQN | V32-B shield | 10 |
| V35-A | V16-C DDQN | V32-B shield | 15/20 |
| V35-B | V22-D M-DQN(k_len=0.30) | V32-B shield | 10/15 |
