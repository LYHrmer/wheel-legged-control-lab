# 一次可复算的 PPO 更新

按[更新学习实验](../../docs/ppo_update_lab.md)运行默认配置：seed 7，CPU float64，8 条
合成一阶系统样本，独立 Gaussian actor 与 critic 共 7 个参数。一次普通 SGD，学习率
0.01，clip_range=0.2，vf_coef=0.5，ent_coef=0。没有优势归一化或梯度裁剪。

复跑时使用新目录：

```bash
PYTHONPATH=src:.local-deps python3 examples/ppo_update_walkthrough.py \
  --output results/my_ppo_update
```

| 固定批次指标 | 更新前 | 更新后 |
| --- | ---: | ---: |
| actor loss | −0.835175622 | −0.839055165 |
| value MSE | 0.769464500 | 0.746670223 |
| total loss | −0.450443372 | −0.465720053 |

所有 7 个参数发生了变化。例如 actor bias：

```text
before = -0.05
gradient ≈ -0.451891961
after = before - 0.01 * gradient ≈ -0.045481080
```

精确值以 [parameters.csv](parameters.csv) 为准，可对每行核对同一个 SGD 等式。
[samples.csv](samples.csv)保留每条样本的 GAE、return、ratio 和 value；
[update.json](update.json)还保存状态转移、软件版本、全部参数梯度与源文件 SHA。

初始当前策略与采样策略相同，所以八条 ratio 都是 1，这次梯度并没有遇到裁剪平台。
单独的 [clipping_fixture.csv](clipping_fixture.csv) 检查正负优势下的非平凡分支，不参与
实际更新。更新后的 ratio 也不是每条都增加，例如第 0 条约为 0.99662；所有样本共享参数，
单条正优势并不保证该条概率在批量更新后一定上升。

这些损失只衡量原来的冻结批次。没有重新评估新策略的回报，没有生成机器人 checkpoint，
不代表单轮或 D1 控制性能提高。原有 SB3 训练入口不受影响。

[manifest.json](manifest.json) 校验四个运行产物，排除自身与后写的本 README。
当时工作区含未提交实现，复现应核对 `update.json` 中五个源文件 SHA，而不能只看 HEAD。
如果要先纸笔练习，可做[五参数 SGD 与冻结目标检查题](../../docs/control_ppo_checks.md#4-一次-ppo-的五个参数如何变化)。

本轮 45 项 PPO 更新测试在本机真实 Torch CPU 环境通过，包括全部 7 个参数的中心差分
梯度、目标梯度隔离及更新可重复性。正式产物的 4 个文件 SHA、5 个源文件 SHA、7 行参数
更新与 8 条样本的前后目标函数也已独立重算。全仓库测试总计 522 项通过。
