# 亲手核对一次 PPO 参数更新

上一节 [GAE 与 clipping 数值实验](ppo_learning_lab.md) 停在 loss。这一节使用 PyTorch 求梯度，更新一个 Gaussian actor 和一个独立 critic。网络总共只有 7 个参数，`parameters.csv` 可以逐项核算。

数据来自一个无量纲的一阶系统，不是 D1，也没有复用单轮闭环实验的成绩。这里检查的是 PPO 更新计算，不判断策略是否学会控制。

## 运行

在仓库根目录运行。需要项目的可选 `rl` 依赖；本机已有 `.local-deps`，CPU 即可。

```bash
PYTHONPATH=src python3 examples/ppo_update_walkthrough.py
PYTHONPATH=src python3 examples/ppo_update_walkthrough.py --output results/my_ppo_update
PYTHONPATH=src python3 -m pytest tests/test_ppo_update.py
```

第一条只打印。`--output` 只能写新目录；已有记录不会被覆盖。默认配置的完整记录见
[results/ppo_update](../results/ppo_update/README.md)。

| 文件 | 用途 |
| --- | --- |
| `update.json` | 完整 rollout、冻结批次、更新前后损失、参数梯度、软件版本及源码 SHA256 |
| `samples.csv` | 每条样本的动作、GAE 和 return，以及前后 log probability / ratio / value / clipping 分支 |
| `parameters.csv` | 每个标量参数的初值、梯度、更新值 |
| `clipping_fixture.csv` | 单独的正负优势裁剪算例，不混进 rollout |
| `manifest.json` | 上述四个产物的 SHA256；不包含 manifest 自身及后加的 README |

没有策略 checkpoint，也没有多轮训练。本节代码不改变 `train.py` 的配置。

## 先认清数据从哪里来

`synthetic_rollout()` 从 `v=-0.4` 开始，参考值固定为 `0.8`。观测是 `[v, 0.8]`，动作直接使用旧策略采样值，没有限幅或 tanh：

```text
v_next = 0.85 * v + 0.25 * action
reward = 1 - (v_next - 0.8)^2 - 0.05 * action^2
```

网络是刻意缩小的教学模型：

```text
actor:  mu = [0.1, 0.15] @ observation - 0.05
        sigma = exp(log_std), 初始 log_std = -0.4
critic: V = [0.2, 0.1] @ observation + 0.05
```

每次用局部随机数生成器采样 `action = mu + sigma * noise`，默认 seed 为 7。初始参数不随 seed 改变。采样期间不求梯度，八步结束后才开始优化。

第 8 步按时间上限截断，保留 `V(final_observation)`。GAE 使用旧网络的 value，`gamma=0.9`、`lambda=0.8`，并调用上一节的 NumPy 实现。整个更新期间优势和 return 都不再重算。

`FrozenBatch` 通过 `detach().clone()` 保存快照：detach 断开梯度路径，clone 隔离存储。单独调用 detach 会与原张量共享存储；这是测试既检查梯度又检查 `data_ptr()` 的原因。[PyTorch detach 文档](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.detach.html)

## loss 与参数变化的关系

同一观测、同一原始采样动作，计算旧策略与当前策略的联合对数密度。多维动作要先对动作坐标求和，得到每条样本一个 log probability。连续分布这里计算的是密度，数值可以大于 1。[PyTorch distributions 文档](https://docs.pytorch.org/docs/2.8/distributions.html)

优化器最小化：

```text
ratio = exp(new_log_prob - old_log_prob)
actor_loss = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
value_loss = mean((V - return_target)^2)
total_loss = actor_loss + vf_coef * value_loss - ent_coef * mean(entropy)
```

这对应 PPO 论文式 (7) 和式 (9) 的最小化写法，默认 `eps=0.2`、`vf_coef=0.5`、`ent_coef=0`。MSE 内部没有再乘 `1/2`。论文的完整算法会重复采样和执行多次小批次更新，本节只走一次 optimizer step。[PPO 原论文，第 3、5 节](https://arxiv.org/pdf/1707.06347)

本节使用普通 SGD，`momentum=0`、`weight_decay=0`。梯度不裁剪，优势不归一化。每个参数都满足：

```text
theta_after = theta_before - learning_rate * gradient
```

默认运行中 actor bias 的梯度为 `-0.451891961`，学习率 `0.01`：

```text
-0.05 - 0.01 * (-0.451891961) = -0.045481080
```

这个数还可以不用自动微分求出来。初始 ratio 为 1，且没有裁剪分支生效时：

```text
d(actor_loss)/d(bias) = -mean(A * (action - mu) / sigma^2)
```

critic bias 的总损失梯度为 `vf_coef * 2 * mean(V - return_target)`。默认 `vf_coef=0.5`，它等于 `-mean(A)=-0.835175622`；bias 从 `0.05` 变到 `0.058351756`。可以从 CSV 重算这两个值，再检查其余权重。

## 按执行顺序停下来检查

在编辑器里打开 [ppo_update.py](../src/wheel_legged_control/ppo_update.py)，按下面的位置下断点。不需要先看 SB3 内部。

| 停在哪里 | 应该看到什么 |
| --- | --- |
| `synthetic_rollout()` 返回前 | `actions.shape=(8,1)`，最后一项 `truncated=True`；旧策略能重算出相同 log probability |
| `one_sgd_update()` 里的 `before = ppo_terms(...)` 之后 | `ratio` 全为 1，`before['actor_loss']` 与 NumPy 对照一致 |
| `before['total_loss'].backward()` 之后 | `actor.bias.grad` 非零；batch 的 old_log_prob / advantages / returns 没有梯度 |
| `optimizer.step()` 前后 | 参数差值等于 `-learning_rate * gradient`，只执行一次 |
| `after = ppo_terms(...)` 之后 | 用同一批旧动作重新计算 ratio 和 value；没有重新采样，没有重算 GAE |

默认 seed 7、学习率 0.01 的本地结果：

| 量 | 更新前 | 更新后 |
| --- | ---: | ---: |
| actor loss | -0.835175622 | -0.839055165 |
| value MSE | 0.769464500 | 0.746670223 |
| total loss | -0.450443372 | -0.465720053 |

这些是固定批次上的损失。没有用更新后策略重新跑评估，不能把这张表写成“控制收益提升”。

默认批次里优势全为正，但第 0 条样本的 ratio 更新后约为 `0.99662`，没有增加。策略参数由所有样本共同决定，不能要求每个正优势样本的概率都单独上升。

## 第一次 ratio=1，裁剪在哪

初始当前策略就是采样策略。这一步的梯度没有遇到裁剪平台。`clipping_fixture()` 另设四个算例，把正负优势两种方向都摊开：

| ratio | A | min 后的目标值 | 目标对 new_log_prob 的导数 |
| ---: | ---: | ---: | ---: |
| 1.5 | 2 | 2.4 | 0 |
| 0.5 | 2 | 1.0 | 1 |
| 1.5 | -2 | -3.0 | -3 |
| 0.5 | -2 | -1.6 | 0 |

这里是目标的导数；actor loss 还要取负号，并除以批次大小。第三行超出区间仍有梯度：负优势动作的概率上升会降低目标值，不能被裁剪掩盖。PPO clipping 也不是把更新后的所有 ratio 强制限制在 `[0.8, 1.2]`。[PPO 原论文，式 (7) 与图 1](https://arxiv.org/pdf/1707.06347)

## 动手改哪些地方

想先减少手算量，可以做[五参数单样本题及参考答案](control_ppo_checks.md#4-一次-ppo-的五个参数如何变化)，再回到八样本批次。

- 先只改 `--learning-rate 0.1`。检查每个参数差值是否扩大十倍，再比较新的 ratio。若步长过大，固定批次 loss 也不保证下降。
- 对比 `--ent-coef 0` 和 `--ent-coef 0.1`。Gaussian 熵对 `log_std` 的导数为 1；总损失里熵是负号，所以后者的 log_std 梯度应比前者小 `0.1`。独立 critic 的梯度应该不变。
- 用 `--vf-coef 0` 运行。critic 应保持原值，actor 应得到同样更新。再解释为什么共享网络时这个结论不能直接照搬。
- 手算 `actor.bias` 梯度，然后阅读 `test_all_seven_parameter_gradients_match_finite_difference`。它对全部 7 个参数做中心差分，而不只判断 loss 是否下降。
- 在自己的练习文件中固定这批数据，再调用一次更新，保持 old_log_prob 不变。看 ratio 是否继续偏离 1。练习后恢复主示例的一步定义，避免把额外更新混入已发布记录。

## 和正式训练还差什么

现有 SB3 训练入口使用实际环境和不同网络结构，本节没有实现完整的 rollout buffer，也没有多轮采样或独立策略评估。它没有复现论文的控制成绩。

这里的原始 Gaussian 动作直接驱动合成系统。迁移到有动作限幅的环境时，需要区分“策略采样动作”和“实际执行动作”；不能把限幅后的动作直接塞回原 Gaussian 密度，当成采样时的概率。若使用 tanh 分布，还要按变换后的分布处理 log probability。

只有基础依赖的测试环境会明确跳过本节 Torch 测试；本机应安装可选 RL 依赖后实际运行。`FrozenBatch` 拷贝输入，但 PyTorch 张量本身仍可原地修改，调用方必须遵守更新期间不改 batch 的约定。

单轮控制实验回答辨识参数是否改善给定控制器；本节回答一次 PPO 梯度怎样得到并改变参数。等这两部分都能手算解释，再把策略接到单轮系统训练，比较反馈控制器与学习策略。
