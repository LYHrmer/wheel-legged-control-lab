# PPO 数值实验：先算明白目标函数

这份实验配合现有的控制与奖励章节使用。它只计算 GAE 和 PPO 的裁剪目标，不训练策略，
也不会改变已有 checkpoint。先把几组数算对，再去看 SB3 的 rollout buffer，会容易很多。

需要继续检查梯度和参数变化时，接着做[一次 PPO 更新](ppo_update_lab.md)：那一节使用
Torch 对小型 actor/critic 执行一次 SGD，保留每个参数的更新记录。本节仍保持纯 NumPy。

在仓库根目录运行：

```bash
PYTHONPATH=src python3 examples/ppo_walkthrough.py
```

不需要 Torch 或 GPU，也不启动机器人仿真；终端会打印短轨迹的计算表。需要保存表格时：

```bash
PYTHONPATH=src python3 examples/ppo_walkthrough.py --output /tmp/d1-ppo-lesson
```

这里的目录必须不存在。脚本写入 JSON 和三份 CSV，已有目录一律拒绝覆盖。默认运行不写
实验结果。实现见 [`ppo_learning.py`](../src/wheel_legged_control/ppo_learning.py)，
检查方法见 [`test_ppo_learning.py`](../tests/test_ppo_learning.py)。

## GAE：同一行数据必须属于同一次转移

一次转移按这个顺序记录：

```text
observation_t → action_t → env.step(action_t) → reward_t, final_observation_t
```

`values[t]` 是动作执行前的 `V(observation_t)`，`next_values[t]` 是该动作实际到达状态的
价值。遇到自动 reset，后者必须使用 reset 之前的 final observation，不能拿新一局的
初始状态代替。连续动作的 log probability 则对所有动作坐标求和，每个样本留下一个数。

接口接受 `(T,)` 或 `(T, N)` 数组，第一维是时间，第二维是互相独立的环境。所有输入
形状必须完全一致，不做广播；两个结束标记必须是布尔数组。NaN 会直接报错。

令 \(d_t\) 为 `terminated`，\(b_t\) 为 `terminated OR truncated`：

\[
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
\qquad
\hat A_t=\delta_t+\gamma\lambda(1-b_t)\hat A_{t+1}.
\]

返回的 `returns` 为 \(\hat A_t+V(s_t)\)，用于拟合价值函数；它是这里的 λ-return
目标，不能一概叫完整 Monte Carlo 回报。GAE 用 λ 调节多步估计的偏差与方差，价值估计
不准时，选择更小的 λ 会更依赖这个估计。[GAE 原论文](https://arxiv.org/abs/1506.02438)

两个掩码负责不同的事：

| 最后一步发生什么 | 用 final observation 的价值 bootstrap | 把后一局的 GAE 传回来 |
|---|---|---|
| 摔倒，`terminated=True` | 否 | 否 |
| 采集时间到，`truncated=True` | 是 | 否 |
| rollout 批次满，环境还没结束 | 是 | 否，批次外的优势未观测到 |

最后一种情况仍将两个标记留为 False；实现把批次外的优势 carry 初始化为零，末步的
`next_values` 负责 bootstrap。若两个标记同时为 True，终止优先。

以单步 `reward=1, V=0.5, V_next=10, gamma=0.9` 为例：摔倒时优势为 `0.5`；
外部超时则为 `9.5`。不能只用一个 `done` 决定是否 bootstrap。任务定义本身的有限时间
终点又是另一种情况：它属于 termination，观察需要包含剩余时间以保持 Markov 性。
[Gymnasium 时间限制说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

例子的三步 TD 误差为 `[1.04, 2.03, 3.02]`，采用 `gamma=0.9, lambda=0.8`，
倒着算得到 GAE `[4.067168, 4.2044, 3.02]`。先遮住输出自己算一次，然后把最后一步
改成摔倒，检查变化从哪里传回去。

## PPO 的 clip 具体裁掉哪一边

策略比率为同一状态、同一采样动作在新旧策略下的概率密度之比：

\[
\rho_t=\exp[\log\pi_{new}(a_t|s_t)-\log\pi_{old}(a_t|s_t)],
\qquad
L_t=\min\{\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\}.
\]

优化器最小化 `loss = -mean(L_t)`。旧 log probability 来自采样策略，在多轮 minibatch
更新中保持固定。优势在 actor 更新中也当作固定目标。连续分布的概率密度可以大于 1，
所以 log density 大于零并不违法。[PPO 原论文](https://arxiv.org/abs/1707.06347)

取 `clip_range=0.2`：

| 优势 A | 比率 ρ | 未裁剪 ρA | 裁剪候选 | 最终 min |
|---:|---:|---:|---:|---:|
| +2 | 1.5 | +3.0 | +2.4 | +2.4 |
| +2 | 0.5 | +1.0 | +1.6 | +1.0 |
| −2 | 1.5 | −3.0 | −2.4 | −3.0 |
| −2 | 0.5 | −1.0 | −1.6 | −1.6 |

负优势样本如果概率还在增大，例如 `A=-2, ρ=1.5`，未裁剪分支仍然生效。这时对新
log probability 的目标导数为 `ρA=-3`，并没有变成零。测试用有限差分核对了四个分支。
在裁剪拐点处不能用这个光滑导数解释。

clip 也不保证一次网络更新后所有样本的比率都在区间内。网络参数由样本共享，其他样本
的梯度还会影响它。训练时仍需观察 KL 和 clip fraction；这份数值实验没有网络参数，
因此不会输出训练 KL 或学习曲线。

本模块故意不归一化优势，也没有价值损失和熵项。比对 SB3 时，应使用同一批数据、
相同的优势归一化口径，否则两个 actor loss 不一定相等。

还要区分采样动作和实际施加的动作。现有 SB3 连续策略在送入环境前可能将动作裁剪到
动作空间，但 log probability 对应策略原本采样的动作。不要用裁剪后的动作重新计算
旧 log probability，再与原本采样动作的新 log probability 相除。

## 与仓库里的机器人训练对应

[`train.py`](../src/wheel_legged_control/train.py) 默认共用 `gamma=0.99`、
`gae_lambda=0.95`、`clip_range=0.2`，每个环境采集 `n_steps=256`。
现在可以通过同名 CLI 参数调整；省略参数时保留原来的训练配置。
平面模型每步 `20 ms`，D1 每步 `10 ms`。同样的离散参数对应不同的物理时间。

对于 \(\gamma^k=\exp(-k\Delta t/\tau)\)，有 \(\tau=-\Delta t/\ln\gamma\)：

| 控制步长 | 折扣降至 1/e 的时间 τ | 1 秒后的权重 | 256 步覆盖时长 |
|---|---:|---:|---:|
| 20 ms | 1.990 s | 0.605 | 5.12 s |
| 10 ms | 0.995 s | 0.366 | 2.56 s |

τ 只描述指数衰减，不能当作硬性的预测截止时间。将 20 ms 的 `gamma=0.99` 换成
10 ms、并保持这个时间常数，需要 `gamma_new = 0.99**(0.01/0.02) ≈ 0.99498744`。
这只是等物理折扣的换算，不能保证训练更好。GAE 的尾部权重由 `gamma*lambda` 决定，
奖励是否按每步或每秒计量也要一起检查。本次没有修改训练默认值。

历史 D1 平地训练入口使用 42 维观察，动作是两个残差合力，缩放为纵向 `±45 N` 和竖直 `±80 N`。
它经过 LQR/MPC+VMC 分配到电机；新的 IDQP 尚未接入这条 PPO 训练路径。观察与动作含义
见[学习指南](learning_guide.md)第 8 节。当前 82 维主线另见[命令条件实验](locomotion_lab.md)。

两个环境都把摔倒作为 `terminated`，采集步数用完作为 `truncated`。SB3 的 rollout
采集会处理超时 bootstrap；不能在送进 SB3 前再次手动加一遍。这份函数只是独立教学
实现，不被训练入口调用。

## 动手记录什么

建议每次只做一项改动，把计算过程记到自己的实验笔记。

1. 将 λ 设为 0，核对 GAE 是否退化成一步 TD。再设为 1，解释 rollout 尾端价值估计
   为什么仍会影响结果。给出计算过程，不只抄函数输出。
2. 在中间一步插入 timeout，后面拼接一局奖励为 1000 的轨迹。前一局优势不应被 1000
   污染。故意把两个掩码写成同一个，再用单元测试指出错在哪个数。
3. 固定负优势，扫描 `ρ=0.5…1.5`，画出 `unclipped` 与 `minimum`。在图上标明仍有
   梯度的区间，解释它与电机力矩 clip 的区别。
4. 看 [`rewards.py`](../src/wheel_legged_control/d1/rewards.py) 的动作变化惩罚。如果将
   控制频率翻倍，却不改权重，是否还在惩罚同一种物理行为？先写下单位和假设，再实验。

准备好这些后，再进行小预算 PPO 对照：保持训练步数预算口径一致，单独改变一个参数，
记录实际仿真秒数。评估仍使用[统一协议](evaluation_protocol.md)，分别报告跟踪误差，
不要只比较训练奖励。当前数值表只验证公式和程序行为，不能作为策略优于经典控制的证据。

## 跑一个可以检查内部指标的短训练

这一节需要已安装 RL 可选依赖。先用平面模型和单个 CPU 环境检查采集、更新及保存流程：

```bash
PYTHONPATH=src python3 -m wheel_legged_control.train \
  --robot planar --device cpu --envs 1 --steps 512 \
  --n-steps 128 --batch-size 64 --n-epochs 2 \
  --learning-rate 0.0003 --gamma 0.99 --gae-lambda 0.95 \
  --clip-range 0.2 --ent-coef 0.0 --log-training-metrics \
  --output /tmp/d1-ppo-pipeline-a
```

512 步只用来检查管线，不能评价 PPO 是否已经学会控制。若要观察另一组 λ，把上面的
`--gae-lambda` 改为 `0.8`，输出改成新的 `/tmp/d1-ppo-pipeline-b`，其余参数保留。
这只是单种子的诊断，不构成超参数优劣结论。

训练入口沿用已有的输出规则，可能覆盖同目录里的模型和配置。上面两个目录都应在首次
运行前不存在，重跑请换目录；它与前面数值实验“拒绝覆盖”的规则不同。

`--log-training-metrics` 会让 SB3 将框架记录的指标写到
`learning_metrics/progress.csv`。默认不改动 SB3 的 logger；加 `--verbose` 可同时
在终端查看日志。常用列的读法：

| 列名 | 用来检查什么 |
|---|---|
| `train/approx_kl` | 同一批采样动作上，新旧策略改变了多少 |
| `train/clip_fraction` | 比率落在 clip 区间外的样本比例；不等于梯度被置零的比例 |
| `train/entropy_loss` | 框架记录的负熵项，读数要连同符号看 |
| `train/value_loss` | 价值函数对当前训练目标的拟合误差，不是控制误差 |
| `train/explained_variance` | 当前价值预测对回报目标变化的解释程度 |
| `rollout/ep_rew_mean` | 已结束 episode 的平均回报，短到没有 episode 结束时可能缺失 |

字段定义可对照 [SB3 的日志说明](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)。
`clip_fraction` 这里按项目所用 SB3 源码中的 `abs(ratio - 1) > clip_range` 解释。

SB3 在一次 rollout 后、这一轮更新前输出日志，首行可能没有 `train/*` 指标。至少采集
两个 rollout 才能看到前一轮更新的统计，最后一轮更新也可能尚未被下一次 dump 写入。
这里直接保留框架的统计语义，没有自行补造缺失指标。

`training_config.json` 的 `ppo_hyperparameters` 保存实际传入 PPO 的超参数，网络结构
位于其中的 `policy_kwargs.net_arch`。当前仍为 `[128, 128]`。种子、实际训练步数和模型
SHA 继续按原字段保存，便于判断两次训练究竟改了哪些条件。

`batch_size` 必须大于 1 且不超过 `n_steps * envs`。不整除时 SB3 可以使用较小的最后
一个 minibatch，并给出提示；为方便对照，起步时可以选整除的组合。`gamma` 与 λ 允许
`[0, 1]`，`clip_range` 限制在 `(0, 1)`，学习率必须为正，熵系数不能为负，非有限参数
会在创建环境和输出目录之前报错。
