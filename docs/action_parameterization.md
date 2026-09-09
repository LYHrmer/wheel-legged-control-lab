# PPO 动作参数化：裁剪、纵向残差和配对实验

这组实验研究一个具体问题：已有 LQR/VMC 控制器能行驶时，PPO 的动作空间怎样影响训练？机器人仍由同一个控制器保持姿态，PPO 只增加机身纵向力和竖直力。其他训练配置保持相同，例如没有重调控制增益。

九个固定预算模型和 240 个留出回合已完成，[独立结果](../results/d1_action_ablation_analysis/README.md)没有支持直接改用均值约束或一维动作。下面说明算法和检查方法，解释这组结果能回答哪些问题。

## 三个候选究竟差在哪

| 名称 | 策略随机变量 | 交给环境的动作 | 物理残差力 |
| --- | --- | --- | --- |
| `full_gaussian` | `a ~ Normal(logits, std)`，二维 | `clip(a, -1, 1)` | `[11.25 ax, 20 az] N` |
| `bounded_mean` | `a ~ Normal(tanh(logits), std)`，二维 | `clip(a, -1, 1)` | `[11.25 ax, 20 az] N` |
| `fx_only` | `a ~ Normal(logits, std)`，一维 | `clip(a, -1, 1)` | `[11.25 ax, 0] N` |

`bounded_mean` 约束的是分布均值。标准差没有被 `tanh` 压缩，采样值仍能越过 ±1。`fx_only` 直接删掉了策略的竖直输出和对应的 `log_std` 参数；它没有先训练二维策略再把第二个动作遮住。后者在[残差反馈对照](../results/d1_residual_feedback/README.md)中研究，属于另外一种干预。

三个候选保留相同的 44 维观测布局。纵向版本中“上一时刻竖直残差”这一项为零，不能把它说成重新设计了观测空间。评估入口用已注册的动作 schema 判断如何把一维输出映射到 `[ax, 0]`，不凭数组长度猜策略含义。

实现可以从 [PPO 策略](../src/wheel_legged_control/d1/ppo_action_policies.py)和[纵向残差环境](../src/wheel_legged_control/d1/residual_action_env.py)开始读。

## 两个 clip 要分开看

环境执行动作前做 `clip(a, -1, 1)`，这是执行器输入边界。PPO 目标中的 `clip(r, 1-ε, 1+ε)` 作用于新旧策略概率比，是另一件事。

对角高斯的未裁剪采样记为 `a`。日志中的概率是：

```text
log πθ(a | o) = Σj [-0.5 ((aj - μj) / σj)² - log σj - 0.5 log(2π)]
rθ = exp(log πθ(a | o) - log πold(a | o))
Lclip = mean(min(rθ A, clip(rθ, 1-ε, 1+ε) A))
```

环境看到的是 `c = clip(a, -1, 1)`。策略梯度可以用原始随机变量 `a` 的概率来优化 `E[R(clip(a))]`。SB3 缓存原始采样及其 `log_prob`，不需要把这段执行裁剪误判为 PPO 的概率计算错误。

裁剪会改变执行动作的分布。一个通道在上边界的概率质量为：

```text
P(c = 1) = 1 - Φ((1 - μ) / σ)
P(c = -1) = Φ((-1 - μ) / σ)
```

不同的原始动作可能对应同一个边界输入。把 `c` 代回普通高斯密度，不能得到这个执行分布的正确概率。分析脚本因此用 NPZ 中的 `raw_action` 复算概率，并逐个样本检查 `executed_action == clip(raw_action)`。

`training_action_diagnostics.csv` 记录的采样裁剪率，与 SB3 的 `train/clip_fraction` 没有相同的分母或含义。前者数执行动作被裁剪的样本；后者统计优化 minibatch 中概率比偏离区间的比例。看训练曲线时先读列名。

## 只给均值加 tanh，为何没有 Jacobian

这里采样的随机变量仍然来自 `Normal(μ, σ)`，只是 `μ = tanh(logits)`。高斯密度公式没有改变。反向传播会经过：

```text
∂μ / ∂logits = 1 - tanh(logits)²
∂log π / ∂logits = ((a - μ) / σ²) · (1 - μ²)
```

均值靠近边界时，梯度也会变小。这一改动同时影响可表达的均值范围和优化路径，所以实验结果不能全部归因于“少了均值越界”。两个网络即使初始权重逐字节相同，`logits` 与 `tanh(logits)` 也不保证初始动作分布完全相同。

另一种常见方法先采样 `u ~ Normal(μ, σ)`，再执行 `a = tanh(u)`。那才是随机变量变换，动作密度需要减去 `Σ log(1 - tanh(u)²)`。本实验没有实现这种 squashed Gaussian，也没有把它的公式套到均值变换上。

[策略测试](../tests/test_d1_ppo_action_policies.py)检查原始高斯概率、均值变换梯度和真实 PPO 更新后的保存/加载。测试并不能证明训练出来的策略更好，它只约束实现是否对应上面的数学定义。

## 删掉 Fz 后，比较的是什么

纵向版本每次 PPO 更新都只优化一个随机动作。它的联合熵少了一个通道；采样维数变了，后续随机数消耗也会变化。竖直残差消失后，机器人状态轨迹会改变，纵向动作面对的观测也随之改变。

这能回答“只学习纵向补偿的整个训练方案是否更合适”。它不能隔离一次竖直推力对速度的瞬时影响，也不能保证三种候选经历逐步相同的探索。固定种子提供可复现的配对起点，不会抹掉这些差异。

初始化处理写在[训练入口](../scripts/run_d1_action_ablation.py)的 `initialize_from_common_policy` 中。相同 seed 的两个二维候选复制同一套权重；一维候选复制所有共享张量，只取动作头第一行及第一个 `log_std`。不同 seed 应得到不同的公共初始化。元数据记录公共参数 SHA 和投影后的 SHA，评估前会重新生成初始化来核验。

## 从一次 rollout 数到最终预算

预先规定的训练种子是 `9000 / 10000 / 11000`，每种参数化各训练三个模型。每个模型用四个环境；单环境收集 128 步后，PPO 获得 512 条 transition。

```text
一个 rollout：4 env × 128 step = 512 transitions
一次 PPO 更新：4 epochs × (512 / 128) minibatches = 16 次优化器更新
最终预算：131072 / 512 = 256 个 rollout
SB3 _n_updates：256 × 4 = 1024 个 epochs
优化器更新总数：256 × 16 = 4096
```

这里的 `_n_updates = 1024` 不能直接解释成 1024 次 minibatch 梯度更新。训练中不使用提前停止，学习率固定为 `3e-4`，每次 minibatch 为 128，`clip_range = 0.2`。控制周期为 10 ms，`γ = exp(-0.01 / 2)`，`GAE λ = 0.95`。奖励乘 `0.01` 后进入 PPO；评估 CSV 保存未缩放的任务奖励。

训练阶段在 `32768 / 65536 / 131072` 步保存检查点。前两个预算只评估开发集中的 6 个案例，最终预算评估全部 24 个开发案例。不能把 `3/6` 和 `13/24` 连起来声称成功率下降。分析脚本从最终 24 个案例中抽出同一组 6 个案例，另画固定案例曲线；完整的最终结果单独列。

开发评估还会加载检查点。加载操作会影响全局随机状态，所以训练入口在 `preserve_training_rng` 中保存并恢复状态，避免开发评估重新启动训练采样序列。

## 留出集和独立复算

新的 24 个留出案例在训练前写进协议，包括新的起伏波长与相位；最终比较固定使用 `131072` 步检查点。评估入口要求九个模型全部完成，先核验文件清单和每个模型的实际训练计数，再开始仿真。

基线控制器在每个案例只运行一次，共 24 个回合。九个 PPO 模型各运行相同的 24 个案例，共有 240 个留出回合。比较结果按训练 seed 列出，不能把 24 种地形案例当作 24 次独立训练，也不能把单次基线复制三份后计算“基线方差”。

[独立分析脚本](../scripts/analyze_d1_action_ablation.py)直接读取 CSV 和 NPZ，不调用训练入口的指标汇总函数。它会复算速度 RMSE 与高度 RMSE，重新检查质量门槛，再核对各层汇总表。墙钟耗时无法从状态日志重建，审计报告会列明这一限制。

原始训练目录保持只读。派生表和图片写到另一个目录，同时保存输入文件 SHA 清单及分析脚本快照。最终比较列出每个 seed 的差值，不做三个 seed 支撑不了的泛化显著性宣称。

## 本地检查

先运行算法和分析测试，不会打开正式留出地形：

```bash
env PYTHONPATH=src python3 -m pytest \
  tests/test_d1_ppo_action_policies.py \
  tests/test_d1_action_ablation_evaluation.py \
  tests/test_d1_action_ablation_analysis.py -q
```

本轮原始文件已齐，独立复算命令为：

```bash
env PYTHONPATH=src python3 scripts/analyze_d1_action_ablation.py \
  --training-roots results/d1_action_ablation_seed9000 \
    results/d1_action_ablation_seed10000 results/d1_action_ablation_seed11000 \
  --holdout results/d1_action_ablation_holdout \
  --output results/my_action_ablation_analysis
```

输出目录必须不存在。已有结果需要复算时，指定另一个明确命名的派生目录，核对差异后再处理旧派生版本。原始失败回合和冻结协议不作为“旧版本”删掉。

学习时可以先手算一个越界样本的 `log_prob`，对照 `raw_action` 与 `executed_action`。然后找同一个 seed 在三个预算上的固定六案例曲线，检查奖励变化是否伴随速度误差下降。若奖励提高而物理误差没有改善，应回到奖励分项或控制参考查原因，暂时不要把它写成 RL 性能收益。
