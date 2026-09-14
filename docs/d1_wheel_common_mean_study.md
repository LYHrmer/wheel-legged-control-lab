# 为什么新训练的 PPO 反而走慢了

这是在既有 v3 地形课程上进行的一次独立实验。Astra ultra 负责假设和对照设计，实际 Claude Opus 编写策略核心，Codex 负责接入、测试、训练和指标复核。[模型分工与原始代码](../results/d1_wheel_common_mean_development/contribution.json)有可追踪记录。

这里的任务仍是 82 维观察、8 维腿轮残差、100 Hz MuJoCo 控制。课程 GUI 使用另一套传统控制器；本实验没有训练 GUI 跳跃、横移或大台阶通过。

## 从观察到干预

上一轮 16,384 步课程 PPO 在同一条 32 秒开发道路上，只前进 4.83 米；基础控制器的零残差策略前进 7.52 米。检查保存的动作发现，PPO 对四个轮子输出了接近常量的共同负修正。

轮速残差每个单位对应 4 rad/s。旧课程策略稳定行走期间的共同修正约为 −0.875 rad/s。按 0.087 米轮半径换算，理想纯滚动速度差约为 −0.076 m/s，与实际减速方向和量级相符。接触、滑移和姿态也会影响速度，因此还需要物理干预确认。

在相同初态和同一旧权重下，每个控制周期根据当前观察重新推理，分别保留腿动作、轮动作，或者去掉四轮动作的共同均值。没有重播原来的动作序列。

| 32 秒道路干预 | 前进 m | 速度 RMSE m/s | 偏航 RMSE rad/s | 高度 RMSE m | 总回报 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 零残差基础控制器 | 7.520 | 0.03972 | 0.01414 | 0.01105 | 61.053 |
| 完整旧 PPO | 4.834 | 0.09054 | 0.02059 | 0.00837 | 56.814 |
| 仅腿动作 | 7.501 | 0.03876 | 0.01394 | 0.01197 | 60.785 |
| 仅轮动作 | 5.183 | 0.08213 | 0.01823 | 0.00694 | 58.147 |
| 去掉四轮共同均值 | 7.209 | 0.03233 | 0.02358 | 0.00798 | 61.808 |

[道路干预结果](../results/d1_wheel_common_mean_development/road_action_ablation_01/summary.json)和[平地干预结果](../results/d1_wheel_common_mean_development/flat_action_ablation_01/summary.json)支持共同负轮速残差是减速的直接原因。恢复速度时偏航仍可能变差，需要分别报告。

旧 PPO 的任务回报低于零残差；去掉共同减速分量后，原任务回报又上升。这些结果不支持直接把问题解释成“奖励就喜欢慢走”。[独立接口审计](../results/d1_wheel_common_mean_development/interface_audit/README.md)也未发现观察错拍、重复叠加动作、保存加载错误或 timeout bootstrap 错误。

## 唯一算法改动

[Opus 编写的策略模块](../scripts/d1_wheel_common_mean_policy.py)在训练和推理时采用同一个高斯均值变换。设网络输出的四轮均值为 `w`，共同分量为 `m = mean(w)`：

```text
共同分量 = 0.05 * tanh(m / 0.05)
新的四轮均值 = w - m + 共同分量
```

腿部均值保持不变，任意两个轮子之间的均值差也保持不变（允许浮点舍入误差）。`None` 分支完全使用原 SB3 策略逻辑。奖励、观察、动作单位、PPO 参数和基础控制器均保持原实验设置。

这个限制只作用于高斯分布的均值。随机采样仍使用原来的 8 维独立正态分布，PPO 使用该分布计算实际样本的概率。没有对样本额外投影，也没有加入不适用的 tanh Jacobian 修正。

默认上限 0.05 对应共同轮速均值 ±0.2 rad/s，约为理想纯滚动速度 ±0.0174 m/s；这不是实际车速保证。SB3 原有的逐项 `[-1,1]` 裁剪可能破坏执行后的共同均值约束，确定性动作也有这个限制。例如 `[3,-1,-1,-1]` 的均值为零，裁剪后均值为 −0.5。因此评测同时记录网络原始均值、变换后高斯均值、裁剪后的动作和最终执行动作。

## 预先固定的比较

训练种子为 48001、48002、48003，每个种子训练原版对照和共同均值受限策略，各 16,384 个实际环境转移，总预算 98,304。每对初始化权重相同，沿用地形课程、32 秒回合、0.25 m/s 前进命令和原 PPO 超参数。每个模型使用固定预算终点的权重。

[评测协议](../results/d1_wheel_common_mean_development/evaluation_protocol.json)在训练前固定：已用于诊断的平地和道路归入开发用例；最后另测两组新地形参数与速度命令。后者复用了已有地形类型，不代表任意地形泛化，也没有借用旧正式实验的 holdout 身份。

主要比较每个训练种子在两项最终任务上的平均速度 RMSE 配对差。三个种子的方向都改善且没有增加失败，才称为本轮一致改善速度跟踪。回报、偏航、高度、接触、限幅和基础控制器对照仍必须同时报告。

## 复现入口

在仓库根目录使用已安装 RL 依赖的 Python 环境。输出目录必须是新目录：

```bash
python scripts/run_d1_wheel_common_mean_study.py --mode train \
  --output results/my_common_mean_training \
  --evaluation-protocol results/d1_wheel_common_mean_development/evaluation_protocol.json

python scripts/run_d1_wheel_common_mean_study.py --mode evaluate \
  --models results/my_common_mean_training --output results/my_common_mean_dev \
  --evaluation-protocol results/d1_wheel_common_mean_development/evaluation_protocol.json --cases dev

python scripts/run_d1_wheel_common_mean_study.py --mode evaluate \
  --models results/my_common_mean_training --output results/my_common_mean_final \
  --evaluation-protocol results/d1_wheel_common_mean_development/evaluation_protocol.json --cases final
```

`--mode smoke` 默认使用独立种子 48999，各运行 64 步，只验证训练管线。评测入口会拒绝这些短程权重，以及源码哈希、模型训练步数或实验身份不匹配的权重。正式训练默认为全部三个种子，也可用 `--seeds 48001` 先跑第一组完整对照；这不改变该模型的 16,384 步预算。

## 实际结果与采纳决定

六个模型均完成 16,384 步训练，每个模型有 128 次 rollout 更新、512 个优化 epoch。三对初始参数哈希一致。由于部分回合提前终止，各策略的课程等级曝光和后续重置时刻不同，完整记录见[训练结果](../results/d1_wheel_common_mean_development/README.md)。

最终评测实际执行 14 个回合、37,992 个环境转移。以下误差与回报按两个场景等权平均；每个场景仅包含该策略实际运行期间的数据，提前终止的持续时间和[共同前缀复算](../results/d1_wheel_common_mean_development/comparison_final_01/comparison.md)必须一起看。

| 训练种子 | 原版 → 受限：速度 RMSE m/s | 原版 → 受限：跑满回合数 | 原版 → 受限：平均回报 |
| --- | ---: | ---: | ---: |
| 48001 | 0.04352 → 0.03974 | 0/2 → 0/2 | 48.326 → 45.187 |
| 48002 | 0.19623 → 0.12330 | 1/2 → 2/2 | 10.557 → 43.389 |
| 48003 | 0.04716 → 0.05864 | 0/2 → 0/2 | 50.764 → 43.408 |

零残差基础控制器跑满 **2/2** 场景，平均速度 RMSE 为 **0.04251 m/s**，平均回报 **60.641**。原版 PPO 共跑满 **1/6** 个模型与场景组合，受限 PPO 为 **2/6**。零残差每个场景只运行一次，不能把表中的重复参照当成更多独立样本。

48001 的速度误差下降，但平均偏航误差从 0.03639 增至 0.04480 rad/s，两个场景均比原版更早越界；48003 的速度、高度、偏航均值和回报都退化。48002 改善明显，仍未接近基础控制器的速度和回报水平。所有最终确定性动作的 Box 裁剪比例为零；受限策略实际记录的最大共同高斯均值约为 0.04980，符合 0.05 约束。本轮的偏航和失稳发生在未触发 Box 裁剪时。

`fall_or_body_contact` 是环境合并的终止标签。本轮对应记录的非期望接触计数为零，离地间隙也未触发下限，实际触发的是倾角阈值，不能把标签直接解释成已经发生机身碰撞。

**候选未满足三个种子一致改善的预声明条件，保留为实验模块，不作为驾驶默认策略。** 这次实验定位了共同轮速偏置的作用，也表明单独限制它不足以保证转向与姿态稳定。限制对同一个网络输出保留差速分量，并不意味着两组重新训练后的差速和腿部输出相同。

全部[六个最终权重及 sidecar](../results/d1_wheel_common_mean_development/models/origin_manifest.json)随仓库提供；可将复现命令的 `--models` 直接替换为 `results/d1_wheel_common_mean_development/models`，无需重新训练。逐拍原始记录保留在本地工作目录，公开文件清单明确区分了已上传权重和本地原始轨迹。
