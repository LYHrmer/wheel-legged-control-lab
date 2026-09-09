# 从上坡倒滑到地形跟踪

这页记录第二版多地形实验的修改依据。第一版的训练确实跑完了，但在 +4° 坡、
0.35 m/s 命令下，零残差控制器 4 秒后还倒退了 0.312 m。仅凭回合存活无法说明任务完成。
旧数据保留在 [v1 结果](../results/d1_terrain_curriculum/README.md)，环境定义见
[第一版地形课程](terrain_curriculum.md)。

[第二版正式结果](../results/d1_terrain_tracking_v2/README.md)已完成 9 个模型训练及 240 回合评测。
所有回合最终前进且存活，但零残差的达标数为 15/24，各 PPO 为 13–14/24。
下面的控制修正与数值诊断不等于已经取得 RL 控制收益。

## 先修控制参考

下面每组都使用同一个 D1 模型、seed 31 和零残差，只改表中所列的变量。
命令先站立 0.3 秒，再在 0.3 秒内平滑升至 0.35 m/s；测试共 4 秒。

| +4° 上坡的改动 | 位移 m | 速度 RMSE m/s |
|---|---:|---:|
| 原控制器，姿态目标为世界水平 | -0.3121 | 0.4070 |
| 只增加坡度重力前馈 | -0.3344 | 0.4116 |
| 只取消位置参考的 ±0.55 m 限幅 | -0.1523 | 0.3743 |
| 只把出生姿态设成沿坡面 | -0.4918 | 0.4519 |
| 上下层姿态目标同时沿地面切线 | 0.9564 | 0.1143 |

这些测试都没有触发关节力矩限幅。重力前馈在这组控制器上未解决倒滑，v2 没有加入它。
完整的平地、上下坡对照在
[probes.json](../results/d1_terrain_tracking_diagnosis/probes.json)。
表中 aligned 是早期只修改世界 Pitch 的诊断；v2 实现完整的当前偏航法向变换后，
同一个 +4° 回归工况的位移为 0.9605 m、速度 RMSE 为 0.1146 m/s，数值略有不同。

外层 LQR 使用距离、Pitch、前向速度和 Pitch 角速度，控制量可写成：

```text
Fx = -K @ (state - reference)
pitch_reference = flat_operating_pitch + command.pitch
```

当前 Pitch 增益约为 -93618 N/rad。仅 1 mrad 姿态误差就对应约 93.6 N 的未限幅反馈项，
不能把它当成与速度跟踪无关的小量。第一版在坡上仍请求世界水平；第二版将同一个地面姿态
目标传给 LQR 和下层 VMC。原 LQR 增益及平地线性化工作点都没有重调。

起伏地形另做了 12 个开发工况、三种参考、两次原样重复，共 72 回合。
局部切线的平均速度 RMSE 为 0.11156 m/s，四轮投影点拟合平面为 0.12043 m/s，
世界水平为 0.21609 m/s。局部切线在 12 个工况中的 10 个误差更低，所以本轮先用它。
拟合平面的姿态请求变化率更低，但各组均未跌倒，不能据此声称它更安全。

也有反例：10 mm 起伏、波长 0.8 m、相位 0 时，局部切线的高度 RMSE 为 15.696 mm，
拟合平面为 12.013 mm。数据和同轨迹参考变化率对照保存在
[地形参考诊断](../results/d1_terrain_reference_diagnosis/summary.json)。

## 坡面方向怎样变成姿态命令

设地面为 z=h(x,y)，斜率为 a=∂h/∂x、b=∂h/∂y，当前偏航为 ψ。按
`R = Rz(ψ) Ry(pitch) Rx(roll)` 定义：

```text
forward_slope = a*cos(ψ) + b*sin(ψ)
lateral_slope = -a*sin(ψ) + b*cos(ψ)
pitch_target = -atan(forward_slope)
roll_target = atan2(lateral_slope, sqrt(1 + forward_slope**2))
```

可以用这两个角重建机身 z 轴，检查它是否与 `[-a,-b,1]` 的单位法向相同。
直接把世界 Pitch 用在任意偏航的机器人上会出错。当前高度场只沿 x 变化，b=0；
这里的坐标变换测试不代表已经训练了横坡或转向。

v2 仍为 44 维观测。前 42 个字段和第一版布局一致，最后两个值改为当前偏航下的地面
Pitch/Roll 目标。奖励的姿态项也改成相对这两个目标的误差，其他权重保持原值。
控制器用步前状态生成命令；奖励和下一观测用步后位置重新读取地面，日志分别记录这两种时刻。

| 契约 | v1 | v2 |
|---|---|---|
| observation_schema | d1-terrain-oracle-v1 | d1-terrain-tracking-oracle-v2 |
| reward_schema | d1-terrain-clearance-v1 | d1-terrain-tracking-v2 |
| control_schema | 原有世界水平参考 | d1-lqr-vmc-local-tangent-v2 |
| 两维动作范围 | ±11.25 / 20 N | 不变 |

相同的观测维数不意味着模型可互换。新版 checkpoint 必须通过版本检查；旧 42 维策略也不会装入这里。

## PPO 为什么要先检查奖励单位

旧模型一次开发 rollout 的 512 条转移样本中（4 个环境各 128 步），价值预测均值约 27.55，
GAE return 均值约 81.66，
explained variance 为 -0.00555。Critic 最后一个 Tanh 层的激活绝对值均超过 0.95。
在实际 128 样本 minibatch 上，Actor 梯度范数为 0.318，带权 Critic 梯度范数为 442.62。
全局梯度裁剪把总范数压到 0.5，裁剪系数约为 0.00113。

“梯度被缩小”还不足以说明 Adam 的实际步长变小，因为 Adam 会用历史二阶矩归一化。
诊断保留了旧优化器状态，实际执行一次更新：全局裁剪时 Actor 参数步长范数为
0.001124，分别裁剪时为 0.028139。这是一个冻结 minibatch 的数值对照，不是控制收益。

随后固定同一批目标，重新初始化 Critic，保留默认输出头，只将目标乘 0.01。
200 次拟合后，换回原单位的 MSE 为 89.04、EV 为 0.506；原尺度条件的 MSE 为 5665.91、
EV 为 -0.028。独立 NumPy GAE 复算与 SB3 return 的最大误差约 1.53e-5。
[原始 rollout 与拟合曲线](../results/d1_terrain_ppo_diagnosis/summary.json)保留了这些检查。

这支持从头训练时试验 `reward_scale=0.01`。它没有证明策略一定变好，也不支持把已经饱和的
旧 Critic 简单换单位后继续用。新版把缩放放在训练包装器中：PPO 接收缩放后的标量，
Monitor、奖励分项和评测回报仍为原始单位。没有同时改网络、学习率或梯度裁剪方法。

超时截断的单位也要一致。下面做一次手算：设原始奖励为 7.5，训练尺度为 0.01，γ=0.9，当前训练单位下的
末状态价值为 0.4。SB3 在写入 rollout buffer 前先把超时 bootstrap 加到奖励中：

```text
TimeLimit: buffer_reward = 0.01*7.5 + 0.9*0.4 = 0.435
terminated: buffer_reward = 0.01*7.5 = 0.075
```

截断回合的最后一步 return 已等于这个修正后的 buffer reward，不能再加一次 γV。
Monitor 记录的原始奖励仍为 7.5。真实执行检查在
[`test_d1_terrain_tracking_experiment.py`](../tests/test_d1_terrain_tracking_experiment.py)和
[`test_d1_terrain_reward_units.py`](../tests/test_d1_terrain_reward_units.py)。

## 训练前固定的对照

先做奖励尺度 A/B：mixed 模式、每组 32768 步、4 个环境，训练 seed 为 7000 和 8000，
分别使用尺度 1 和 0.01。只评测开发工况。正式多地形对照每组每个 seed 65536 步，
seed 为 0、1000、2000；相邻向量环境的 seed 偏移不跨这些训练种子。

[已经完成的 A/B](../results/d1_terrain_reward_ab_scaled/README.md)中，缩放没有提高任务达标数，
两个种子的全程速度误差还略增。正式三组仍统一采用 0.01 作为训练数值单位；
这不是已经验证最优的控制配置。更长预算和训练分布的效果要另看正式对照。

| 组别 | 地形采样 | 用途 |
|---|---|---|
| zero_residual | 不训练 | 测量共同经典控制器 |
| flat | 始终平地 | 检查只训平地的保留与迁移 |
| mixed | 始终采用第 3 阶段分布 | 检查直接混合地形 |
| curriculum | 按固定预算分四阶段 | 检查课程安排是否有帮助 |

四组使用相同的 v2 控制器。每个训练运行只使用最终预算 checkpoint，不按评测成绩挑模型。
各阶段的分布沿用第一版，训练坡度仍为 ±2°/±4°，起伏振幅仍为 5–10 mm。
mixed 与 curriculum 的累计地形曝光也不同，不能把差异全部归因于课程顺序。
本轮比较完整的采样策略；要单独检验顺序，还需把同一批地形回合重排，并保持总曝光一致。

每个评测 split 有 24 个固定工况：4 个平地、12 个起伏、8 个坡道。起伏使用两个振幅与
三种波长、两种相位的全组合；坡道用四种坡度各配 0.25/0.40 m/s。
开发集波长为 0.8/1.0/1.2 m、相位为 0 和 π/2、坡度为 -4/-2/2/4°。
新留出集波长为 0.75/0.95/1.15 m、相位为 0.53/2.17、坡度为 -3.5/-2.5/2.5/3.5°。
这些范围不支持大范围分布外泛化结论；训练相位本来就是连续随机采样。
旧 v1 留出成绩已经看过，本轮只把它当开发资料。

4 秒回合的 `quality_success` 要求同时满足：

- 完整运行、未触发终止，实际位移与命令积分之比在 0.65–1.35；
- 最后 1 秒速度 RMSE 不超过 `max(0.06 m/s, 目标速度的25%)`；
- 全程 clearance RMSE 不超过 30 mm，最大绝对偏航不超过 0.15 rad，平均关节限矩比例不超过 5%。

全程速度 RMSE 仍然保留，不能把起步瞬态从日志中删掉。末段指标用于区分持续跟踪与起步过程。
这些是本学习项目预先固定的任务门槛，没有真机安全认证含义。未通过的 case 要照常列出。
训练日志中的 `rollout/success_rate` 仍沿用存活到回合上限的定义，不能拿它替代评测的 `quality_success`。

## 自己复算

先在开发集运行，不使用新留出结果调参。每次另给一个不存在的输出目录。

```bash
# 奖励尺度开发对照的一侧；另一侧只把reward-scale改成1
python scripts/run_d1_terrain_curriculum.py --profile tracking-v2 \
  --conditions mixed --reward-scale 0.01 --steps 32768 --envs 4 \
  --seeds 7000 8000 --evaluation-split development --episode-seconds 4 \
  --output results/my_d1_reward_scaled

# 固定预算的三种训练分布，对同一批新留出工况评测
python scripts/run_d1_terrain_curriculum.py --profile tracking-v2 \
  --conditions flat mixed curriculum --reward-scale 0.01 --steps 65536 --envs 4 \
  --seeds 0 1000 2000 --evaluation-split holdout --episode-seconds 4 \
  --output results/my_d1_tracking_v2
```

源码未提交时也可以使用[运行前导出的快照](../results/d1_terrain_tracking_v2_source/README.md)。
这些实验不会自动装入键盘演示。

```bash
# 原倒滑症状会以非零退出码结束；这是保留的诊断红灯
python scripts/diagnose_d1_terrain_tracking.py \
  --variants original --slopes 4 --seconds 4 --require-forward

# 只改变姿态命令，检查开发坡道的反事实
python scripts/diagnose_d1_terrain_tracking.py \
  --variants aligned --slopes 0 4 -4 --seconds 4 --require-forward
```

先核对 +4° 的未限幅 LQR 各项，再改变一个因素。若想研究平面拟合，使用
`diagnose_d1_terrain_reference.py` 保持控制增益和残差不变；不要同时加滤波又改奖励，
否则难以判断改进来自哪里。

这里的机器人状态和地面法向都来自仿真 oracle。当前任务限定为纵向小起伏与缓坡上的直行，
没有验证台阶攀爬、跳跃或真机迁移。
