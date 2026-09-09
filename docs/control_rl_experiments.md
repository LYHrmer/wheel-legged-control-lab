# 把 D1 实验读成自己的控制与 RL 笔记

本页的 CSV 和源码归档随 Release 提供，见[下载与复算](reproducibility.md)。

这份入口适合已经能运行仿真、但还不能解释曲线的人。按下面顺序读，留一份自己的手算记录。面试时可以从其中一次失败讲起，说明配对实验只改了哪个量，以及数据是否支持继续改。

这里的硬件是仿真 D1。IMU 与编码器也是仿真测量，尚无真机验证；已有数据没有证明 RL 稳定优于传统控制，也没有证明 sim2real 已经可行。

所有命令从仓库根目录运行，需要本机的 `.local-deps`。生成结果时使用新目录，保留用于复核的正式记录；练习副本不混入正式成绩。下面的测试命令不重新训练 PPO。

## 先分清速度误差，再动控制器

[参考动态实验](reference_dynamics.md)保留了 120 个零残差回合。先读 `scripts/audit_d1_reference_dynamics.py` 的 `recompute_episode()`：输入是逐步 CSV 的速度误差，单位 m/s；输出包括末秒 bias、标准差和 RMSE，单位仍是 m/s，方差则是 m²/s²。

练习：用开发 case 04 的 `bias=-0.11693 m/s`、`std=0.01312 m/s` 手算 `sqrt(bias²+std²)`，应得到 `0.11766 m/s`，按这些舍入值计算允许 `1e-5 m/s` 的误差。然后找到逐回合复算表（`../results/d1_reference_dynamics/derived_metrics.csv`）中的 case 10，判断相同量级的 RMSE 是否来自同一种误差。

```bash
env PYTHONPATH=src python3 -m pytest tests/test_d1_reference_audit.py -q
```

接着跟到 [`moving_height_velocity()`](../scripts/diagnose_d1_reference_dynamics.py)。它接收地面 pitch（rad）和世界 x 向速度（m/s），返回高度参考变化率（m/s）；乘 `Kd=180 N·s/m` 才得到竖直前馈（N）。当前探针使用 COM 速度近似基座原点速度，文档给出了漏掉的旋转力臂项。

这次前馈使起伏案例末秒 RMSE 均值从 0.11535 降到 0.11190 m/s，质量通过数仍为 1/12。50/100 ms 姿态低通更差，通过数为 0/12。不要用“曲线平滑了”代替控制验收；这些开发案例也已经不能算新留出集。

## 拿掉一个残差通道，检查回报为什么增加

[残差反馈对照](../results/d1_residual_feedback/README.md)加载旧模型，不更新参数。代码入口是 [`select_action()`](../scripts/diagnose_d1_residual_feedback.py)：输入、输出都是无量纲的归一化动作，物理映射为 `Fx=11.25*ax N`、`Fz=20*az N`。

练习：给原始动作 `[0.4, 1.2]`，先写出完整模式和 Fx-only 的执行力。答案分别为 `[4.5, 20] N` 与 `[4.5, 0] N`。再看 `verify_nominal_replay()`：没有扰动时，回放名义动作应逐步复现状态；故意改掉一拍状态，检查必须失败。

```bash
env PYTHONPATH=src python3 -m pytest tests/test_d1_residual_feedback.py -q
```

正式记录的 18 组名义回放全部一致。加扰动后再比较在线策略与动作回放，LQR/VMC 在两边都保持反馈，只有 RL 层的反馈被拿掉。预定的末段速度收益门槛是下降至少 10%，并至少下降 0.005 m/s；三个种子都没达到，但质量通过数之差为 +2、0、+1，不能据此说策略完全不看状态。

移除 Fz 后，三个种子的高度 RMSE 都减少约 2.7 mm，六个名义场景的质量通过数仍为 3。回报增加主要来自努力惩罚和高度项。读数使用 metrics_corrected.csv（`../results/d1_residual_feedback/metrics_corrected.csv`），原表有两列姿态目标被误命名为奖励；更正没有改变原始奖励或物理指标。

## 从一次参数更新，接到动作分布实验

先做 [PPO 单步更新](ppo_update_lab.md)。[`one_sgd_update()`](../src/wheel_legged_control/ppo_update.py)的输入是冻结的八样本批次与学习率，输出参数梯度和更新前后损失；教学系统中的状态与动作均无量纲。这一节只有一次 SGD，不能用固定批次 loss 下降证明机器人控制变好。

```bash
env PYTHONPATH=src python3 examples/ppo_update_walkthrough.py
env PYTHONPATH=src python3 -m pytest tests/test_d1_ppo_action_policies.py -q
```

练习从 actor bias 开始：`-0.05 - 0.01*(-0.451891961) = -0.045481080`。跟到 `backward()`，检查冻结的优势不求梯度，第一次更新前的 ratio 为 1。

随后用 `mu=tanh(0.7)`、`sigma=0.8`、原始动作 `a=1.7` 手算一维 Normal log density，结果为 `-1.63361527`。与 [`BoundedMeanActorCriticPolicy._get_action_dist_from_latent()`](../src/wheel_legged_control/d1/ppo_action_policies.py)及测试对照，允许 `1e-6` 数值误差。这些量按归一化动作坐标计算，尚未换成 N。

这里仅对均值做 tanh，随机动作仍来自 Normal。log probability 不加 tanh Jacobian，梯度却要乘均值的 tanh 导数。若对随机动作本身做 tanh，才是另一种变换分布。环境裁剪 Gaussian 动作是合法设计；把裁剪后的动作塞回原始 Gaussian 密度，才会破坏本实验的采样概率对应关系。

新的 [动作训练入口](../scripts/run_d1_action_ablation.py)固定了三个独立分支：

| 分支 | 学习动作 | 原始样本分布 |
|---|---|---|
| `full_gaussian` | `[ax, az]` | `Normal(raw_mean, sigma)` |
| `bounded_mean` | `[ax, az]` | `Normal(tanh(raw_mean), sigma)` |
| `fx_only` | `[ax]`，Fz 恒为零 | 一维 `Normal(raw_mean, sigma)` |

三个分支各用三个新种子训练至 131072 步，保留 32768/65536 的 checkpoint。[正式对照](../results/d1_action_ablation_analysis/README.md)已经完成：基线通过 14/24，九个模型各通过 14–15/24。一维方案全程速度 RMSE 在三个种子上都高于同种子的双通道方案，不能把前一节遮掉旧策略 Fz 的收益直接沿用到重新训练。共同初始权重也不等于共同初始分布，推导见[动作参数化](action_parameterization.md)。

`ActionDiagnostics._on_step()`保存原始样本与执行样本，并要求同一未更新策略重算 log probability 的最大误差不超过 `1e-5`。检查训练裁剪率要数实际样本，不能用理论尾概率代替。Fx-only 的 [`step()`](../src/wheel_legged_control/d1/residual_action_env.py)严格映射 `[ax] → [ax, 0]`，对应的物理回归测试核对观测和奖励，不只是动作长度。

## 让传感器误差真的进入控制

读 [传感器估计](sensor_estimation.md)，再打开 [`D1ProprioceptiveEstimator.update()`](../src/wheel_legged_control/d1/sensor_estimation.py)。输入测量包含比力（m/s²）与角速度（rad/s），编码器给出 rad、rad/s；输出位置（m）及 RPY（rad），线速度（m/s）属于基座 COM。基座位置却属于可见原点，两者不能混用。

练习先只算 IMU 速度预测，不加轮里程计修正。在直立静止、初速度为零的条件下，令比力为 `[0,0,9.81] m/s²`，用 `a=R*f+[0,0,-9.81]` 推进 10 ms，速度应仍为零。自由落体时比力为零，速度增量应为 `[0,0,-0.0981] m/s`。再按 COM 力臂测试提供的向量计算叉乘，验证融合后的速度，测试绝对容差为 `1e-12 m/s`，相对容差为 `1e-7`。

```bash
env PYTHONPATH=src python3 -m pytest tests/test_d1_sensor_estimation.py -q
```

这里还假定有四路理想着地开关，初始化使用已知放置位姿；融合器没有外部定位和在线偏置估计。[144 回合实验](../results/d1_sensor_estimation/README.md)中，无延迟的 36 个估计反馈回合全部跑满，20 ms 延迟组有 17/36 跌倒。跑满也不等于速度跟踪合格。

检查 CSV 时，把估计速度减真值的误差，与真实速度减命令的误差分开。`offline` 只是旁听，`closed_loop` 才让估计值驱动控制。测量回放一致能证明计算可复现，不能证明延迟下稳定。

## 把同一个 checkpoint 跑到 45 秒以后

[连续任务](continuous_task.md)在固定道路上行驶，中途不 reset。[`YawRatePI.compute()`](../src/wheel_legged_control/d1/continuous_task.py)接收偏航角速度误差（rad/s）和周期（s），输出轮侧差动转矩（N·m）；`D1ContinuousTask.step()`接收 `[ax,az]`，返回传感器观测和奖励。RL 没有直接输出偏航转矩。

先手算 PI 的第一拍：积分初值为零，`kp=2`、`ki=3`，误差 `0.08 rad/s`，周期 `0.01 s`，输出应为 `0.1624 N·m`。然后完成本项练习：重载保存的 seed19000 最终策略，与同条件零残差各运行完整 45 s，检查 `replay_check.json` 的状态数组和 CSV 是否都与原记录一致。

```bash
env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/replay_d1_continuous_policy.py \
  --run results/d1_continuous_policy/sensor45_v1 --output results/my_sensor45_replay
```

观测必须匹配 `d1-continuous-sensor-command45-v1`，不能把旧 44 维真值策略补零后当成已适配。45/50 s 留出评价只更换噪声序列和命令时长，道路不变。

[三个正式种子的结果](../results/d1_continuous_policy/README.md)是完成 18/18、质量通过 0/18；零残差六个条件也没有质量通过。PPO 的全程高度 RMSE 比配对基线高约 3.34 mm，训练回报下降，竖直动作存在大量裁剪。完成视频已经保存，不用重新训练才能看。

验收看 `summarize_task()`：逐阶段剔除首秒后，高度 RMSE 必须 ≤25 mm，偏航角速度 RMSE 必须 ≤0.06 rad/s，还要满足速度等门槛并实际经过全部地形。seed17/45 s 的零残差左转高度 RMSE 为 26.27 mm，最终 seed19000 策略为 29.20 mm，两者都失败。全程 RMSE 与阶段 RMSE 不要混比。视频和 CSV 还保留了已记录的 2 ms 状态取样相位差，复核方式见结果说明。

完成这份练习后，给自己的项目介绍留一页证据即可：写出一个你改过的控制公式，附上配对曲线，并说明哪项验收仍未通过。当前可以准确讲清的工作是仿真控制实验及传感器接口，不应把这些经历写成已完成真机部署或在实习中做过的工作。
