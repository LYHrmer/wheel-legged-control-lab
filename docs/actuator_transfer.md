# 从合成力矩标定到 D1 补偿对照

这项练习把一个已知的合成力矩响应装进 D1，再检查拟合模型能否帮助控制。标定数据直接使用加噪后的电机输出力矩。真实 D1 是否提供可靠的关节侧力矩测量，本项目没有验证；这里不能声称完成了真机辨识。

[本次结果](../results/d1_v3_actuator_transfer/README.md)保留了一个值得检查的差别：补偿减小了力矩跟踪误差，两条开发道路上的速度误差却都略有增加。下面解释怎样得到这组对照。

## 先认清被拟合的参数

旧的[单转轴台架](../src/wheel_legged_control/actuator_bench.py)拟合运动方程。`armature` 是附加转动惯量，单位 kg·m²；`damping` 是黏性阻尼，单位 Nm·s/rad。`coulomb_friction` 则是 `tanh(速度 / 0.05)` 平滑摩擦的幅值，单位 Nm。[单轮速度控制](../src/wheel_legged_control/wheel_control.py)的前馈使用这套模型。

本实验拟合[执行器通道](../src/wheel_legged_control/d1/actuator_channel.py)：从请求力矩到输出力矩的响应。增益 `gain` 无量纲，时间常数 `time_constant_s` 用秒，`delay_steps` 数的是 2 ms 物理步。它们不能由旧台架参数直接改名得到。

整机模型原有的被动摩擦保持不变。本脚本不追加台架的 `tanh` 摩擦，也不把它填进 MuJoCo `frictionloss`；后者的约束摩擦机制与平滑摩擦律并不等价。

## 每一个样本对应什么时刻

控制器每 10 ms 更新一次请求。下面的通道和补偿器每 2 ms 执行一次，一个控制周期内执行五次。所有力矩均为关节侧 Nm，轮速用 rad/s。

设输入为 `u[k]`，响应状态为 `x[k]`，输出为 `y[k]`。按代码顺序：

\[
\begin{aligned}
\bar u_k &= \operatorname{clip}(u_k,-12,12),\\
x_k &= a x_{k-1}+(1-a)\bar u_{k-d},\qquad a=\exp(-0.002/\tau),\\
y_k &= \operatorname{clip}(g x_k,-12,12).
\end{aligned}
\]

`τ=0` 使用当拍直通响应。每次 reset 后，响应状态和延迟队列都为零。日志中的 `y[k]` 是执行这一物理步后的输出；`d=2` 表示前两个输出样本仍使用零历史，比无延迟通道晚 4 ms。它与观测侧的两个 10 ms 控制步延迟不同。

## 标定数据与独立验证

脚本从真实 `ActuatorChannel.step()` 采集数据。标定使用 4 s 分段随机常值输入，幅值不超过 3 Nm。另生成 3 s 扫频与正弦叠加输入，只做预测验证。两种波形各自使用局部随机数发生器，输出测量加 0.01 Nm 标准差的独立高斯噪声。

拟合函数只接收标定日志。它枚举 `d=0…4`，对每个候选延迟拟合四轮各自的增益与时间常数，最小化从零状态开始的整段输出误差。预测过程中不拿上一拍带噪测量替换模型状态。搜索范围预先固定为 `g∈[0.5,1.5]`、`τ∈[1,30] ms`，所有候选的收敛状态都保存在 `fit.json`。

这些激励处在未饱和区，适用于这里的一阶线性拟合。目标文件里的 `diagnostic_truth_nm` 只供核对造数过程，拟合函数不接收该字段。验证波形也不参与候选选择。真实电机若存在温漂、偏置或额外极点，当前合成数据不能反映这些误差。

## 因果补偿能消掉多少

用拟合值计算 `â=exp(-0.002/τ̂)`。目标力矩为 `r[k]`，补偿器提交：

\[
v_k=\frac{r_k-\hat a r_{k-1}}{\hat g(1-\hat a)}.
\]

它只保存上一拍目标力矩，不读取仿真状态或真实输出力矩。理想拟合且没有限幅时，可以得到 `y[k]=r[k-d]`；纯延迟仍然存在。程序没有把已知命令序列向前移动。

先手算一个小例子。取 `g=0.8`、`a=0.5`，目标从零变成 1 Nm。前两次补偿请求应为 2.5 Nm、1.25 Nm。若 `d=2`，输出应从 `0, 0, 1, 1…` 开始。[测试](../tests/test_d1_actuator_transfer.py)逐样本核对了这个关系。

目标突变时，逆响应可能要求很大的力矩。请求仍受原来的 12 Nm 限幅约束；一旦限幅，上面的精确等式就不成立。该补偿没有根据实际限幅输出重新估计响应状态，也没有重新整定原轮速 PI。应检查补偿后的峰值请求，不能只看稳态增益。

## 如何接到整机而不偷换对照

[实验脚本](../scripts/evaluate_d1_actuator_transfer.py)中的 `WheelOnlyChannel` 放在现有逐物理步调用的位置。环境 reset 会重建原通道，所以替换发生在 reset 之后、第一步仿真之前。

四个轮关节使用合成响应，其余 12 个腿关节保持当拍直通和原力矩限值。两个非理想对照共享完全相同的隐藏物理参数；只有补偿组使用标定得到的模型计算额外请求。拟合参数没有反过来替换被测对象。

16 轴 `config` 只承载模型要求的时钟和力矩上限。实际每轴响应参数写在 `parameters.json` 的 `installed_channel` 中，延迟也按 16 轴分别记录。它比 reset 时记录的名义通道配置更具体。不要从名义配置的 `delay_steps=0` 推断四轮没有延迟。

控制器使用 oracle 状态源，RL 残差固定为零。三个对照固定同一条开发道路和命令，reset 使用相同 seed，低层控制参数不随对照改变。闭环轨迹仍会不同：腿部控制器可能对车身运动作出不同响应，不能要求三组腿力矩逐样本相同。

## 运行与复算

在仓库根目录执行，使用已经安装项目基础依赖的 Python 环境；本实验不需要训练 PPO：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/evaluate_d1_actuator_transfer.py \
  --seed 17 --duration 60 --output results/my_actuator_transfer

pytest tests/test_d1_actuator_transfer.py
```

脚本按串行顺序跑两条开发道路，每条有理想、非理想、拟合补偿三个对照。输出目录必须是新的。`--duration 0.03` 可检查接口，但会被标为短时测试，不能拿来替代 60 s 任务。已经发布的实验使用其目录内 `source.tar.gz` 所记录的源码；当前源码的重跑不自动等于历史结果的逐字节复现。

每个对照保存 `metrics.csv`，以及包含 2 ms 力矩记录的 `physical_trace.npz`。其中 `controller_requested_nm` 是补偿前请求，`drive_requested_nm` 是补偿后请求。`drive_limited_nm` 经过输入限幅，`applied_nm` 是真正施加到关节的电机力矩。

可以按下面的顺序检查：

1. 从原始标定输入和拟合参数重算响应，核对 `calibration_prediction.npz`；随后计算独立验证误差。不要用验证误差重新挑一个延迟。
2. 对轮轴核对 `delayed_nm[2:] == drive_limited_nm[:-2]`，前两拍应为零；对腿轴核对当拍直通。
3. 用 `applied_nm - controller_requested_nm` 重算四轮力矩 RMSE。另数 `drive_requested_nm != drive_limited_nm`，注明分母是四个轮轴还是全部 16 轴。
4. 从逐物理步记录计算 `sum(abs(applied_nm * joint_velocity_rad_s))`，每五步取均值，与 CSV 的机械活动量核对。它没有包含电机效率，不能解释成电池功率。

跟踪误差和终止原因分开看。若一组提前摔倒，先比较完成时长，再检查相同时间前缀；很短的存活轨迹可能给出很小的 RMSE。本次详细数值与审计范围留在[结果页](../results/d1_v3_actuator_transfer/README.md)。
