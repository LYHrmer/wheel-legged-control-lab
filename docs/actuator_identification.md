# 单转轴台架：从力矩响应到参数辨识

这组实验只需要 CPU。先研究一个轴，能把延迟和摩擦的影响看清楚，再回到整机控制。
`wheel` 是悬空转子，`pendulum` 是带重力负载的单摆。后者用于学习腿关节负载的基本问题，
没有实现完整 D1 单腿。全部数据由 MuJoCo 生成，质量和电机参数是教学假设。

已有整机 IDQP 和 PPO 不会被这些脚本修改。这里报告的是离线预测能力，尚未证明辨识会
改善整机控制，也没有辨识真实 D1 的参数。

单轮范围的后续控制对照已单独实现，见[PI 与辨识前馈实验](wheel_control_lab.md)。它使用
新的闭环参考，在同一对象上比较三组控制器；本节的离线预测数据和结论保持不变。

本节拟合使用位置和独立合成速度测量。若想检查去掉速度通道会怎样，接着做
[仅位置编码器辨识](encoder_identification.md)：它使用单独的日志接口，和本节方法做配对对照。

## 先运行，再看实现

在仓库根目录、已安装项目依赖的 Python 环境中运行：

```bash
python scripts/run_actuator_identification.py \
  --bench wheel --scenario matched --output results/my_wheel_identification
```

默认每段运动 3 秒，物理步长 2 ms。脚本采集两段标定运动，随后报告不同扫频的验证结果，
以及阶跃与换向的留出测试。目录必须是新的，已有实验不会被覆盖。

运行包括结构失配的四组对照：

```bash
python scripts/run_actuator_identification.py \
  --bench both --scenario both --output results/my_actuator_comparison
```

结果索引与实际数字见[本仓库的合成实验记录](../results/actuator_identification/README.md)。
学习时另建输出目录，不覆盖这里的基线。

| 输出 | 怎么用 |
| --- | --- |
| `experiment.json` | 采样、噪声、拟合预算和源文件 SHA；合成真值单独标为生成端信息 |
| 每组的 `calibration_0/1.csv` | 唯一允许用于选参的两段数据 |
| `validation_2.csv`、`holdout_3.csv` | 只用于报告预测，没有进入优化器 |
| `fit.json` | 所有延迟候选及求解状态、选中参数、局部雅可比奇异值 |
| `*_predictions.csv`、`prediction.png` | 原始测量与名义/辨识模型的完整连续预测 |
| `metrics.csv` | 位置/速度 RMSE，另列低速样本的速度误差与样本数 |
| `summary.json`、`manifest.json` | 是否跑完、耗时、结果文件 SHA-256 |

程序先落 CSV，再由拟合器重新读文件。拟合接口收不到生成端的 `ActuatorBench` 对象，
也不接收真值参数。它仍是一个进程内可审计的数据分工，并非操作系统级的访问隔离。

## 1. 先推导单轴动力学

本台架的连续时间模型可以写成：

\[
(J_{load}+I_a)\ddot q+b\dot q+c\tanh(\dot q/v_\epsilon)+g(q)=\tau_{applied}.
\]

`I_a` 是附加在关节侧的转动惯量；已知负载惯量 `J_load` 不参与拟合。单轮的
`J_load=0.5mr²`，取 `m=0.5 kg`、`r=0.087 m`，得到 `0.00189225 kg·m²`。
半径参考公开 D1 模型，质量没有经过实测。

单摆长 `0.25 m`、质量 `1 kg`，质心在中点。它的负载惯量包含平行轴项，
`g(q)=mgL/2·sin(q)`；`q=0` 表示自然下垂。这里保留了重力随关节角度变化的非线性。

| 量 | 单位 | 在哪里进入仿真 |
| --- | --- | --- |
| `armature` | kg·m² | MuJoCo joint 的附加惯量 |
| `damping` | N·m·s/rad | MuJoCo joint 的黏性阻尼 |
| `coulomb_friction` | N·m | 步初计算的平滑摩擦力矩，写入 `qfrc_applied` |
| `delay_steps` | 物理步 | 限幅后电机命令的 FIFO 队列 |

`v_epsilon=0.05 rad/s`。`tanh` 是可平滑过零的近似，不能把参数 `c` 解释成已经测得的
静摩擦阈值。内置 `frictionloss` 设为零，防止同一摩擦计算两遍。
黏性阻尼使用 MuJoCo `implicitfast`；平滑摩擦按本步开始时的速度计算并保持。
因此数值步长也是模型假设的一部分。增大步长或大幅增加低速摩擦，可能引入离散数值振荡。

动手前先算一个值：单轮初速为零，关闭平滑摩擦，施加常值力矩时，本实现第一步速度是

\[
\dot q_1=\frac{\Delta t\,\tau}{J_{load}+I_a+b\Delta t}.
\]

用 `tests/test_actuator_bench.py` 中的解析对照核对。然后把 `armature` 加倍，解释为什么
角加速度不会严格变成原来的一半：负载惯量也在分母中。

## 2. 命令和测量的时序

每一行 CSV 代表一次状态转移：

```text
q[k], v[k] → 记录 torque_command[k] → 限幅/延迟 → MuJoCo 走一步 → q[k+1], v[k+1]
```

`delay_steps=0` 表示本步使用当前命令；`delay_steps=1` 表示本步使用上一条命令。
`reset()` 会清空历史，初始队列全部为零。拟合阶段必须采用同样的初始化，否则会把
上一段运动留下的力矩误认为当前动力学。

CSV 保存连续展开的关节角，不把轮角压回 `[-π, π]`。表头明确单位；时间从零开始，步长
必须一致，相邻行的 next-state 必须相符。读取器遇到乱序或非有限值会拒绝，尚不支持
真实异步日志的重采样。模拟速度测量独立加噪，并非已经实现编码器差分滤波或真实驱动器协议。

预测只用第一个测量状态初始化，之后完全依赖命令向前积分。每一步都用真实下一状态
重新初始化，会把长时漂移隐藏起来，这不属于本实验的预测口径。

## 3. 优化器究竟在拟合什么

参考 [PACE](https://github.com/leggedrobotics/pace-sim2real/tree/f07259c09b517ab5118bb1d01b0a6078cf8e1c31)
的已知输入辨识思路，本实现选择 `I_a, b, c` 三个连续参数和一个整数延迟。
已知输入是关节侧力矩命令，无内部位置 PD。已知惯量、力矩单位及限幅是前提，
不能直接拿未知控制模式的硬件日志替换 CSV。

PACE 官方默认还辨识编码器偏置，采用 Isaac Lab 并行环境与 CMA-ES。本实验没有照搬这些
配置，也没有宣称复现论文结果。连续参数少，使用现有 SciPy 的有界非线性最小二乘即可；
整数延迟在外层枚举，默认 0–4 步，不对离散变量假装求连续梯度。

将两段标定轨迹连接成残差向量，但各自从独立初态开始仿真：

\[
L_{id}=\frac{1}{N}\sum_k\left[
\left(\frac{\hat q_k-q_k}{0.05\;rad}\right)^2+
\left(\frac{\hat v_k-v_k}{0.5\;rad/s}\right)^2\right].
\]

分母是预先选定的物理尺度，不是测量噪声的统计估计。改变它们会改变拟合偏好；不能
观察留出结果后反复改尺度，还继续把该数据称为独立测试。

三个参数在以下教学范围内搜索，并归一化到 `[0,1]`：

```text
armature          0.002 .. 0.060 kg·m²
damping           0.001 .. 0.350 N·m·s/rad
coulomb_friction   0.000 .. 0.200 N·m
```

这些范围没有来自 D1 实测。`max_nfev` 限制每个延迟候选的优化评估预算；数值雅可比还会
额外调用仿真，因此它不等于总轨迹回放次数。求解器只承诺寻找局部解，所有候选的停止
状态保存在结果中。[SciPy 定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

先比较 `fit.json` 中各延迟候选的损失。如果选中的延迟碰到搜索上界，应记录这一点；
扩大范围属于新的开发实验。雅可比满秩只描述当前解附近、固定延迟条件下的连续参数敏感性，
不证明全局唯一，更不是参数置信区间。

## 4. 两种结果要分开解释

`matched` 的生成器和拟合器使用同一模型结构，测量中默认加小幅噪声。它检查实现和时序，
也适合先做 `--noise-scale 0` 的参数恢复练习。成功不代表真实 D1 就服从这个模型。

`mismatch` 只在生成器中增加低速摩擦：

\[
\tau_{extra}=-0.08\exp[-(\dot q/0.3)^2]\tanh(\dot q/0.05)\;N\!\cdot\!m.
\]

拟合器仍然只有三个连续参数。它可能通过改变延迟或阻尼来部分吸收缺失的摩擦。
此时评价未见运动的预测误差，不能要求每个参数恢复真值。`metrics.csv` 将
`|v_measured|<0.3 rad/s` 的样本另列，避免平均误差掩盖低速问题；同时检查该组的样本数。

官方 PACE 建议换激励频率、幅值或 PD 增益验证，本项目用整段不同运动保留这个原则。
[官方验证建议](https://pace.filipbjelonic.com/tutorials/best_practice/)
当前发布例子只覆盖一个合成参数对象、一个噪声种子和有限的运动，不是跨对象鲁棒性评测。

## 5. 自己完成的练习

先把实验预测写在个人记录里，运行后再补数字。只运行命令不能替代这一段判断。

1. **故意把延迟模型设错。** 运行 `--max-delay-steps 0`，另存一个目录。和默认配置比较
   标定损失及阶跃测试的速度 RMSE。即使损失变小，也要问参数是否合理、测试是否同步改善。
2. **降低激励信息量。** 在开发分支中，将第二段标定命令改成第一段的复制；保留原来
   的验证运动。比较各延迟候选的损失间隔和参数变化。重复数据更多不等于激励更丰富。
3. **看模型缺项。** 比较 matched 与 mismatch，检查低速误差。不要为了贴合结果给拟合器
   直接传入生成端的额外摩擦值；若决定扩展模型，应使用新的开发数据与留出运动。
4. **解释辨识损失与控制目标。** 本节最小化预测残差；PPO 的奖励评价任务表现。
   先读 [PPO 数值实验](ppo_learning_lab.md)，说明为什么预测误差下降不能自动推出策略
   回报提高。控制器访问的状态和约束是否与辨识运动相同？哪些模型误差会被反馈放大？

记录至少包含改动的代码位置、运行配置和原始 CSV。若得到失败案例，保留条件和首次出现
异常的时间。维护一份当前说明即可，临时图表在确认被有效结果替代后清理；必要的对照数据
保留，避免失去解释结论的依据。

## 6. 回到整机之前

下一步可让辨识参数进入 IDQP 的独立模型，并在同一个固定的非理想对象上比较名义配置和
标定配置。摩擦前馈要进入总力矩约束，延迟补偿要单独验证；辨识脚本不会自动完成这一步。

轮地接触摩擦、传动间隙以及真实 IMU/编码器融合也不在本台架中。未来加入日志回放或
硬件接口时，还需要确定关节顺序与驱动命令模式。当前文件格式只为数据流提前做准备。

代码阅读顺序：

- [`ActuatorBench.step`](../src/wheel_legged_control/actuator_bench.py)：力矩在什么时候生效。
- [`ActuatorLog` 与拟合器](../src/wheel_legged_control/actuator_identification.py)：哪些数据能被读取。
- [`run_actuator_identification.py`](../scripts/run_actuator_identification.py)：如何划分运动并输出证据。
- [`test_actuator_identification.py`](../tests/test_actuator_identification.py)：参数恢复与留出预测各自验证什么。
