# 从轮速环到命令条件 PPO

这轮实验把整机训练接到同一个控制循环。运行时可以选择两维 LQR/MPC 力残差，或八维腿伸缩、轮速残差。命令包含前进速度、偏航角速度和离地净空。代码仍用 CPU MuJoCo，控制周期 10 ms，每拍包含五次 2 ms 的物理积分。

先保留一个明确的失败：八维控制在无延迟转向探针中，累计转角约为指令积分的 76.7%；20 ms 传感器延迟下仍会跌倒。源码和原始轨迹见[动作探针记录](../results/d1_v3_action_probes/README.md)。这些数字属于零残差低层控制，不能算作 PPO 的训练收益。

## 先跑零残差

以下命令在仓库根目录、装好项目 RL 依赖后执行。每个输出目录都必须是新的，脚本不覆盖旧实验。

```bash
python scripts/run_d1_locomotion.py --source oracle --seconds 12 \
  --output results/my_locomotion_flat

# 60 s 的开发道路。完成后仍须检查实际曝光和跟踪误差。
python scripts/run_d1_locomotion.py --source oracle --seconds 60 \
  --terrain development --terrain-index 0 --output results/my_locomotion_road

# 在本机桌面显式打开键盘窗口；不加载 PPO 时就是零残差。
python scripts/run_d1_locomotion.py --keyboard --source sensor \
  --output results/my_locomotion_keyboard
```

新键盘入口用 W/S 改前进速度，A/D 改转向，R/F 升高/降低净空。Space 请求停车，Esc 退出；没有按键输入一段时间后，速度与转向命令归零。它不提供跳跃。旧入口的 Space 是跳跃，两套按键不要混用。

`telemetry.csv` 记录已经执行的命令及误差；`states.npz` 保存真实仿真状态和各物理子步施加的力矩。回放这份 CSV 时，使用相同道路、状态源及 seed：

```bash
python scripts/run_d1_locomotion.py --source oracle --seed 17 \
  --command-csv results/my_locomotion_flat/telemetry.csv \
  --output results/my_locomotion_replay
```

键盘代码已经过自动化检查，自动化测试不代表有人实际操作过图形窗口。

## 一拍里发生了什么

```text
当前状态发布 → 当前命令与基线预览 → 82 维观测 → 策略动作
                                              ↓
下一拍状态发布 ← 五次物理积分/执行器响应 ← 控制器输出 16 轴力矩
```

[控制循环](../src/wheel_legged_control/d1/control_loop.py)要求先 `prepare(command)` 再 `step(action)`。重复读取同一拍不会再次积分 PI，也不会重新消耗测量噪声。动作执行后，环境准备下一拍的命令与观测。因此 CSV 里的已执行命令来自 transition，不能从刚返回的下一拍观测反推。

旧 MuJoCo 读取路径曾把积分后的 `qpos` 和积分前的派生位置放在一起。新路径在独立 `MjData` 上复制并前向计算测量，保留真实积分器的 warm-start 状态。历史实验默认的 `legacy_mixed` 路径仍保留，不能把旧数据静默解释成同步采样。

还有一个坐标问题容易漏掉：`base_position` 是机身坐标系原点，线速度是该刚体惯性 COM 的速度。若 COM 相对原点的偏移为机身系向量 r，则

\[
v_{origin}^{world}=v_{COM}^{world}-\omega^{world}\times(Rr).
\]

高度误差用原点位置，高度阻尼就应使用对应的原点速度。先在[采样测试](../tests/test_d1_sampling.py)里找有限差分检查，再读力残差适配器的补偿。不要用速度很小作为两个点等价的理由。

## 八维动作到底控制哪里

| 输出 | 物理含义 | 归一化动作 ±1 的尺度 |
|---|---|---|
| 0–3 | 四腿在名义姿态附近的轮心伸缩修正，正值向下伸腿 | ±4 cm |
| 4–7 | 四个轮子的速度目标修正 | ±4 rad/s |
| 旧两维对照 0–1 | 纵向、竖直机身力残差 | ±11.25 N、±20 N |

八维方案同时更换了低层控制结构。与两维方案的差异不能全部归因于动作维数，也不能把零残差控制器的进步算成 RL 收益。每种控制器都单独保留零残差对照。

腿部 IK 在启动时生成查表，运行时插值到关节目标。随后是关节 PD、支撑力矩与轮速 PI。速度目标限幅和力矩限幅都不能保证碰撞瞬间的实测关节速度一定小于限值；探针仍需记录实际状态。

四轮差速的几何目标为 `wheel_speed = (vx - yaw_rate * lateral_offset) / 0.087`。D1 需要侧向滑移才能原地转向，轮速跟上不意味着机身转角跟上。当前额外使用有界偏航反馈：

\[
\dot\psi_{request}=clip(\dot\psi_{cmd}+4(\dot\psi_{cmd}-\omega_z),-0.6,0.6).
\]

增益扫描保留了 0、2、4、6 的结果。4 是首先通过 70% 转角门槛的值；6 的转角更大，但没有因此选最大增益。这里的反馈量是机身系 gyro z，小倾角时接近欧拉偏航角速度，大倾角下二者不同。

轮速环默认 `Kp=2.2, Ki=3.0`，偏航反馈增益为 4。可以在运行或训练命令中显式传入
`--wheel-kp 0.55 --wheel-ki 1.5 --yaw-feedback-gain 4` 做对照；仅限 `wheel_leg` 基线。
还可分别用 `--leg-feedback-scale` 和 `--attitude-feedback-scale` 缩放腿部 PD
80/3 与姿态 PD 180/24，默认均为 1，必须为正数；每一组的 Kp、Kd 同时乘以该比例。
重力支撑和 IK 目标不随比例改变。加载策略时必须匹配全部五个参数。
旧 sidecar 只有三个轮速/偏航增益时，仅兼容未缩放的旧 schema；完全缺少增益记录时，
仅兼容原固定默认控制器。相同观测维数不足以证明策略可互换。

动手检查：先读[控制器](../src/wheel_legged_control/d1/wheel_leg_controller.py)，预测单腿动作 `+0.5` 的目标变化。应是 2 cm 伸长，不是 0.5 rad。对照每腿轮心高度、目标关节角及力矩，解释实际伸长为何不一定等于 2 cm。

## 观测与奖励

[82 维编码](../src/wheel_legged_control/d1/locomotion_observation.py)包含允许的状态估计、当前命令、当前基线、控制器记忆和上一拍实际动作。两维与八维控制共用编码长度，但基线类别和动作 schema 不同。高度参考来自所选状态源：oracle 读取碰撞几何，fusion 用 IMU/编码器与理想接触开关估计支撑面。后者仍需要已知初始放置姿态，没有实现真实定位或在线 IMU bias 估计。

控制器记忆的缺失值有显式 mask。例如轮速 PI 的积分为零，与这个控制器根本没有轮速积分，不是同一条信息。连续两个状态看似相同、但积分不同，下一拍力矩也可能不同。

[奖励代码](../src/wheel_legged_control/d1/locomotion_rewards.py)把普通项当作奖励率，乘以控制周期；摔倒罚分是一次事件。速度和偏航跟踪使用指数项，高度与姿态误差用归一化平方项。

机械活动量按五个子步的 `sum(abs(applied_torque * joint_velocity))` 求均值，单位 W。两个轴一正一负做功不会互相抵消。这还不是电池功率：没有电机效率、驱动器损耗或回生模型。动作变化项衡量归一化策略指令的变化，不能标成 W 或 Nm。

检查一个数：平均机械功率 13 W、权重 0.02、尺度 200 W、周期 0.01 s 时，该项为 `-0.000013`。把周期改成 0.02 s，贡献应翻倍；一次终止的 `-2` 不翻倍。对应手算测试在[奖励与观测检查](../tests/test_d1_locomotion_signals.py)。

## 从真实 PPO 更新里复算

```bash
# 实测 1/2/4 worker；不要从 CPU 核数猜采样速度。
python scripts/run_d1_locomotion_experiment.py benchmark --steps 1024 \
  --output results/my_locomotion_throughput

# 固定预算的学习起点。32768 步不保证策略有收益。
python scripts/run_d1_locomotion_experiment.py train --seed 24000 --steps 32768 \
  --workers 4 --output results/my_locomotion_ppo
```

执行前核对可用内存。实验脚本最多安排四个环境，正式评测同时报告时间和资源占用。

[更新审计器](../src/wheel_legged_control/d1/ppo_update_audit.py)继承 SB3 的 PPO 更新，不重新实现优化目标。每次更新保存前 128 个 env-major 样本的观测和原始 Gaussian 动作，以及更新前后概率。这个片段偏向第一个环境，不能当成全 rollout 的无偏统计；训练 CSV 才覆盖每个环境的所有采样。

按以下顺序复算一个 `updates/sample_000000.npz`：

1. 用完整 `gae_rewards/gae_values/gae_episode_starts` 和末端值倒序递推 GAE，再按 env-major 顺序抽取，核对 `advantages`。
2. 核对 `returns = advantages + old_values`。仅验证这个恒等式不够证明 GAE 递推正确。
3. 计算 `exp(log_prob_after - old_log_prob)`，核对概率比。更新前的比率应接近 1。
4. 分别选一个正优势、负优势样本，手算 PPO 裁剪分支。最终比率仍可能超出裁剪区间，clip 不是硬约束。

这里的 `gae_rewards` 已含 SB3 对时间上限的价值 bootstrap，可能与环境原始奖励不同。结束掩码不能让下一局奖励接回上一局。审计记录还保存了参数 hash；权重确实变化与控制性能改善是两项不同检查。

还有一种失败发生在物理积分完成后：融合的支撑面参考有限，但已超出低层控制器的允许范围。
当前环境将其记为 `control_reference_out_of_envelope`，属于真正终止，计入一次失败罚分。
此时没有可执行的下一拍基线，terminal observation 使用上一份可执行 decision 的编码，
并在 info 标明 `terminal_observation_source=last_executable_decision`；它不能用于超时
bootstrap。真实末状态和已执行命令仍从 transition 记录。NaN、普通数值异常和无效配置
不会被转换成这样的终止。旧源码可能在此抛异常，重现时要区分这个停止规则修复。

现在可以用[独立数学复算器](../scripts/audit_d1_ppo_math.py)检查一个训练目录，不加载 checkpoint：

```bash
python scripts/audit_d1_ppo_math.py results/my_locomotion_ppo/updates \
  --output results/my_ppo_math.json
```

它从折扣 TD 误差展开求和，与学习器的倒序递推交叉核对；再用保存的均值、标准差和原始动作
重算 Gaussian 对数概率、概率比、KL 近似和裁剪比例。输出中的正负优势例子使用原始优势，
方便手算；SB3 实际训练使用 minibatch 归一化优势，不能把例子的均值冒充优化器损失。
已归档六模型的 384 次更新全部通过数值检查，见[复算记录](../results/d1_v3_locomotion_report/ppo_math_audit.json)。

加载新模型时必须给 sidecar：

```bash
python scripts/run_d1_locomotion.py --source oracle \
  --policy results/my_locomotion_ppo/checkpoint.zip \
  --metadata results/my_locomotion_ppo/checkpoint.json \
  --output results/my_locomotion_policy_replay
```

旧 42/44/45 维 checkpoint 不兼容。相同维数但基线或状态源不同也会被拒绝。只加载自己生成或信任的 SB3 文件，其序列化格式能够执行 Python。

## 过地形的证据怎样看

固定地图为 12×6 m，出生点在 `(-3.8, 0)` 的平地。若 12 s 回合只前进一小段，地图里存在起伏并不能说明策略遇到了它。环境记录轮心/机身投影处的实际地形高度和坡度是否超过曝光阈值，并累计距离。这个指标是几何曝光，尚不等于轮子保持接触并成功越过整个障碍。

开发与留出道路的布局和生成参数分开，仍只是一个小范围地形集合。名义 60 s 命令也不一定到达地图末尾的台阶。先在轨迹中确认遇到了哪些地形，再比较误差和完成数。地图边界退出按失败终止，不算成功走完整张地图。

高度、速度和偏航的质量门槛独立于奖励。训练回报上升却高度误差增大时，应保留这项退化。

播放一次策略只产生演示记录。正式评测要在相同条件下分别运行零残差和策略，并检查整组道路：

```bash
python scripts/run_d1_locomotion_experiment.py evaluate \
  --baseline wheel_leg --source oracle --split development --duration 60 \
  --output results/my_zero_development

python scripts/run_d1_locomotion_experiment.py evaluate \
  --baseline wheel_leg --source oracle --split development --duration 60 \
  --policy results/my_locomotion_ppo/checkpoint.zip \
  --metadata results/my_locomotion_ppo/checkpoint.json \
  --output results/my_policy_development

python scripts/audit_d1_locomotion.py results/my_policy_development \
  --zero results/my_zero_development --output results/my_paired_audit.json
```

日常改参只用 `development`。先写下固定预算和选择规则，再将两条评测命令的 split 都改成
`holdout`，输出到新的留出目录。看过留出后再挑参数，这批道路就只能算开发数据。融合来源的
CLI 名称是 `imu_encoder_fusion`；与运行入口的 `--source sensor` 是同一个来源的不同命名，
不要将旧入口的 `estimated` 混入这里。

六个固定预算模型现已完成开发/留出评测，见[实验报告与完整录像](../results/d1_v3_locomotion_report/README.md)。轮腿 PPO 在留出道路上的平均速度 RMSE 从 0.04016 降到 0.03328 m/s，但高度 RMSE 增加约 1.92 mm，机械活动量增加约 4.47 W。练习：从原始 CSV 分别复算速度、高度和功率，不只检查总回报；解释这个策略改善了什么、付出了什么代价。

默认控制器在整包测量延迟 20 ms 时会在进入起伏道路前失稳。反馈缩放的诊断配置已经能
驶入起伏区域，仍未通过完整任务，见下面的失败案例。

## 一个短探针通过后仍然失败的例子

保持腿部控制不变，将轮速 PI 调为 `0.55/1.5`，在 20 ms 延迟的平面探针上完成了两次 8 s
转向，转角分别达到命令积分的 70.83% 和 70.71%。这个结果刚刚超过预定的 70% 门槛。
同一组增益接入完整开发任务后，四个案例却都在 3.1 s 以内失稳，当时还没开始运动命令；
无延迟的四个案例均完成 60 s。不能据此替换默认增益，也不适合直接加大 PPO 训练预算。

这适合做一次诊断练习。先用同一测量 seed 复跑，再检查两个程序的地面碰撞表示、reset
顺序和姿态参考。地图高度为零不保证碰撞几何相同。每次只改变一个因素，记录实际关节速度
与施加力矩；超过限幅后的执行结果也要记录。假设只有得到对照支持后才能写进结论。

后续 2×2 对照排除了中性随机化调用；单独降低腿部或姿态 PD 后，4 s 站立通过，
完整道路仍在约 20–22 s 失败。具体参数、假设和复现命令集中在
[延迟诊断报告](../results/d1_v3_delay_diagnosis/README.md)，不按成功率挑删实验。

接着做[延迟与短历史练习](delay_learning_lab.md)：保持低层五参数一致，再比较单帧、
四帧和随机化训练。先核对 10 ms 测量拍与 2 ms 执行器拍，再读完成数和失败原因。
