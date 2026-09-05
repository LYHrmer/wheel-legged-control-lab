# D1 状态反馈接口

## 为什么要单独做这一层

v0.3 的 LQR、MPC 和 VMC 直接读取 MuJoCo 的基座速度、姿态、关节状态和接触点。策略观察
虽然加了噪声，但经典控制仍然使用无延迟真值。这适合先验证控制结构，不适合回答“换成真实
传感器后还能不能工作”。

v0.4 在动力学和控制器之间加入一份不可变状态快照：

```text
MuJoCo state source ─┐
                     ├─> D1StateEstimate ─> VMC / LQR / MPC / PPO / safety
noisy delayed source ┘
```

一次控制周期只生成一份快照。低层力矩、外层反馈、地形姿态、PPO 观察和跳跃准入使用同一
时刻的数据，避免同一周期里混入不同采样时刻。

## 快照里有什么

`D1StateEstimate` 固定使用 SI 单位和以下顺序：

- 机身位置和 `base_link -> world` 旋转矩阵；
- 机体系与世界系基座线速度、角速度；
- 16 个关节的位置和速度，顺序与 `D1_JOINT_NAMES` 一致；
- `FL / FR / RL / RR` 四个轮端的位置、平移 Jacobian、接触标记、接触点、法向和接触点 Jacobian；
- 当前控制时刻、测量时刻以及由两者得到的状态年龄。

所有数组在构造时复制并设为只读。控制器拿到快照后，即使仿真继续推进，这份状态也不会
变化。测试会先保存快照、再任意修改 `plant.data`，确认同一命令的 VMC 输出保持一致。

## 两种 adapter

### `oracle`

`D1MujocoTruthStateSource` 从 MuJoCo 读取同一时刻的完整状态。它用于回归测试、旧 checkpoint
兼容和性能上限，不代表真实机器人可获得的信息。

### `estimated`

`D1NoisyDelayedStateSource` 对完整真值快照按字段加入噪声，并用整数控制步队列模拟状态延迟。
噪声作用于基座位置、姿态、速度、关节、足端运动学、接触点、接触法向和接触点 Jacobian，
LQR、MPC、VMC、安全逻辑和 PPO 都会受到影响。

这个 source 只模拟字段噪声和延迟，没有 IMU/编码器融合或 EKF。它能先检查控制器对状态
误差和延迟是否敏感；后续的互补滤波或 EKF 可以替换 source，控制器接口不用再改。

## 可选的短时延迟补偿

`constant_velocity` 使用快照自身的机体系速度、角速度和关节速度，把连续量外推到当前控制
时刻：

\[
\hat R=R\operatorname{Exp}(\omega_b\Delta t),\qquad
\hat p=p+R\operatorname{Exp}(\tfrac12\omega_b\Delta t)v_b\Delta t,
\qquad
\hat q=q+\dot q\Delta t.
\]

足端位置用冻结 Jacobian 做一阶更新。接触标记、接触点、法向、接触点 Jacobian 和非轮接触
数量保持原测量值，程序不会根据 MuJoCo 当前帧补齐它们。constrained 分配器仍会使用这组
旧时刻几何。超过 `50 ms`、腿关节外推量超过 `0.35 rad`，或预测位置越过关节限位时，控制器
退回原始快照并在日志中写明拒绝原因；轮关节位置保持无界。

外推后的 `measurement_time_s` 仍是原测量时刻，所以 `state_age_ms` 不会变成零。日志另存
`latency_compensation_horizon_ms` 和状态 `bypassed / applied / horizon_exceeded /
kinematic_horizon_exceeded`。`raw_estimated_state` 与实际送入控制器的 `control_state` 会同时
保留，以便直接计算补偿前后的估计误差。这个方法只检验“短时常速度假设能否抵消一部分
延迟”，不估计 bias、协方差或接触切换。

## 延迟语义

动作延迟和状态延迟现在是两个独立参数：

- `action_delay_steps`：策略残差从产生到施加的延迟；
- `state_delay_steps`：整条控制链拿到的测量年龄。

在 `100 Hz` 控制频率下，`state_delay_steps=3` 对应稳态 `30 ms`。旧参数 `delay_steps`
只保留“残差动作延迟”的含义，状态延迟必须显式写 `state_delay_steps`。CSV 每行保存两个
延迟、噪声比例、estimator seed 和 `state_age_p95_ms`；`evaluation_config.json` 记录源码
commit/dirty 指纹和 policy SHA-256。

训练奖励、摔倒判断和最终 RMSE 使用 MuJoCo 真值。这些量只负责监督和评测，不送回控制器。
环境的 `info` 同时提供 `truth_state`、`raw_estimated_state` 和 `control_state`；旧字段
`estimated_state` 暂时作为 `control_state` 的兼容别名。

## 运行对照

真值状态保留旧行为：

```bash
wheel-legged-d1-benchmark --state-mode oracle --contact-allocation legacy
```

让完整控制链承受状态噪声和延迟：

```bash
wheel-legged-d1-benchmark \
  --state-mode estimated \
  --contact-allocation legacy \
  --no-policy
```

固定延迟集合的配对实验：

```bash
wheel-legged-d1-benchmark \
  --state-delay-sweep \
  --state-mode estimated \
  --latency-compensation none \
  --contact-allocation legacy \
  --audit-episodes 30 \
  --seed 21 \
  --no-policy \
  --output results/d1_state_delay_raw

wheel-legged-d1-benchmark \
  --state-delay-sweep \
  --state-mode estimated \
  --latency-compensation constant_velocity \
  --contact-allocation legacy \
  --audit-episodes 30 \
  --seed 21 \
  --no-policy \
  --output results/d1_state_delay_compensated
```

延迟固定为 `0/10/20/30/50 ms`。动作延迟和噪声关掉；程序核对每个评测种子的域参数、
估计器种子、初态、初始命令和计划推力指纹。每个条件少于 20 个回合时，报告标为探索性结果。
当前 30-seed 结果见 [`raw`](../results/d1_state_delay_raw/state_delay_sensitivity.md) 与
[`constant_velocity`](../results/d1_state_delay_compensated/state_delay_sensitivity.md)。补偿在
`10 ms` 下减小了部分误差；`20 ms` 的成功数变化还没有得到不跨 0 的配对区间，且约三分之二
控制步触发运动学拒绝。

交互课程也可以切换：

```bash
wheel-legged-d1-play \
  --state-mode estimated \
  --contact-allocation legacy \
  --state-delay-steps 2 \
  --sensor-noise 1.0
```

训练多个独立策略种子：

```bash
wheel-legged-train \
  --robot d1 \
  --state-mode estimated \
  --latency-compensation none \
  --contact-allocation legacy \
  --steps 400000 \
  --envs 8 \
  --seed 7 \
  --runs 5 \
  --device cpu \
  --output results/d1_estimated_ppo
```

使用 8 个环境时，多次训练写入 `seed_0007`、`seed_0015` 等目录，避免相邻 run 复用七条
环境随机流。每个目录单独保存训练开始时的 Git commit/dirty 指纹、Python 与关键依赖版本、
实际设备、实际训练步数和 checkpoint SHA-256；根目录另有本轮 manifest。

## 下一步怎么换成真正估计器

后续实现应从 IMU、编码器、轮速和可选接触信号构造状态，至少包含：

1. 陀螺仪积分与加速度计重力校正；
2. 编码器和轮速的机体系速度约束；
3. bias、随机游走、异步采样和丢包；
4. 支撑概率与相对地形高度；
5. innovation、协方差或置信度，以及超时时的安全降级。

在这些完成前，仓库只声称“估计状态通道下的鲁棒性实验”，不声称已经实现实机状态估计。
