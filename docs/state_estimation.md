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
- `FL / FR / RL / RR` 四个轮端的位置、平移 Jacobian、接触标记和接触点；
- 当前控制时刻、测量时刻以及由两者得到的状态年龄。

所有数组在构造时复制并设为只读。控制器拿到快照后，即使仿真继续推进，这份状态也不会
变化。测试会先保存快照、再任意修改 `plant.data`，确认同一命令的 VMC 输出保持一致。

## 两种 adapter

### `oracle`

`D1MujocoTruthStateSource` 从 MuJoCo 读取同一时刻的完整状态。它用于回归测试、旧 checkpoint
兼容和性能上限，不代表真实机器人可获得的信息。

### `estimated`

`D1NoisyDelayedStateSource` 对完整真值快照按字段加入噪声，并用整数控制步队列模拟状态延迟。
噪声作用于基座位置、姿态、速度、关节、足端运动学和接触点，LQR、MPC、VMC、安全逻辑和
PPO 都会受到影响。

这个 source 只模拟字段噪声和延迟，没有 IMU/编码器融合或 EKF。它能先检查控制器对状态
误差和延迟是否敏感；后续的互补滤波或 EKF 可以替换 source，控制器接口不用再改。

## 延迟语义

动作延迟和状态延迟现在是两个独立参数：

- `action_delay_steps`：策略残差从产生到施加的延迟；
- `state_delay_steps`：整条控制链拿到的测量年龄。

在 `100 Hz` 控制频率下，`state_delay_steps=3` 对应稳态 `30 ms`。旧参数 `delay_steps`
只保留“残差动作延迟”的含义，状态延迟必须显式写 `state_delay_steps`。CSV 每行保存两个
延迟、噪声比例、estimator seed 和 `state_age_p95_ms`；`evaluation_config.json` 记录源码
commit/dirty 指纹和 policy SHA-256。

训练奖励、摔倒判断和最终 RMSE 使用 MuJoCo 真值。这些量只负责监督和评测，不送回控制器。
环境的 `info` 同时提供带 `truth_` 与 `estimated_` 前缀的字段，避免 callback 把两者混用。

## 运行对照

真值状态保留旧行为：

```bash
wheel-legged-d1-benchmark --state-mode oracle
```

让完整控制链承受状态噪声和延迟：

```bash
wheel-legged-d1-benchmark \
  --state-mode estimated \
  --no-policy
```

交互课程也可以切换：

```bash
wheel-legged-d1-play \
  --state-mode estimated \
  --state-delay-steps 2 \
  --sensor-noise 1.0
```

训练多个独立策略种子：

```bash
wheel-legged-train \
  --robot d1 \
  --state-mode estimated \
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
