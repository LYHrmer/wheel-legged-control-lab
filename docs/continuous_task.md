# D1 连续 45 秒任务

这个任务补的是四秒直行实验没有覆盖的部分：同一次运行里停车，再开动，经过不同地形，最后低速左右转向。默认运行 4500 个控制步，中途不 reset。地图在开始前生成一次，机器人靠移动进入下一段地形。

早期使用仿真真值的零残差开发基线完成了 45 秒。左右转角分别为 +0.261 rad 和 -0.147 rad，距离命令积分仍有差距；最终横向路径误差为 -0.116 m。这只是一条固定路线的开发结果。

新传感器 PPO 已完成三个种子各 131072 步训练。最终策略都跑完了六个 45/50 s 留出条件，但质量通过率仍为零，离地高度 RMSE 比配对基线增加约 3.34 mm。完整记录见 [实验结果](../results/d1_continuous_policy/README.md)。

## 路线与命令

机器人从 `(-3.8, 0, 0.455)` m 出生。碰撞地图长 12 m、宽 6 m，边缘留 0.5 m 的终止缓冲区。MuJoCo 高度场与地面查询共用同一份离散采样，测试直接用 `mj_ray` 检查高度和法向。

地形沿世界 x 轴排列。`[-3.0, -1.5]` m 是振幅 5 mm、波长 0.8 m 的起伏；进入和退出的 0.2 m 有平滑包络。随后是最大坡度 +4° 的上坡、0.4 m 长的平台及 -4° 下坡，`x > 1.7` m 恢复平地。坡度的首尾也有 0.2 m 平滑过渡。整张地图不会在机器人脚下改变。

| 时间 / s | 命令速度 / m·s⁻¹ | 命令偏航角速度 / rad·s⁻¹ | 操作 |
|---|---:|---:|---|
| 0–2 | 0 | 0 | 站稳 |
| 2–10 | 0.25 | 0 | 开动，经过起伏 |
| 10–13 | 0 | 0 | 停车 |
| 13–23 | 0.25 | 0 | 继续行驶 |
| 23–29 | 0.16 | 0 | 减速 |
| 29–35 | 0.12 | +0.08 | 左转 |
| 35–41 | 0.12 | -0.08 | 右转 |
| 41–45 | 0 | 0 | 停车 |

阶段切换使用 0.6 s 的 smoothstep 命令过渡。表里的操作描述不参与地形判定；CSV 的 `terrain_section` 来自实际位置。速度误差可能让机器人晚于计划抵达坡面，所以不能直接把时间阶段当成已经通过某段地形。

`--duration` 支持 30–60 s。时间按比例缩放，速度和偏航角速度反向缩放，保持命令路线长度。默认 45 s 是已验证条件，不能把它的通过结果直接用于其他时长。

## 停车后为什么开始抖

初次沿用偏航比例增益 5，直行开头没有异常；停车后出现约 1.3 rad/s 的偏航速度振荡，37.61 s 越界结束。同一道路和同一组命令，只把增益降到 2，便能完成 45 s，但左右转角只有 +0.075/-0.094 rad，转向跟踪不合格。

新任务自己的偏航支路采用 `kp=2, ki=3`。积分项限制在 ±2 N·m，轮侧差动转矩限制在 ±4 N·m；比例项已饱和且误差继续推向饱和方向时，不再累积积分。公共控制器文件没有改动。LQR 的纵向力、VMC 的支撑与腿部增益均保持原值。

| 保存的对照 | 完成时长 / s | 左 / 右实际转角 / rad | 质量判据 |
|---|---:|---:|---|
| `zero_legacy_p5` | 37.61 | +0.246 / -0.331（右转未完成） | 未通过 |
| `zero_p2_no_integral` | 45.00 | +0.075 / -0.094 | 未通过 |
| `zero_pi_development` | 45.00 | +0.261 / -0.147 | 通过开发门槛 |

PI 的左右转向稳态角速度 RMSE 为 0.0359/0.0500 rad/s。这里的“通过”允许明显的跟踪误差：逐阶段剔除首秒后，纵向速度 RMSE ≤0.14 m/s，侧向速度 RMS ≤0.10 m/s，偏航角速度 RMSE ≤0.06 rad/s，离地高度 RMSE ≤0.025 m；左右转角还需各达到该阶段命令积分的 25%–175%，并完成全部阶段、实际经过全部地形。这个转角门槛用于排除几乎不转弯的假通过，不是精确路径跟踪指标。

这些门槛和 PI 参数是在开发过程中确定的，不能当作事先冻结的留出集评价。要继续提高要求，可以先固定新的路线和种子，再测试更严格的转角比例与路径误差限制。

## 运行与视频

在仓库根目录执行，输出目录必须不存在：

```bash
env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/run_d1_continuous_task.py --mode zero --output results/my_continuous_zero

env PYTHONPATH=src MUJOCO_GL=egl \
  python3 scripts/render_d1_continuous_task.py --run results/my_continuous_zero \
  --output results/my_continuous_zero_video.mp4
```

每次运行保存 `protocol.json`、逐步 `telemetry.csv`、`states.npz`、`summary.json` 和 SHA256 清单。当次实际编译模型含碰撞地形，无损压缩后存入同级 `models/<SHA>.mjb.gz`，相同模型由各次运行共享。清单同时核验压缩文件和解压后模型的哈希。视频加载这份模型和同次 qpos/qvel，只计算几何位置，不调用控制器或物理积分。

视频保留初始帧和终止帧，标明零残差或 PPO、模型哈希、命令速度与残差力。速度一栏是仿真真值审计，当前 renderer 明确写成 `truth`，不能把它当成策略收到的传感器估计。下方剖面来自存档碰撞高度场，纵轴单独标注；不能凭它的视觉高度判断实物起伏。默认 20 fps，每五个控制步取一帧，45 s 运行加初始展示帧对应 45.05 s 视频。

## Checkpoint 的限制

旧 oracle-v2 的 44 维观测布局在 oracle 模式下保留，动作仍是两维力残差，满量程为纵向 11.25 N、竖直 20 N。这份布局没有偏航命令。旧 checkpoint 用于这条路线时，停车和转向都属于训练任务之外的测试，偏航 PI 也属于新控制配置。

必须显式允许这次迁移，并提供有哈希与 schema 的模型元数据：

```bash
env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/run_d1_continuous_task.py --mode policy \
  --checkpoint /path/to/model.zip --metadata /path/to/metadata.json \
  --allow-oracle-v2-task-transfer --output results/my_continuous_policy
```

runner 检查 checkpoint SHA256、观测和动作尺寸及 schema。PPO 只在 policy 模式导入，使用 CPU 单线程，确定性推理。没有 checkpoint 的运行不能标成 PPO。

动作接口只有两种注册映射：`d1-terrain-residual-quarter-v1` 的 `[ax, az]`，以及 `d1-terrain-residual-longitudinal-only-v1` 的 `[ax] → [ax, 0]`。后者的竖直残差在每一步严格为零。新元数据必须包含 `residual_force_scale_n`，分别为 `[11.25, 20.0]` 或 `[11.25]`；旧 v2 缺这个字段时，仅显式迁移选项允许按已知 schema 处理。未知 schema 即使维数相同也会拒绝。

传感器模式使用 `--state-source sensor`。初始位姿是任务给定的放置先验；其后 command 和 observation 使用传感器估计的地面，不读碰撞地图高度或法向。真值查询只用于误差记录和终止判定。这个模式有独立观测标签，会拒绝旧 oracle checkpoint。已有同路线的传感器估计控制记录 `zero_sensor_development`：完成 45 s，左右转角 +0.267/-0.168 rad，但左转阶段离地高度 RMSE 为 0.0262 m，略超 0.025 m 门槛，所以整体质量仍记为未通过。

## 给新 PPO 的接口

`observation_layout="command45"` 在原 44 项后增加第 45 项偏航命令，单位 rad/s。oracle 与 sensor 的 schema 分别为 `d1-continuous-oracle-command45-v1`、`d1-continuous-sensor-command45-v1`。训练出来的模型需要使用同一布局，44 维 checkpoint 不能自动补零迁入。

```python
from functools import partial
from wheel_legged_control.d1.continuous_task import ContinuousTaskConfig, D1ContinuousTask
from wheel_legged_control.d1.sensor_estimation import D1SensorStateSource

config = ContinuousTaskConfig(
    duration_s=45.0,
    ground_reference_mode="estimated",
    observation_layout="command45",
    yaw_controller="pi", yaw_kp=2.0, yaw_ki=3.0,
)
source_factory = partial(
    D1SensorStateSource,
    initial_position=(-3.8, 0.0, 0.455),
    initial_rpy=(0.0, 0.0, 0.0),
)
env = D1ContinuousTask(config, state_source_factory=source_factory)
```

配置和 `partial` 可序列化，便于子进程创建环境。`reset(seed=...)` 返回 `(obs, info)`，`step(action)` 返回 Gymnasium 的五元组。跑满时长记为 `truncated`；跌倒、非有限状态和地图越界记为 `terminated`，不会自动重置。控制周期为 0.01 s。

每步原始奖励是

```text
exp(-(vx_error / 0.2)^2)
+ exp(-(yaw_rate_error / 0.15)^2)
- (true_clearance_error / 0.08)^2
- 5 * terminated
```

训练端可以显式包 reward scale，但日志应保留原始值。PI 积分和 LQR 的内部距离参考没有单独进入 45 维观测，这份接口仍带有部分可观测性；新增偏航命令不等于补全了所有控制器内部状态。

## 匹配传感器输入的 PPO 训练

`scripts/train_d1_continuous_policy.py` 从新初始化的策略开始。PPO 在每次 45 s 运行中接收 IMU/编码器融合结果，地面命令也由同一个传感器状态源计算。真值只用于奖励与终止判定。偏航差动转矩仍由 PI 输出，策略只调节纵向与竖直残差力；它可以学转向时的支撑补偿，但这轮实验没有让 RL 直接输出偏航转矩。

传感器噪声逐样本独立，标准差为陀螺仪 0.002 rad/s、加速度计 0.03 m/s²、关节位置 0.0005 rad、关节速度 0.005 rad/s。偏置和测量延迟设为零。这些是用于学习的示例幅值，没有真实硬件辨识依据。20 ms 延迟已在单独的传感器矩阵中暴露失稳，不应从这份零延迟策略推断部署稳定性。

| 训练项 | 固定设置 |
|---|---|
| 独立训练种子 | 19000、20000、21000 |
| 每个种子的 checkpoint | 32768、65536、131072 个物理环境步 |
| 并行环境 | 4 个；整台电脑一次只运行一组训练 |
| Actor/Critic | `[64, 64]`，CPU，Torch/BLAS 单线程 |
| PPO | `n_steps=128`、`batch_size=128`、`n_epochs=4`、学习率 `3e-4` |
| 折扣与 GAE | `gamma=exp(-0.01/2)`、`gae_lambda=0.95` |
| 奖励缩放 | PPO 接收原始奖励的 0.01 倍，Monitor 保留原始奖励 |
| 开发评价 | 噪声种子 17/29/43，45 s；中间预算只跑协议写定的一个种子 |
| 最终留出评价 | 噪声种子 617/629/643，各跑 45 s 和 50 s |

每个最终模型都跑完六个留出条件，保留全部三个训练种子；没有按评价成绩选 checkpoint。零残差基线只运行一次同条件记录，配对比较时复用其值，不能把复用算作新的独立实验。留出条件仅改变测量噪声种子与命令时长，地图仍是这张固定道路，不提供新地形泛化证据。

`n_steps=128` 对每个 worker 是 1.28 s。一次 PPO 更新处理四个 worker 的 512 个样本，45 s episode 会跨过多次更新。奖励主要约束当前速度和高度，折扣的时间尺度为 2 s；“能运行 45 s”不等于策略学会了 45 s 长时序规划。读训练曲线时也要同时看 episode 长度，提前跌倒会改变累计回报的可比性。

高斯策略先采样原始动作，环境再裁剪到 `[-1,1]`。训练保存这两份动作，不用正态分布尾概率代替实际裁剪率。裁剪本身不说明 PPO 的概率计算错误；要检查 buffer 保存的是哪份动作，再检查 log probability 是否使用同一份。

自动 reset 的 `seed=None` 由每个 worker 自己的确定性随机流生成新测量种子，每次 seed 都写入 JSONL。开发评价和 checkpoint 重载还会保存、恢复训练主进程的全部 RNG；否则一次 `PPO.load` 可能重设后续探索噪声，训练曲线便会受评价频率影响。

传感器支撑面返回的是世界 x/y 方向各自的坡角。把它换成机体目标姿态时，使用 `slope_x=-tan(pitch)`、`slope_y=tan(roll)`，再按当前偏航旋转地面法向。它们不是依次复合的 Euler roll/pitch。修正后的控制标签为 `d1-lqr-vmc-sensor-world-slopes-yaw-pi-v2`；旧 `zero_sensor_development` 仍作为历史记录保留，不能与新协议的配对零基线混用。

正式训练和独立复算使用新输出目录：

```bash
env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/train_d1_continuous_policy.py --output results/my_sensor45_training

env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 scripts/analyze_d1_continuous_policy.py --run results/my_sensor45_training \
  --output results/my_sensor45_analysis
```

`--smoke` 只训练 1024 个样本，评价只执行 16 个控制步，不碰留出条件。它用于检查短训练到 checkpoint 重载的完整流程，不能拿来报告任务成功率。正式产物保留源文件归档和模型 SHA；复算脚本从逐步 CSV 重算质量判据，还会检查连续状态记录与实际动作裁剪。

独立复算还查出一个底层时间约定：当前 plant 在最后一次 `mj_step` 后没有刷新派生运动学，因此 CSV 中通过 `xpos` 取得的机体位置属于最后 2 ms 物理子步积分前，`states.npz` 的 `qpos` 已是积分后。真实 smoke 的全部位置逐步满足 `truth_xyz = qpos_xyz - 0.002*qvel_xyz`，误差为零；直接把两份位置视为同一时刻会相差约 0.17 mm。复算会按这个明确关系检查，视频展示积分后的 `qpos`，不把不同时间相位的数据强行说成完全同步。冻结训练沿用此约定；若增加 `mj_forward`，应另开控制版本并重新评价。

## 建议亲手检查的部分

从 `zero_p2_no_integral` 与 PI 的转向 CSV 开始，画 `yaw_rate_error_rps` 和 `yaw_integral_nm`。积分项会在转向时逐渐建立，但反向指令到来后需要时间卸掉原有积分，这能解释为什么右转误差仍然偏大。仅凭此记录还不能把全部偏差归因于摩擦。

然后读 `scheduled_command` 的时间缩放和 `YawRatePI.compute`。试着写出饱和时积分该不该继续的判断，再运行 `tests/test_d1_continuous_task.py`。测试包含一条完整 45 s 的真实 MuJoCo 回归，还会把转角记录改成零，确认“活到最后”不会单独满足任务质量判据。
