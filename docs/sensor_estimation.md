# 从 IMU、编码器到 D1 控制状态

这份实现把测量包和融合器分开。`D1MujocoSensorSource` 可以读仿真数据来生成传感器输出；`D1ProprioceptiveEstimator` 只接收测量包，自己持有一份名义模型和独立的 `MjData`。把实时机器人的位置改成别的值，再回放同一份测量，估计结果必须逐位相同。

代码在 [sensor_estimation.py](../src/wheel_legged_control/d1/sensor_estimation.py)，对应 [单元测试](../tests/test_d1_sensor_estimation.py) 和 [开发实验](../results/d1_sensor_estimation/README.md)。原来的完整真值加噪声通道仍保留，二者不能混称为同一种估计器。新通道标识是 `d1-imu-encoder-complementary-v1`。

## 先确定测到了什么

| 测量 | 单位、位置与坐标系 | 此处的假设 |
|---|---|---|
| 三轴角速度 | rad/s，轴与可见 `base_link` 一致 | 可含白噪声和固定偏置 |
| 三轴比力 | m/s²，测量点在基座惯性 COM，轴与 `base_link` 一致 | 当前是虚拟 IMU 安装位置，不能直接套到其他安装点 |
| 16 个关节角度和速度 | rad、rad/s | 速度来自电机编码器速度测量；未模拟差分测速器的内部滤波 |
| 4 路轮子着地标志 | bool | 每轮总法向载荷超过 1 N 的理想开关；载荷数值和法向不进入融合器 |

测量包没有基座位置或线速度，也没有地面高度、坡角。接触开关是额外传感假设。现有 D1 URDF 没有因此就多出四个真实载荷传感器。

机体碰地不在这些测量中，估计快照的 `undesired_ground_contacts` 固定为零，含义是没有这项观测，不能据此认定机体没有触地。仿真评测的跌倒保护仍单独读取真值，不能拿这个占位字段替代安全检测。

实际实现时发现，MuJoCo 的接触 margin 会让 `dist > 0` 的接触仍传递支撑力。开发样本中约 0.4–0.6 mm 间隙对应 35–85 N 支撑，所以不能把 `dist <= 0` 当着地开关。测试专门保留了这个案例。

## 比力和 COM 约定

令 `R` 将可见机体系向量转到世界系，世界重力 `g = [0, 0, -9.81]`。COM 处的加速度计模型是：

```text
f_m = Rᵀ (a_COM - g) + b_a + noise
gyro_m = omega_body + b_g + noise
a_COM_est = R_est f_m + g
```

自由落体的理想比力为零。直立静止并有支撑时，比力为 `[0, 0, 9.81]`。测试用独立的 1 kg 球体、9.81 N 支撑和实际 XML accelerometer 验证了这两点。

这里调用 `mj_rnePostConstraint` 后读取 `mj_objectAcceleration`，其线性输出已经符合上述比力语义，不再额外减一次重力。MuJoCo 没有相应传感器时，不会自动完成这部分后约束计算。参见 [MuJoCo 加速度 API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectacceleration) 和 [accelerometer 定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer)。

`D1StateEstimate` 有一个需要记住的历史约定：`base_position` 是可见基座原点 O，线速度却属于惯性 COM C。名义模型中的 `r_OC_body` 约为 `[0.02986616, 0.00014984, -0.01465674] m`，不能把二者当成同一点。

```text
v_COM_world = v_origin_world + omega_world × (R r_OC_body)
```

转动时漏掉这一项会产生确定性的速度误差。测试把角速度设成明确的 `[0, 0.2, 0] rad/s`，逐项核对接触点力臂和 COM 力臂，不靠“抬头为正”一类容易混淆的口头约定。

## 融合器做了哪些计算

姿态先按相邻两帧陀螺仪的平均值积分四元数。存在接触，且比力模长与 `9.81` 的差小于 `1.5 m/s²` 时，才加入加速度计的重力方向修正：

```text
up_measured_body = f_m / norm(f_m)
up_predicted_body = R_estᵀ [0, 0, 1]
omega_correction = cross(up_measured_body, up_predicted_body) / 1.0 s
```

这个门限不能分开持续加速度和重力。机器人加速时，姿态误差可能增加；绕重力方向的 yaw 也没有绝对校正。本版没有在线估计 IMU 偏置。

轮里程计使用估计姿态和编码器重建每条腿的运动学。对于着地轮 i，把接触点近似放在轮心沿支撑法向向下一个轮半径的位置，假设接触点不滑动：

```text
0 = v_origin + omega_world × (p_contact_i - p_origin) + J_joint_i qdot_i
v_COM_i = -omega_world × (p_contact_i - p_origin)
          - J_joint_i qdot_i
          + omega_world × (R r_OC_body)
```

`J_joint_i` 包含该腿的四个关节，轮子转动也在其中。若只用轮速乘半径，会漏掉摆腿速度和基座转动。多个着地轮给出的速度取平均；本版没有按打滑概率重新加权。

IMU 先预测 COM 速度，再向滚动约束速度做互补修正：

```text
v_pred = v_previous + (R_est f_m + g) dt
alpha = dt / (0.02 s + dt)
v_est = (1 - alpha) v_pred + alpha mean(v_COM_i)
```

无接触时保留惯导预测。位置通过 COM 速度积分，再减去旋转后的 COM 偏移，恢复可见原点位置。没有外部定位，因此水平位置和世界高度都会漂移。

## 地面参考从哪里来

至少三个不共线的着地轮心可拟合一个支撑面，法向朝世界上方。轮心平面向下移一个半径后，得到用于控制的地面平面。少于三个有效支撑点时，法向沿用上一帧；没有接触则不更新地面平面。`support_plane_observable` 和 `support_plane_status` 记录了这个分支。

这一几何近似要求轮轴大致平行于支撑面，未包含轮胎压缩或轮宽修正。起伏地形中，四个轮子之间的平均面不等于基座正下方的局部切面。两者的坡角误差在实验表里单列，不能只报机器人没跌倒。

`source.ground_reference(state)` 返回估计世界坐标中的高度和两个世界轴方向的坡角。控制代码再结合估计 yaw 转换姿态目标。控制器、策略观测和地面参考都使用这条估计通道；真实地形只能进入事后指标和安全终止判断。

初始位置与 RPY 必须显式给出，来源是已知放置先验。独立开发实验统一从 `(0, 0, 0.480) m`、零 RPY、零速度开始，所有组相同。不从碰撞穿透修正结果反查初态。连续任务在已知平地从 `(-3.8, 0, 0.455) m` 开始。未知初始姿态、未知初始高度和空中初始化不在当前验证范围内。

## 延迟与回放

`D1SensorStateSource(..., delay_steps=2)` 在融合前延迟纯测量包。在 100 Hz 控制下对应 20 ms。启动时保持初始测量，不偷偷向前仿真；`control_time_s` 是当前发布时刻，`measurement_time_s` 是实际消费的测量时刻。这里没有做延迟预测补偿。

实验保存每步状态误差 CSV，以及只含传感器字段的 `*_measurements.npz`。新建一个融合器回放这些包，就能复查估计；不需要原来的实时 `plant`。测试还会禁用基座真值读取和 MuJoCo 对象速度/加速度 API，确认融合器没有旁路查询。

重新跑实验时换一个输出目录，脚本拒绝覆盖已有结果：

```bash
env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 scripts/evaluate_d1_sensor_estimation.py \
  --output results/d1_sensor_estimation/my_repeat \
  --seeds 17 29 43 --delays 0 2
```

`offline` 中机器人由真值控制，融合器只旁听测量；`closed_loop` 才让估计值进入实际控制。当前独立脚本的残差动作固定为零。接入 PPO 的连续任务实验另行记录，不能把这份零残差结果写成策略表现。

## 可以自己检查的实验

先看自由落体测试。如果误把比力当成普通世界加速度，竖直速度会立刻出现一个重力量级的错误。再看 COM 力臂测试，尝试在本地副本去掉最后的叉乘项，观察是哪一条断言失败。

随后读取平地的 `pitch_error_rad` 与速度曲线。加速段姿态误差比稳态大时，检查加速度计门限是否启用，不要先改奖励。对照 0 和 20 ms 的 CSV，区分测量误差与信息变旧造成的跟踪变化。

最后看起伏上的 `estimate_ground_pitch_rad`。即使编码器无噪声，支撑面与局部切面也有模型差异。若以后加入视觉或激光地形感知，应在新的测量接口里增加外部信息，再与这份纯本体感知基线比较。
