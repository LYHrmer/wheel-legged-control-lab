# D1 评测协议

这份协议约束仓库中的 D1 固定场景、随机域审计、地形课程和状态延迟实验。视频用于检查运动
是否符合直觉，pytest 用于守住接口与验收条件，多种子结果才用于讨论鲁棒性。三类证据不能
互相替代。

当前控制周期为 `10 ms`，默认回合为 `6 s`。下文的状态年龄、动作延迟和回合时长均按实际
控制步计算，不用配置值代替日志值。

每条 D1 结果都要记录 `contact_allocation`。带 PPO 的运行还要核对 checkpoint 中的
`baseline`、状态模式、延迟补偿和接触分配模式；任一字段不匹配时中止，不生成对比结果。

## 场景

固定场景共用同一条速度和高度命令：前 `0.75 s` 静止，随后依次执行前进、后退和低速前进。
它们只检查指定扰动下的闭环行为，不代表域外泛化。

| 场景 | 域参数与外力 | 延迟和噪声 |
|---|---|---|
| `nominal` | 名义质量、阻尼、摩擦和执行器强度，无外力 | 默认均为 0 |
| `push` | `2.6 s` 时施加 `+140 N × 0.12 s` 世界系水平力 | 默认均为 0 |
| `mismatch_delay` | 质量 `1.12`、阻尼 `1.25`、摩擦 `0.65`、执行器强度 `0.85`；`3.6 s` 时施加 `-130 N × 0.12 s` | 动作延迟 3 步、状态延迟 3 步、噪声比例 `1.0` |

`mismatch_delay` 的含义受状态模式约束。`oracle` 下，状态延迟不会进入经典控制，噪声只作用于
PPO 观察；`estimated` 下，状态延迟和字段噪声会进入 VMC、LQR/MPC、PPO、安全逻辑与地形
估计。结果表必须写明 `state_mode`，不能把两种模式放在同一列比较。

随机域审计使用 `training` 场景。每个回合独立采样：

| 参数 | 范围 |
|---|---:|
| 基座质量比例 | `0.90–1.12` |
| 关节阻尼比例 | `0.80–1.25` |
| 地面摩擦比例 | `0.65–1.30` |
| 执行器强度比例 | `0.85–1.05` |
| 动作延迟 | `0–3` 个控制步 |
| 状态延迟（仅 `estimated`） | `0–3` 个控制步 |
| 初始 Roll / Pitch / Yaw | `±0.035 / ±0.045 / ±0.05 rad` |
| 腿关节初始偏差 | 每关节 `±0.025 rad` |
| 基座初始高度偏差 | `-8…+12 mm` |
| 速度命令 | `-0.75…+0.75 m/s`，每 `1.5 s` 重采样 |
| 高度命令 | `0.445–0.475 m`，每 `1.5 s` 重采样 |
| 随机推力 | `90–170 N`，正负方向随机，持续 `80–140 ms` |

接触分配审计固定为 LQR、oracle、无补偿，并把动作延迟、状态延迟和传感器噪声设为零；质量、
阻尼、摩擦、执行器强度、命令和推力仍按随机域采样。legacy 与 constrained 会同时切换分配器、
匹配辨识模型和固定 `R`，因此比较对象是两套模式匹配闭环配置。constrained 的 `R=1e-6` 只用
开发 seed 21 的 nominal/push 选择，预留 seed 121–150 不参与调参。

状态延迟灵敏度实验仍使用随机域，但把动作延迟和字段噪声固定为 0，只扫描
`0/10/20/30/50 ms` 状态延迟。`none` 与 `constant_velocity` 分开运行；两次运行必须使用相同
起始种子和回合数。

地形课程是另一组确定性探针。六个区域的时长、目标速度和最低进度直接写在
[`run_scripted_demo()`](../src/wheel_legged_control/d1/interactive.py) 中：起点、乱石、坡道、台阶、
波浪路和跳跃道的最低进度分别为 `1.0/2.0/4.0/3.5/3.5/1.7 m`。跳跃道还要求至少完成一次
四轮离地的跳跃，并让基座越过第一根横杆末端 `0.30 m`。`start` 固定运行 `7 s`，其余区域
为 `8–14 s`；课程结果不替代随机域审计。

## 成功、摔倒和连续指标

仿真出现任一条件即判为摔倒并终止回合：

- `qpos` 含非有限值；
- 基座高度低于 `0.22 m`；
- `|Roll| > 0.85 rad` 或 `|Pitch| > 0.85 rad`。

固定场景和随机域回合只有运行到 `6 s` 时间上限且未摔倒才记为成功。这里的成功是“生存到
回合结束”，不包含速度、姿态或能耗门槛；这些量单独报告。地形课程的成功还要满足对应区域
的最低进度，跳跃道另有跳跃和横杆条件。

`success` 与 `episode_duration_s` 是鲁棒性的主指标。速度 RMSE、Pitch RMSE、高度 RMSE、功率、
饱和比例和接触比例只对实际记录到的轨迹计算。若机器人在 `2.1 s` 摔倒，这些连续指标只描述
摔倒前 `2.1 s`，不会补齐为 `6 s`。早摔回合的 RMSE 可能反而更小，因此不能脱离成功率与回合
时长解释。CSV 必须保留失败回合，不允许只汇总成功样本。

接触分配把求解器结果和 wrench 跟踪结果分开记录。前者统计 `converged`、
`feasible_nonconverged`、fallback 与无接触比例；后者按工程容差统计 `tracked/limited`。
力容差是 `1 N + 0.5% × ||F_d||`，力矩容差是 `0.5 N·m + 0.5% × ||M_d||`。另外报告分配器
端到端 P99 和归一化最大约束违反量。wrench 误差分为 requested→allocated、
allocated→MuJoCo 和 requested→MuJoCo 三段；每段都把力与力矩拆开，分别使用 N 与 N·m，
不能合成一个六维欧氏范数。

MuJoCo actual wrench 从 `mj_contactForce` 汇总轮地约束力，统一换算为 ground-on-robot 的
世界系 wrench，力矩参考点为本周期控制器记录的 `base_link` 原点。每个 `10 ms` 控制周期
平均 `5` 个 `2 ms` 物理子步。它只进入离线评估，不包括非轮部位触地、外加推力或传感器
噪声，也不反馈给控制器。

## truth、raw 与 control

一次 `estimated` 控制步的数据流为：

```text
MuJoCo truth
  ├─> 奖励、摔倒判断、最终评测
  └─> 带噪延迟状态源 ─> raw_estimated_state
                              └─> 可选短时外推 ─> control_state
                                                        └─> VMC / LQR / MPC / PPO / safety / terrain
```

`truth_state` 是 MuJoCo 当前真值。`raw_estimated_state` 是加入字段噪声并经过延迟队列的原始
快照。`control_state` 是本周期真正送入控制链的状态；关闭补偿时，它与 raw 相同。兼容字段
`state` 仍指向 truth，`estimated_state` 暂时指向 control，新代码不应依赖这两个容易混淆的
名字。

`estimated` 模式下，控制器和策略观察不得读取当前 truth。真值只用于监督奖励、终止条件和
离线指标。`oracle` 是显式声明的真值上限实验，此时控制状态本来就来自 MuJoCo；它不能被写成
传感器闭环结果。原始估计误差和补偿后误差应分别由 raw/truth 与 control/truth 计算。

## 噪声比例

`sensor_noise=1.0` 使用
[`make_default_d1_estimator_impairments()`](../src/wheel_legged_control/d1/state_estimation.py)
中的下列参数。连续噪声为各分量独立的零均值高斯噪声；比例 `s` 将表中标准差乘以 `s`。

| 字段 | `scale=1.0` | 说明 |
|---|---:|---|
| 基座位置 | `0.002 m` | 世界系每轴标准差 |
| 基座旋转 | `0.004 rad` | 右乘旋转向量每轴标准差 |
| 基座线速度 | `0.020 m/s` | 机体系每轴标准差，世界系量由扰动后姿态重算 |
| 基座角速度 | `0.010 rad/s` | 机体系每轴标准差，世界系量由扰动后姿态重算 |
| 关节位置 | `0.001 rad` | 16 个关节各自扰动 |
| 关节速度 | `0.010 rad/s` | 16 个关节各自扰动 |
| 足端位置 | `0.001 m` | 每个足端的世界系三轴 |
| 足端平移 Jacobian | `0.001 m/rad` | 每个矩阵元素 |
| 接触点 | `0.002 m` | 有接触时的世界系三轴 |
| 接触法向 | `0.004 rad` | 对单位法向施加小旋转后重新归一化 |
| 接触点 Jacobian | `0.001 m/rad` | 每个矩阵元素 |
| 轮地接触翻转概率 | `0.005` | 每个轮、每次发布独立判断 |

非轮部位触地计数当前不加噪。接触被翻为 false 时，对应接触点清零；被翻为 true 时，用该轮
轮心沿合成法向下移一个轮半径生成接触点，并在该点重新计算 Jacobian。这个通道用于可复现
的灵敏度实验，不是 IMU 与编码器估计器。

## 延迟队列和短时补偿

状态源在 reset 时用同一份初始测量填满延迟队列，因此 reset 返回的状态年龄为 0。此后第 `k`
个控制步的年龄为 `min(k, delay_steps) × 10 ms`；经过 `delay_steps` 个控制步后才达到稳态年龄。
延迟实验会逐步核对整条 `state_age_ms` 轨迹，而不只检查配置字段。

`constant_velocity` 仅外推连续量。接触开关、接触点、法向、接触点 Jacobian 和非轮触地
计数仍使用延迟测量。constrained 分配器会用这组旧时刻接触几何计算约束。满足以下条件时
才应用外推：

- 状态年龄不超过 `50 ms`；恰好 `50 ms` 仍允许；
- 任一腿关节的 `|qdot × age|` 不超过 `0.35 rad`；
- 预测腿关节位置不越过关节限位。

超过时间边界时状态为 `horizon_exceeded`；关节位移或限位检查失败时为
`kinematic_horizon_exceeded`。两种情况都退回 raw 快照，并把实际外推时长记为 0。轮关节不
参与 `0.35 rad` 与位置限位检查。外推成功后仍保留原测量时间，`state_age_ms` 不会被改成 0。
扫描结果同时报告补偿应用比例和拒绝比例，避免把“请求了补偿”误写成“每一步都完成了外推”。

## 随机种子与配对

传入基础种子 `seed` 和 `N` 个审计回合后，评测种子固定为
`seed+100, …, seed+100+N-1`。同一评测种子必须在所有控制器和延迟点上复用。状态延迟扫描还
会逐回合核对：

- 域参数、动作延迟、噪声比例和 estimator seed；
- 初始 truth 状态与初始命令的 SHA-256 指纹；
- reset 时已经采样好的完整推力时序指纹。

指纹检查包含计划中的推力，而不只比较摔倒前已经施加的力。这样可以避免两个早摔回合因为
推力前缀都为 0 而被误判为输入匹配。

接触分配审计使用单独的预注册种子规则。默认 `seed=21, episodes=3` 直接运行 21–23，只作
开发检查。正式运行必须恰好使用 121–150，即 `seed=121, episodes=30`。其他起点即使运行
30 回合也保持 exploratory，manifest 不会给出 promotion pass。

`none` 和 `constant_velocity` 当前由两次 CLI 调用分别生成。程序只保证单次运行内部配对；
跨目录对照还要核对两份 manifest 的种子列表、协议字段和逐回合输入指纹。缺少任一 seed、存在
重复 seed 或配对条件不一致时，不生成差异结论。

## 统计方法

多种子报告使用 95% 区间：

- 绝对成功率使用 Wilson score 区间；
- 相对 `0 ms` 的成功率差先在同一 seed 内作差，再做 `20,000` 次配对 percentile bootstrap，
  bootstrap 随机种子固定为 `20260830`；
- 连续指标的绝对均值使用双侧 Student t 区间；
- 连续指标相对 `0 ms` 的差使用同 seed 配对 t 区间；
- PPO 与 LQR 的连续指标差也按 evaluation seed 配对后使用 t 区间。

接触分配审计同样报告配对 t 区间，但当前晋级门使用预先固定的点估计阈值：成功率不劣于
legacy、物理侧力/力矩跟踪误差各降低至少 10%、fallback 平均比例不超过 1%、任一回合的
`feasible_nonconverged + fallback` 不超过 1%、约束违反不超过 `1e-6`，且任一回合的分配器
P99 不超过 `10 ms`。把可行候选继续用于控制不会抹掉求解器未收敛记录。这些门槛是工程选择，
不应写成统计学非劣效证明。

少于两个配对样本时区间写为不可用，CSV 中保存 `NaN`。每个条件少于 20 个回合时，报告标记
为 exploratory。当前没有做多重比较校正；单个次要指标的区间不用于宣称控制器整体更优。

## Provenance 和产物

本机基准环境为 Intel Core i7-14650HX、24 个逻辑 CPU、15 GiB RAM。当前 NVIDIA 驱动不可
用，训练和评测按 CPU 执行。控制步耗时受并发负载影响，报告 P95 时必须同时保留该运行的
配置，不能把一次本机数值写成硬件无关结论。

一次正式状态延迟运行至少生成：

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | schema、run ID、协议、种子、控制器、状态模式和补偿模式 |
| `delay_sweep_episodes.csv` | 每个 controller × delay × seed 的原始记录，包含失败回合 |
| `delay_sweep_summary.csv` | 点估计、95% 区间、相对 0 ms 的配对差和统计方法 |
| `state_delay_sensitivity.md/.png` | 可读表格和带误差线的四联图 |
| `delay_sweep_manifest.json` | 完成状态、输入配对检查、记录数和所有产物 SHA-256 |

接触分配审计生成 `evaluation_config.json`、逐回合 CSV、汇总 CSV、Markdown 报告、PNG 图和
`contact_allocation_manifest.json`。manifest 最后原子写入，并记录每对输入指纹、seed pool、
晋级检查和其余五个产物的 SHA-256。晋级门还要求每个控制步恰好记录 `5` 个物理子步 wrench
样本。

普通固定场景与随机域基准使用同一规则。`benchmark_manifest.json` 在最后一步原子写入；程序会
从最终 CSV 回读行数，核对控制器间的初态、命令和推力指纹，再记录配置、CSV、图像及可选
GIF 的 SHA-256。运行中断时目录里不会保留旧的 `complete` 标记。

源码 provenance 在运行开始时记录 Git commit、dirty 状态和 dirty worktree SHA-256。加载策略时
另存 checkpoint 路径、选择方式和模型 SHA-256。正式 README 数字应来自 clean commit；dirty
运行可以用于调试，但要保留指纹并标成 exploratory。新训练必须把 Python、依赖版本、设备、
实际步数和模型哈希写入同目录 `training_config.json`；仓库里的旧 oracle checkpoint 缺少其中
一部分字段，缺省按 `oracle / none / legacy` 兼容，不作为新实验的元数据范本。

原始延迟与补偿结果使用固定目录 `results/d1_state_delay_raw` 和
`results/d1_state_delay_compensated`。重新生成时覆盖这些当前产物，旧版本由 Git 历史保存，
不创建 `final_v2` 一类副本。

## 主张与证据

| 主张 | 测试 | 证据文件 | 当前结论 |
|---|---|---|---|
| D1 模型能以 `23 nq / 22 nv / 16 nu` 运行，执行器与关节顺序一致 | 模型维度、关节、接触和静止闭环测试 | [`test_d1_model.py`](../tests/test_d1_model.py)、[`d1_model_card.md`](d1_model_card.md) | 只在 MuJoCo 整机仿真中验证；没有实机参数对照 |
| 一次控制周期内所有控制模块读取同一份状态快照 | 快照不可变性、整回路状态延迟与环境 info 测试 | [`test_d1_state_estimation.py`](../tests/test_d1_state_estimation.py)、[`test_d1_env.py`](../tests/test_d1_env.py) | oracle 与带噪延迟状态源可切换；estimated 仍是人为误差通道 |
| constrained 分配遵守接触、摩擦和执行器约束 | 理想接触、接触丢失、低摩擦、冲突力矩与 fallback 测试 | [`test_d1_contact_allocation.py`](../tests/test_d1_contact_allocation.py)、[30 组留出审计](../results/d1_contact_allocation/contact_allocation_audit.md) | 正式最大违反比例 `1.41e-9`，但 9 回合可行未收敛比例超过 1%，晋级未通过；实际 wrench 跟踪改善伴随姿态误差增大 |
| MuJoCo actual wrench 与分配器预测可以独立比较 | 静止重力、参考点平移、不可变性和五子步平均测试 | [`test_d1_model.py`](../tests/test_d1_model.py)、[`test_d1_env.py`](../tests/test_d1_env.py) | 只证明仿真内测量链路；平均单接触点模型仍会丢失接触偶矩 |
| LQR/MPC/PPO 能完成当前固定平地场景 | 单种子 nominal、push、mismatch 回放 | [`metrics.csv`](../results/d1_benchmark/metrics.csv)、[`metrics.md`](../results/d1_benchmark/metrics.md)、[`benchmark_manifest.json`](../results/d1_benchmark/benchmark_manifest.json) | 当前提交的 legacy 三种控制器通过固定回放；这是回归结果，不是域外鲁棒性证据 |
| 已提交 PPO 稳定优于 LQR | 30 个匹配随机域 seed 的连续指标配对区间 | [`randomized_audit.csv`](../results/d1_benchmark/randomized_audit.csv)、[`randomized_audit.md`](../results/d1_benchmark/randomized_audit.md) | 三个误差区间均跨 0，平均奖励还显著降低；当前证据不支持“稳定优于” |
| LQR 能通过当前物理地形课程 | 六区域脚本探针与对应 pytest | [`course_metrics.csv`](../results/d1_interactive/course_metrics.csv)、[`test_d1_interactive.py`](../tests/test_d1_interactive.py) | oracle/LQR 当前为 6/6；验收要求第一根横杆，本次终点超过前两根，`60 mm` 横杆、estimated 与实机均未验证 |
| 常速度外推改善状态延迟鲁棒性 | raw 与 compensated 的同 seed `0/10/20/30/50 ms` 扫描 | [`raw 结果`](../results/d1_state_delay_raw/state_delay_sensitivity.md)、[`补偿结果`](../results/d1_state_delay_compensated/state_delay_sensitivity.md)、[`test_d1_experiments.py`](../tests/test_d1_experiments.py) | `10 ms` 下减小部分误差；`20 ms` 成功数提高但配对区间跨 0，`30/50 ms` 均全部失败，不支持延迟裕量已改善 |
| 常速度外推遵守短时运动学边界 | `0/50/>50 ms`、腿关节位移和限位边界测试 | [`test_d1_state_estimation.py`](../tests/test_d1_state_estimation.py) | 边界已有测试；方法仍不预测接触切换 |
| 仓库已经实现可上实机的状态估计 | 无 | [`state_estimation.md`](state_estimation.md) | 未实现，不作该主张 |
