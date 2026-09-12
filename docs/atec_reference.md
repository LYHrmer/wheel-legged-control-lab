# ATEC 参考仓库如何用于当前 D1 项目

优先借鉴的是**视觉导航到运动命令的分层接口、分地形训练的组织方式，以及连续任务的证据记录**。电机动作策略和稳定器参数需要重新适配。本页是方案，没有导入 ATEC 权重、修改正式实验的 77 文件冻结集，或新增训练结果。

对照版本固定为 ATEC [`4f79d7f`](https://github.com/LYHrmer/atec-robotics-projects/commit/4f79d7f05507f45674e67abde06eddfe7c038a17) 与本项目 [`31eca03`](https://github.com/LYHrmer/wheel-legged-control-lab/commit/31eca0342c243a909ce62629ef22358247461979)。读取了 11 份 ATEC 源码/记录并核对 Git blob SHA；本地 7 份对照文件与指定 commit 逐字节一致。

当前 formal D1 策略输出腿伸长、轮速的残差，PD/PI 增益由配置固定；既有 LQR/VMC 分支输出机身合力残差。[本地动作映射](https://github.com/LYHrmer/wheel-legged-control-lab/blob/31eca0342c243a909ce62629ef22358247461979/src/wheel_legged_control/d1/wheel_leg_controller.py#L214) · [合力残差分支](https://github.com/LYHrmer/wheel-legged-control-lab/blob/31eca0342c243a909ce62629ef22358247461979/src/wheel_legged_control/d1/control_loop.py#L74)

| 接口 | ATEC Task A | 当前 D1 formal | 可移植边界 |
| --- | --- | --- | --- |
| 低层基座 | 冻结 D1 平地神经网络；G2 保持默认姿态 | 固定腿 PD、轮 PI 与姿态反馈 | 借鉴残差与基座分离、基座 SHA 绑定；控制律不同 |
| 策略输入 | 331 维：基座历史、本体信息、基座动作与上次残差 | 默认单帧 82 维控制上下文 | 不能截取数组凑维数；字段、归一化、历史时序需重新定义 |
| 策略输出 | 16 维残差叠到 23 维动作的前 16 项；最后 7 项为 G2 | `shared2` 或 `independent8`，最终映射到四腿伸长与四轮轮速 | 前者是 12 个关节位置与 4 个轮速接口，后者是 4 个腿长与 4 个轮速接口 |
| 动作尺度 | 残差先裁到 ±3，乘腿 1、轮 0.5；环境再乘腿 0.25 rad、轮 5 rad/s | 归一化动作裁到 ±1；腿长 ±0.04 m、轮速 ±4 rad/s，再受 IK、关节与目标限幅 | 上游尺度对应限幅前附加目标，不是 N·m 或实际关节位移 |
| 频率与机体 | 残差适配器要求 50 Hz；Isaac Lab D1+G2 | 默认 100 Hz 控制；MuJoCo D1 | 相同机器人名称不能消除执行器、惯性、接触和采样差异 |

上游依据：[残差状态与载入校验](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/task_a/tools/d1g2_taska_residual.py#L14)、[关节顺序与物理尺度](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/task_a/tools/d1g2_taska_env.py#L26)。本地依据：[策略到物理动作映射](https://github.com/LYHrmer/wheel-legged-control-lab/blob/31eca0342c243a909ce62629ef22358247461979/src/wheel_legged_control/d1/locomotion_env.py#L229)、[默认控制周期](https://github.com/LYHrmer/wheel-legged-control-lab/blob/31eca0342c243a909ce62629ef22358247461979/src/wheel_legged_control/d1/model.py#L240)。

最小导航接入可以沿用现有控制循环，只新增一个高层命令来源：

```text
RGB-D / 图像航向 + 本体角速度
  → 带仿真时间戳的视觉估计（坐标系、有效性、原因、XY、yaw）
  → 导航状态与失联处理
  → D1MotionCommand(vx, yaw_rate, clearance)
  → prepare(command) → 用 decision.context 推理 → step(action)
```

ATEC `VisualNavigator` 输出 `[前向速度, 0, 偏航速度]`，可将第一、三项送入 `D1MotionCommand`，离地高度先保持 0.455 m。其视觉锚点须先于同一时刻的本体更新消费，防止重复积分；独立航向更新不刷新 XY 的新鲜度。现有本项目控制循环已支持由不同来源提供运动命令，但这不表示已经具备 RGB-D 里程计。[上游视觉导航](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/task_a/tools/d1g2_taska_visual_navigation.py#L11) · [本地命令与逐拍顺序](https://github.com/LYHrmer/wheel-legged-control-lab/blob/31eca0342c243a909ce62629ef22358247461979/src/wheel_legged_control/d1/control_loop.py#L191)

拟议接口应拒绝倒序、重复和非有限视觉数据，明确初始坐标系与相机外参。失联时发站立命令并保留低层姿态闭环；上游 0.3 s 后减速、0.8 s 后停车可作为试验候选值，不能直接当成本机安全界限。暂停期间不推进仿真时间，恢复后重新检查视觉有效性；只有真正开始新回合才重置全程恢复预算。尚未实现的相机采集、标定、视觉里程计及其故障注入应作为明确交付，不能用仿真根位姿代替视觉输入。

课程训练最值得借鉴的是按地形类型记录真实暴露、按前进进度更新难度，以及单独保留困难地形诊断。ATEC 用 8 m 训练 tile、8 个难度行，课程晋级读取相对 tile 原点的前进位移；`upstairs`、`rough` profile 同时改变地形分布和奖励，因此其阶段成绩不能单独证明课程顺序的收益。[课程与专项配置](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/task_a/tools/d1g2_taska_train_env.py#L95)

本项目当前道路约束为名义纵坡 ≤3°、横坡 ≤2°、起伏与带斜边台阶 ≤1 cm；ATEC 训练配置中的 2–20 cm 台阶超出这一任务定义。应先用零残差控制验证拟加入地形的运动学、接触和终止条件，再设独立训练协议。不得通过放宽旧地形类限制，把新实验伪装成本轮 formal 的延续。[本地地形边界](https://github.com/LYHrmer/wheel-legged-control-lab/blob/31eca0342c243a909ce62629ef22358247461979/src/wheel_legged_control/d1/locomotion_terrain.py#L47)

后续对照设零残差、平地 PPO、随机混合地形 PPO、阶段课程 PPO。训练组对齐总转移预算、种子数、动作模式、网络、控制器、奖励及传感器；混合与课程组预先固定相同地形采样配额，并报告每类地形实际转移数、回合数和提前终止，避免名义配额掩盖曝光差异。先固定一种动作模式，再另做动作维数实验。本地早期课程结果并未显示明确优势，且缺随机混合组；这个缺口应补成对照，而非预设课程一定有效。[既有课程定义与限制](https://github.com/LYHrmer/wheel-legged-control-lab/blob/31eca0342c243a909ce62629ef22358247461979/docs/terrain_curriculum.md)

连续越障评测须预先固定赛道、起点、物理时限与终止规则，一回合跨越所有地形，首次终止即结束；评测中不重置姿态或移动地形。恢复应同时限制单次和全程盲行时间/距离，重锚不能清空预算。保存完整轨迹，联合报告终点到达、跌倒/非法接触、进度、跟踪误差、饱和与恢复次数；短暂存活、较低 RMSE 或训练课程等级都不能单独代表通过。按训练种子配对，不把每一控制步当独立样本；固定留出集不用于选择 checkpoint。ATEC 的 505.66 s、约 286 m 原赛道通过仅是一条本机完整记录，仓库还保留同配置失败，不能据此估计跨种子成功率。[通过记录及限定](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/task_a/evidence/acceptance_summary.json)

Task B 稳定器提供另一个有用负例：共同轮速/差速分解、变化率限制、倾角门控可以独立评估，但正常站姿 `sin(tilt)≈0.0633` 已超过初版 0.06 门限，造成持续误限速。参考站姿校准保留独立绝对倾角检查，这个设计原则值得借鉴；B2wPiper 的门限、腿运动符号和每 50 Hz 调用的步长不能照搬 D1，在真实坡面开始时还须防止把坡度吸收为“中性站姿”。本地应优先在运动命令层验证限速，腿级稳定动作另做 IK、支撑与接触试验。上游延长存活后仍转向很弱、Task B 0 分、没有完成物理抓取，不能将其称为通用稳定移动操作方案。[稳定器](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/task_b/stability.py#L1) · [参考站姿](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/task_b/stability_profiles.py#L1) · [实际实验限制](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/docs/TASK_B_EXPERIMENTS.md)

建议按以下三个阶段实施，每一阶段都有可独立判断的结果：

1. **命令接口与离线回放。** 完成视觉估计到 `D1MotionCommand` 的小适配器；测试时间戳、坐标轴、失联、暂停及回合重置，同一输入回放得到相同命令。此阶段不训练。
2. **零残差连续越障基线。** 完成相机与视觉的实际接入，先在固定低速范围测转向、站立和单段地形，再跑冻结的连续任务；交付全过程日志与失败案例。新地形或侧步接触方案需分别验收。
3. **独立训练对照。** 基线与感知接口通过后冻结新协议，实施平地/混合/课程实验并独立留出评测；本轮 56 指令 formal 结果保持原身份。

权重迁移需要新的任务定义与训练，不能直接 `load`：331 与 82 维观察、16 与 2/8 维动作、关节顺序、物理单位、历史、归一化、50/100 Hz、D1+G2 负载，以及 RSL-RL/基座 ONNX 与本地 SB3 都不一致。上游本身还校验基座 SHA 和动作语义；修改 sidecar 绕过本地校验不会消除这些差异。

本页由本地核验整理。确实请求过 Opus 主笔完整设计，但该次调用（`b87da832-7af3-42cd-9ab7-e534e6f83211`）以连接中断返回，输入/输出计数均为零，没有可采用的方案产物；未将其记为成功贡献。原始请求、响应及源码清单保存在持久工作目录 `wheel-legged-control-lab-work/recovery-20260912/atec_reference/`。
