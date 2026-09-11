# 文档导航

仓库里的文档地图，只回答四个问题：从哪开始、按什么顺序学、当前主线在哪、历史路线和证据在
哪。每份文档自己维护细节和数字，这里不复制它们的结论。

## 从哪开始

1. 按 [README](../README.md) 装好依赖，跑一个 12 s 零残差回合，先看得懂自己的输出目录。
2. 读[命令条件轮足实验](locomotion_lab.md)，对着那份 `telemetry.csv` 看命令、误差和一拍
   时序。
3. 公式和代码对不上的时候，回[学习指南](learning_guide.md)第 0 节挑一项基础练习补。

跑通命令不算学会。每项实验先写下预测，只改一个参数，再用原始 CSV 检查；与预测不符的结果
要保留，不要改门槛去迁就它。

## 学习顺序

| 顺序 | 文档 | 学到什么 | 前置 |
|---|---|---|---|
| 1 | [learning_guide.md](learning_guide.md) 第 1–3 节 | 六状态模型、线性化、LQR 与约束 MPC | 无 |
| 2 | [ppo_learning_lab.md](ppo_learning_lab.md) | 手算 TD/GAE，区分摔倒终止与时间上限 | 1 |
| 3 | [ppo_update_lab.md](ppo_update_lab.md) | 小型 actor/critic 上一次真实 SGD 更新 | 2 |
| 4 | [control_ppo_checks.md](control_ppo_checks.md) | 纸笔检查题，答案可以逐步核对 | 2 |
| 5 | [locomotion_lab.md](locomotion_lab.md) | 当前主线：82 维观测、八维动作、奖励单位 | 1–3 |
| 6 | [evaluation_protocol.md](evaluation_protocol.md) | 成功定义、配对统计、三类证据的分工 | 5 |
| 7 | [actuator_identification.md](actuator_identification.md)、[wheel_control_lab.md](wheel_control_lab.md) | 单转轴辨识台架，单轮 PI 与前馈对照 | 1 |
| 8 | [sensor_estimation.md](sensor_estimation.md) | 测量包与融合器分开的 IMU/编码器估计 | 5 |
| 9 | [contact_allocation.md](contact_allocation.md)、[inverse_dynamics.md](inverse_dynamics.md) | 约束接触力分配，整机任务空间 QP | 6 |
| 10 | [delay_learning_lab.md](delay_learning_lab.md) | 四帧观测、两种延迟、固定预算与失败配对 | 5、8 |
| 11 | [actuator_transfer.md](actuator_transfer.md) | 合成辨识进入整机通道，跟踪改善与运动收益的区别 | 7 |
| 12 | [encoder_identification.md](encoder_identification.md) | 只用位置辨识、可辨识性限制、四组单轮前馈对照 | 7 |
| 13 | [encoder_feedback.md](encoder_feedback.md) | 从带噪位置估速，再按采样时刻检查闭环误差 | 12 |
| 14 | [friction_delay_identification.md](friction_delay_identification.md) | 检查漏建模摩擦怎样改变延迟选择及前馈效果 | 12 |
| 15 | [shared_action_learning_lab.md](shared_action_learning_lab.md) | 同底座动作结构对照，二输出概率与八通道物理执行 | 2、5、6 |
| 16 | [budget_learning_lab.md](budget_learning_lab.md) | 四档存档逐种子作图，保留未完成回合对应缺口 | 15 |

第 7、9、12–14 行是独立支线，没接进主线控制器；第 8 行的融合器同时被主线的 `sensor` 状态源使用。

## 当前主线：82 维共用循环

训练、评测和键盘共用一个控制循环与一份 82 维观测。文档入口是
[locomotion_lab.md](locomotion_lab.md)，代码分布在

| 文件 | 负责什么 |
|---|---|
| [control_loop.py](../src/wheel_legged_control/d1/control_loop.py) | `prepare(command)` / `step(action)` 时序，同步采样 |
| [locomotion_env.py](../src/wheel_legged_control/d1/locomotion_env.py) | Gym 任务、随机化、终止判定、回合元数据 |
| [wheel_leg_controller.py](../src/wheel_legged_control/d1/wheel_leg_controller.py) | 腿 IK 查表、关节 PD、轮速 PI、偏航反馈 |
| [locomotion_observation.py](../src/wheel_legged_control/d1/locomotion_observation.py) | 82 维编码与缺失值 mask |
| [locomotion_rewards.py](../src/wheel_legged_control/d1/locomotion_rewards.py) | 奖励率乘控制周期，摔倒罚分为事件 |
| [locomotion_checkpoint.py](../src/wheel_legged_control/d1/locomotion_checkpoint.py) | sidecar 校验：schema、history、空间、增益、SHA-256 |

三个脚本入口：`run_d1_locomotion.py` 跑单次回合、键盘或命令回放；
`run_d1_locomotion_experiment.py` 做固定预算的 `benchmark / train / evaluate`；
`audit_d1_locomotion.py` 从记录独立复算。固定预算训练和留出评测：

```bash
python scripts/run_d1_locomotion_experiment.py train --seed 24000 --steps 32768 \
  --workers 4 --output results/my_locomotion_ppo

python scripts/run_d1_locomotion_experiment.py evaluate --split holdout \
  --policy results/my_locomotion_ppo/checkpoint.zip \
  --metadata results/my_locomotion_ppo/checkpoint.json \
  --output results/my_locomotion_holdout
```

默认 `--baseline wheel_leg`，两维力残差用 `--baseline lqr`，评测时的 baseline 必须和
checkpoint 一致。这两个模式需要 `[rl]` 可选依赖，其中包含用于记录内存的 `psutil`。
同底座的共享两维动作另用 `--baseline wheel_leg --action-mode shared2`，不是两维力残差；
对应地形套件与完整参数见[动作结构练习](shared_action_learning_lab.md)。下面的默认种子仍对应旧 `v1` 套件。
评测种子固定为开发 `17 / 29`、
留出 `617 / 629`，oracle 下同一路面的两个种子会产生逐字节相同的记录。

当前状态：固定预算的六模型对照已完成，结果在
[实验报告](../results/d1_v3_locomotion_report/README.md)；20 ms 整包测量延迟下融合状态仍会
失稳，见[延迟诊断](../results/d1_v3_delay_diagnosis/README.md)。后续
[九模型历史/随机化对照](../results/d1_v3_delay_ablation/README.md)已完成，未得到稳定的延迟收益。
新增[同底座动作对照](../results/d1_shared_action_study/README.md)共用轮腿低层，
分别训练共享两维与独立八维策略；不能与上面的旧两维力残差结果混为一个实验。

[预算阶梯学习页](budget_learning_lab.md)接着检查连续训练的四档存档，包含
[绘图入口](../scripts/plot_d1_budget_study.py)及计数表的读法。当前 2900 项测试通过只说明
实现验收通过，正式学习曲线尚未完成，不能提前写成增加预算带来了性能提升。
[最终短流程记录](../results/d1_budget_smoke_final/README.md)已完成存档重载和独立分析；
0.2 s 回合没有进入非平地任务，仅用作接口检查。

## 历史路线与 QP 原型（LEGACY）

按观测维数分组。checkpoint 不跨组兼容，入口和评测口径也不同，它们的数字不并入主线成绩。

| 路线 | 观测与动作 | 文档 | 入口 |
|---|---|---|---|
| 42 维残差 PPO 与键盘课程 | 42 维观测、两维力残差 | [interactive_course.md](interactive_course.md)、[state_estimation.md](state_estimation.md)、[contact_allocation.md](contact_allocation.md)、[contact_allocation_development.md](contact_allocation_development.md)、[learning_guide.md](learning_guide.md) 第 8–13 节 | `wheel-legged-d1-play`、`wheel-legged-d1-benchmark`、`wheel-legged-d1-contact-audit` |
| 44 维地形课程与跟踪 v2 | 44 维，含 clearance 误差与地面参考 | [terrain_curriculum.md](terrain_curriculum.md)、[terrain_tracking_v2.md](terrain_tracking_v2.md)、[reference_dynamics.md](reference_dynamics.md)、[action_parameterization.md](action_parameterization.md) | `scripts/run_d1_terrain_curriculum.py`、`scripts/run_d1_action_ablation.py` |
| 45 维连续任务 | 44 维布局末尾加偏航命令 | [continuous_task.md](continuous_task.md)、[control_rl_experiments.md](control_rl_experiments.md) | `scripts/train_d1_continuous_policy.py`、`scripts/run_d1_continuous_task.py` |
| 任务空间逆动力学 QP | 无 RL，直接解 16 路力矩 | [inverse_dynamics.md](inverse_dynamics.md) | `scripts/evaluate_d1_inverse_dynamics.py`、`scripts/benchmark_d1_inverse_dynamics.py` |

受保护跳跃演示只属于第一行的旧键盘入口，新键盘接口没有跳跃，也不要把旧 GIF 当成主线能力。
QP 原型没有接 PPO，也没有取代默认控制器。模型来源与许可证核对在
[d1_model_card.md](d1_model_card.md)。

## 证据怎么找

主线脚本要求每次运行写一个新目录，不覆盖旧结果；部分历史脚本仍会覆盖同名输出，重跑前须换目录。
主线单次回合目录里有 `protocol.json`、
`model.mjb`、`telemetry.csv`、`states.npz`、`summary.json`、`manifest.json` 和源码归档；
实验目录另有 `training_samples.csv`、`updates/`、`checkpoint.zip` 与同名 sidecar、
`training.json` 或 `evaluation.json`，失败时留下 `failure.json`。

主线证据的位置：

- [实验报告与录像索引](../results/d1_v3_locomotion_report/README.md)
- 训练目录 `results/d1_v3_locomotion_ppo/`，协议 `results/d1_v3_locomotion_protocol.json`
- 录像 `results/d1_v3_locomotion_recordings/`（两段 60 s，道路不同，不是逐帧对照）
- 零残差转向探针 [`results/d1_v3_action_probes/README.md`](../results/d1_v3_action_probes/README.md)
- 延迟诊断 `results/d1_v3_delay_diagnosis/`
- [执行器辨识移植与补偿](actuator_transfer.md)：合成台架到整机轮力矩通道
- [延迟与历史观测练习](delay_learning_lab.md)：三组固定预算对照、单位与归因边界
- [原始数据下载与复算](reproducibility.md)：Git 中的摘要与 Release 中的大体积数据
- [预算阶梯的运行与读图](budget_learning_lab.md#先跑短流程)：新目录运行、独立复算后再绘图

单轮支线的记录见[编码器速度反馈](../results/encoder_feedback_position_only/README.md)和
[摩擦／延迟辨识](../results/friction_delay_cross_study/README.md)。后者的控制仍使用独立速度
测量，不能与前者的位置估速反馈混为一组结果。

复算一份已有记录，不导入仿真器：

```bash
python scripts/audit_d1_locomotion.py --matrix-root results \
  --output results/my_arithmetic_check.json
```

读结果时按 [evaluation_protocol.md](evaluation_protocol.md) 的分工：pytest 守接口和验收
条件，视频只用来看运动是否符合直觉，多种子结果才用来谈鲁棒性。曝光标签是几何判据，哈希
一致只说明字节一致。已经报告过的失败记录保留在原目录里。
