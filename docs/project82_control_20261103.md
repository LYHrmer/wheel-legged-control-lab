# 82：延迟 MPC 被拒绝，重新核对估计器与轮速 PI 的证据

本轮新的延迟 MPC 开发对照 **12/12 完整，完整性门全部通过，原性能门失败**。候选不设为默认，完整阶段 `NOT_RUN`。估计器与轮速 PI 的新增工作是标准库独立读回，没有新增训练或重复旧物理实验；50 的训练中断仍未解决。跨仓工程适配也未取得 ROS2 或实机部署资格。

[机器摘要与来源清单](../reports/project82_results_20261103.json)汇总本轮结果；以下区分完整执行、性能判定和事后解释，不回写旧报告或放宽原门槛。

**新增延迟 MPC：实现成立，效果未通过。**

[预注册](experiments82/delay_mpc82_preregistered.json)只改变 MPC 内部对已知输入延迟队列的预测，保持原优化器、代价、权重和约束。入口为 [scripts/delay_mpc82.py](../scripts/delay_mpc82.py)，实现为 [delay_mpc82.py](../src/wheel_legged_control/delay_mpc82.py)。三个开发 seed 为 490101、490102、490103，均来自已经观察过的 49 开发集，不能称为新盲测。每个 seed 覆盖 nominal / mismatch_delay、baseline / candidate，共 12 场，每场 400 控制步。

实际执行 **4,800 控制步、48,000 显式积分子步、4,812 次求解**，最后一项包括 12 次初始求解。训练更新为 0。运行报告与结束后来源核验均为 `COMPLETE`，CPU 6.208706 s、任务墙钟 6.279219 s。六场原 49 基线轨迹复现、三场 nominal 候选与基线同值、延迟队列核对、原模型快照保留、指标独立重算全部通过；轨迹同值核对排除求解耗时字段，不声称整个文件字节相同。监督单元正常回收，记录的 memory high / max / oom / oom_kill 事件均为 0。

| mismatch_delay 指标 | 候选相对基线 | 原门 | 结果 |
|---|---:|---:|---|
| 高度 RMSE | +0.0466059% | 至少下降 10% | 失败 |
| 前向速度 RMSE | +7.6450344% | 最多增加 5% | 失败 |
| pitch RMSE | +16.5014581% | 最多增加 5% | 失败 |
| 最大绝对 pitch | +11.4633879% | 最多增加 5% | 失败 |
| 归一化 effort RMS | −1.4705521% | 最多增加 5% | 通过 |
| 轮力 RMS | +4.7224644% | 最多增加 5% | 通过 |
| 腿力 RMS | −1.9006340% | 最多增加 5% | 通过 |

nominal 六项跟踪/effort/force 护栏指标比均为 1（求解耗时不在此列）。候选最慢单场的求解 P95 为 0.481218 ms，满足原 20 ms 门；这是该 CPU 仿真求解测量，不是 ROS2 全链路截止期或优化器最优性的证明。[完整新原始工件包](assets/project82/delay_mpc82_20261103_01.zip)包含全部 12 场 CSV、模型快照、逐场指标、原预注册副本和报告。报告保留旧阶段文字 `eligible for independent full comparison` 作为来源字段，但正式判定是 `accepted=false / REJECTED`，不能据那段文字启动完整阶段。

[独立 MPC 读回](assets/project82/mpc_independent82.json.gz)又以标准库核对全部 4,800 行、14 项原性能门与全部原始工件 SHA；9,746 项数值比较通过。高度误差的事后分解显示，失配基线的均值偏差平方占均方误差的 93.0364%，末两秒平均伸缩位置为 −0.1200439 m。这描述了偏差占主导，尚未识别其原因。[场景](../src/wheel_legged_control/experiments.py)将 torso 质量从 14 kg 放大至 17.5 kg；相应重力为 171.675 N，而名义值为 137.34 N。[模型资产](../src/wheel_legged_control/assets/wheel_legged_planar.xml)的伸缩关节下限为 −0.12 m。这些静态事实与持续高度偏差相容，但未通过逐项反事实证实。不能据此宣称质量补偿、约束修改或增加积分一定有效；本轮未调整这些量。

**轮速 PI：先确认参考需求，已有抗积分饱和。**

[原连续近似读回](assets/project82/control_evidence82.json.gz)与[离散读回](assets/project82/pi_discrete82.json.gz)分别保留，不能混作同一个计算。后者核对 12 场全部 **36,000 行**，来源结束复核通过，新增模型调用、物理步和参数更新均为 0。其[标准库代码](assets/project82/pi_discrete82.py)按原 `implicitfast` 方程使用下一拍速度计算阻尼、当前拍速度计算摩擦：

`tau_required = M * (r_next - r_now) / dt + damping * r_next + friction(r_now)`。

固定 stress 参考的连续逆动力学峰值为 **2.895587377 Nm**，离散峰值为 **2.904486555 Nm**；两者均在每场 **1,512 / 3,000 = 50.4%** 的区间超过原 ±2 Nm 限制。离散读回还核对实际两步力矩延迟、相邻速度连续性及记录的动力学平衡。最后一拍 `r_next` 来自同一解析参考在 6.0 s 的值，不是续跑物理得到的新样本。重复参考及相同 plant 不能算作 12 个独立可行性发现。

这意味着该参考不能在现有力矩界内每个采样区间都精确跟踪，不提供 tracking RMSE 的下界，也未证明它是候选退化的唯一原因。原 [wheel_control.py](../src/wheel_legged_control/wheel_control.py)已经包含条件积分 anti-windup，不能把“补抗积分饱和”当作尚未实现的修复。49 中缩小辨识前馈虽然减少饱和，tracking RMSE 仍在 matched / mismatch 分别增加约 20.11% / 19.22%，原候选拒绝结论保持不变。

**估计器：49/50 的配对子集必须分别读。**

[本轮独立读回](assets/project82/estimator_crosscheck82.json.gz)及其[代码](assets/project82/estimator_crosscheck82.py)核对 49 的 12 场、50 的 18 场，共 **10,765 行**。49 的 none / constant_velocity，50 的 none / constant_velocity / measured_leg_q 三臂，均各 **4/6 完成、2/6 跌倒**。跌倒为已记录终态，不计作完成 400 步；所有失败轨迹保留。50 的 18 项单字段消融完整性核对与来源门均通过，性能门仍失败。

| 原比较 | 共同完整的原配对 | baseline → candidate 的平均单场 tracking RMSE（m/s） | 比值变化 |
|---|---|---:|---:|
| 49：none → CV | noise 17、29、43 | 0.258932191 → 0.302538474 | +16.840812% |
| 50：none → 去额外腿 PD 的 CV | noise 17、43；bias 43 | 0.267884154 → 0.317874984 | +18.661361% |
| 50：CV → 去额外腿 PD 的 CV，机制对照 | noise 17、43；bias 17 | 0.338443349 → 0.321965030 | −4.868856% |

最后一行既不是第二行的基线，也不是相同子集，不能抵消正式门失败。50 仅在完整执行原 CV 外推及 guard 后，将控制器消费的 12 个腿关节位置还原为测量值，移除外推引入的 `−Kp × h × qdot` 项（20 ms 时为 `−0.8 × qdot`，10 ms 时为 `−0.4 × qdot`；零龄或 guard 拒绝时不作外推）；其余速度、接触及部分时间字段未统一预测到当前时刻。结果否定了“仅去除此项即可修复这些开发场景”，没有证明其他时序耦合无关。完整矩阵未启动，见 [49 结果](control_candidates49_results_20261022.md)与 [50 结果](experiments50/estimator_results50.md)。

历史 144 场材料中的 sensor_estimation.py SHA 为 `433aa128…156d`，当前 49/50 绑定版本为 `3072d194…22d5`。旧“17/36 跌倒”等计数只能标为历史来源；本轮没有把它当作当前实现完整矩阵的复验。当前新增的 wheel-center Jacobian 路径为显式同步模式选项，49/50 仍核对其登记的 legacy 路径，不能将两种模式混用。

**训练与部署的未解决边界。**

50 的初始探索 σ 减半训练仍是 **0/6 完整、首基线 INCOMPLETE、其余 5 次 NOT_RUN**。祖先 `user@1000.service` 内存压力触发 systemd-oomd，首臂实际训练步数未知；不能把准入成功、预算预留或残留 `RUNNING` 报告当作完整训练。没有续跑、补 seed 或修改内存限额。原因、资源样本和终态证据见 [50 交付记录](optimization50_delivery_20261024.md)。

本轮没有重跑 48、没有重绑 `runtime/d1_b22/runtime_manifest_v27.json`、没有更换冻结策略或默认控制器，也没有用新训练结果排序。新 MPC 是独立入口；PI 与估计器是旧数据读回。仍待解决的是失配下高度静态偏差与动态响应各自的来源、估计器混合时刻状态的闭环影响，以及在既定内存纪律下稳定完成训练的问题。

D1 的接口与响应交叉核验见[另仓 82 结果](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/project82_crosscheck_20261103.md)。新增 adapter 只交付独立原生入口与合成故障验证；未接管当前 ROS2 控制路径，未验证 MIT PVT 底层限幅或实机停止行为。上述研究诊断、原生测试和仿真求解通过均不等于真实机器人控制效果改善。
