# 50：控制消融被拒绝，训练因系统内存压力中断

本轮完成了新的控制消融、独立来源审计和真实模型初始化准入；正式六臂训练在第一个基线 worker 阶段被 systemd-oomd 终止。**没有新增合格控制候选或完整训练模型，也没有新的性能提升结论。** 原四段无 UI 演示及 49 的全部结果保留，未更换默认控制器或策略。

执行源码提交为 `24c55bd8de778c7cfac49e32bc2d4f197592f829`。本页及外部终态报告为运行后的归档，未修改执行过的脚本、测试、预注册或旧来源清单。训练原 `report.json` 的残留 `RUNNING` 状态保持原字节；[独立终态报告](../reports/training50_results_20261024.json)依据系统日志记录实际中断，不能用它冒充完整训练回执。

| 工作 | 实际结果 | 判定 |
|---|---|---|
| 12 个腿关节位置的 PD 输入消融 | 离线 36 例；闭环 18 场全部到终态，三臂各完成任务 4/6、跌倒 2 场 | 相对新鲜 none 的共同完整配对 RMSE **+18.66%**，原门最多允许 +5%，拒绝；144 场未启动 |
| 独立来源核验 | **4,072 项 / 4,053,653,469 bytes**；928 个仓内冻结条目匹配 | 仅接受事前登记的 libXtst 单项差异；v27 未重绑 |
| 真实初始化准入 | **2 次 PPO.load、6 次 torch.load、2 次 predict / 64 行**；两份初态证明独立读回 | 通过；训练、优化器更新、物理步数均为 0 |
| σ 减半的正式训练 | **1 次 attempt 已消耗、0/6 完整、5 次 NOT_RUN** | 首基线不完整；其实际训练、模型和 native 计数未知 |
| 六模型导出及任务评测 | **0 次导出、0 个完整模型评测、0 新评测控制步** | NOT_RUN；均值、极差、性能资格均不可判定 |

## 控制结果

唯一变量是：完整执行原 constant-velocity 外推和所有 guard 后，仅将控制器消费的 12 个腿关节位置改回测量值。20 ms 外推在冻结 PD 中引入的额外项为 `−0.8 × qdot`；独立读回确认新候选确实移除了这个项。姿态、轮角、速度、接触、预测几何和时间字段仍按原路径处理。

相对新鲜 none，三个共同完整配对的 tracking RMSE 从 **0.267884154 增至 0.317874984 m/s**，完成数没有增加。原 CV 与候选在另一组共同完整配对中降低 4.87%，但配对子集不同，不能替代正式基线或解释成整体改善。实际新增 **6,495 控制步 / 32,475 显式积分子步**；6 场跌倒都有完整终态，partial 为 0。独立读回核对 185 项来源、182 项产物、6,495 条控制记录和 6,513 个含初态状态，未构造模型或推进物理。

全部逐场数字、门槛、混合时间字段边界与失败轨迹见[控制结果](experiments50/estimator_results50.md)。这说明移除额外腿 PD 项不足以修复这些场景的闭环退化，不能推出 PD 耦合完全无关。失败后未扫描参数、未运行完整矩阵，也未把失败轨迹制作成成功演示。

## 训练问题与已完成准入

[预注册](b22_sigma50_preregistration.md)只问：初始探索标准差减半，能否缩小固定开发任务上的三个训练 seed 的 R 极差，同时不恶化均值。预指定 PPO seeds 为 201051、201061、201071，每个 seed 各一个基线与候选，共六次各 32,768 控制步。课程、奖励、均值网络、critic、Adam、原 PPO 与 CPU 单环境协议保持不变。

新增入口先用原 loader 核验完整 B22 与 Adam，再对候选的 16 维 float32 log_std 加一次 float32 `ln(0.5)`，之后按原算法学习。真实模型准入验证其余 policy/critic/Adam/RNG 不变，确定性父探针逐位相同；模型进程退出后，由不导入 Torch 的独立 reader 核验两份初始化证明。准入模型已销毁，不能当作六次训练的 final。

来源审计允许的唯一系统差异为 libXtst：`0c81be070f7857bdcab6bd528161d1049c374f1237aabf76d92e097b2d8ba16a` → `4198b32dc9fba7dfc0ee2bb1b6bc85067c89adfc68d0279059ad82c207cc3c35`。旧 v27 的 SHA 始终为 `d3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09`。新合同显式保留这个差异，没有冒称当前环境满足旧 v27，也没有调用或恢复 48 的一次性实验。

## 正式任务的中断证据

单元 `control50-wheel-train-20261024-01.service` 在本地时间 **01:09:25–01:11:27** 运行，整个 prepare/bootstrap/worker 单元耗时 **122.018 秒**、CPU **41.727 秒**。首基线 session 直到 **01:10:47.172847** 才创建，因此不能称为“训练了 122 秒”。

[系统日志](../reports/training50_terminal_20261024_01/oomd_journal.txt)明确记录：祖先 `user@1000.service` 的 memory pressure 为 **72.36% > 50%**，伴随 reclaim 持续超过 20 秒；systemd-oomd 杀死该任务单元的 6 个进程，主进程 `SIGKILL`。这与内核 cgroup `oom_kill=0` 不矛盾，不能将后者解释成未被系统杀死。

118 个资源样本中，最低有效可用内存 **8,150,683,648 bytes（7.5909 GiB）**，记录到的单元峰值 **2,148,007,936 bytes（2.000488 GiB）**。MemoryHigh 2 GiB、MemoryMax 3 GiB、swap 0 保持不变；最后样本 `high=10491`、`max/oom/oom_kill=0`。这不是触及 3 GiB 硬上限，也没有触及 2 GiB 可用内存保留线。Nice10、单核/库线程 1 的启动设置未改变。

这些是终止前的样本，不是缺失的终态资源回执。来源扫描页缓存、依赖导入工作集与祖先内其他负载对 reclaim 的各自贡献尚未分离；没有通过重跑证明具体内存根因，也没有修改 oomd、放宽资源限额或停止其他程序。

| 阶段 | 最终分类 | 计数解释 |
|---|---|---|
| bootstrap | COMPLETE | 已封存模型调用、控制步、normal/compiler-native 均为 0 |
| baseline_s51 | INCOMPLETE / 外部 SIGKILL | attempt 已预留、不可退款；实际 controls/native/model/pure-servo 为 unknown |
| candidate_s51、baseline_s61、candidate_s61、baseline_s71、candidate_s71 | NOT_RUN | 没有阶段目录、reservation 或 worker 启动记录 |
| 全部 final 导出、物理评测、评测读回 | NOT_RUN | 没有合格 final 作为输入，不评估 partial 或借旧模型补位 |

baseline 仅留下 session、adapter proof 和 PID 三个文件，没有 worker 终态、初态证明、训练数据块、完整 checkpoint 或训练读回。原父报告的 `known_actual_counts=0` 只累计了 bootstrap，**正式训练总实际消耗未知，不能写成 0**。首臂预留的 32,768 controls / 163,840 normal-native / 2 compiler-native 也只是不可退款预算，不代表全部实际执行。

整组训练按预注册关闭，不续跑、不重试、不补 seed。原训练报告、资源 JSONL、reservation、系统日志的 SHA 和副本见[终态证据目录](../reports/training50_terminal_20261024_01/)与[机器可读报告](../reports/training50_results_20261024.json)。归档时核实原单元 cgroup 和记录的 worker PID 均已不存在。执行结束的完整 4.05 GB 来源复核未完成；归档只重新核对已登记 authored 文件、release 和 v27 身份，不将它说成完整闭包复验。

## 验证与交付边界

新增针对性验证共 **63 项通过**：来源 10、控制 12、训练/读回/评测/真实准入相关 41；新代码 Ruff 通过。一轮中间测试结果 **27 通过 / 1 失败**完整保留，原因是测试未隔离继承的 preload 环境，触发了较早的来源保护。执行前仅修正测试环境，生产保护未放宽，之后 41 项通过。正式运行失败为 1 次系统终止；控制任务跌倒另计 6 场，不混为训练失败。

本机未再次运行先前受内存限制的全量测试。新增实现及本次归档分别提交；wheel 最终提交的完整 [CI](https://github.com/LYHrmer/wheel-legged-control-lab/actions/workflows/tests.yml) 状态在交付时按 SHA 查询，不用旧提交的成功代替。D1 本轮无修改，不制造空提交；已有 HEAD `114baaa3d2b9341acb00f5a0d582e22bba95f03f` 的 [CI 39795994899](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/39795994899)已重新核验为 completed/success。

尚未解决的是：σ 减半对三个 seed 的均值与极差有什么影响；完整训练在当前资源约束下能否稳定执行；状态外推中姿态、支撑几何、陈旧速度与 guard 切换各贡献多少。课程和奖励效果仍不能从少量 seed 中排序。D1 的综合 J 相对 25 基线仍差 5.89%，ROS2 全链路、20 ms 截止期与实机效果尚未验证。本轮交付的是可审查的实现、控制负结果与训练中断记录，不是优化成功声明。
