# 新PPO跳跃实验：训练完成，独立物理验收失败

固定一次全新seed62001训练完成131072控制步 /655360 native、1024次完整PPO更新 /4096 optimization epochs；三个保存点均通过零额外物理步的严格重载参数和动作比对。前置640 smoke权重没有复用。

最终五条件各zero/policy一次，共6000控制/30000 native，记录完整、同条件初态逐字节一致，zero hold全轨迹qpos/qvel和parent85与既有readiness hold一致。最终策略 **0/5通过**；这不是完整目标完成，也不是强化学习方法总体无效的证据。

| 请求条件 | 最终策略请求窗口最大净空 mm | 最长有效卸载 ms | 净空目标 mm | 结果 |
|---|---:|---:|---:|---|
| jump_late_a | 0.468 | 10 | 20 | 未通过 |
| jump_late_b | 0.278 | 8 | 20 | 未通过 |
| jump_lower_friction | 0.576 | 12 | 20 | 未通过 |
| jump_higher_friction | 0.071 | 4 | 20 | 未通过 |

净空扣除了编译接触margin 1 mm；要求至少20 ms真实零负载飞行及稳定落地。四个learned请求条件末段稳定性门槛均通过，但没有合格飞行，因此不能称为成功落地。全场约1.39 mm的净空峰发生在初始settling，不作为请求跳跃成绩。

无跳跃指令时，最终策略平面漂移峰值0.156127 m，超过0.10 m；zero基线为0.001089 m。最终策略因此也未取得静止保持资格。

训练218个完整episode（175 jump /43 hold），共16个jump episode取得训练定义的flight_once，但没有一个达到完整净空进度或landing_once。5/10/20 mm阶段最佳进度分别0.54066/0.28364/0.09885。单个训练seed、oracle状态与有限friction条件不证明实机或跨种子鲁棒性。

Root独立逐行核对训练计数、事件奖励边界、原始零前进/偏航指令、更新次数；独立核对评估全部30000 native状态链、计时、外力0、原80/12 Nm力矩限制和请求窗口卸载判据，新增物理0。几何/COM另有readiness全部control endpoints及指定native帧复算，明确为抽样native几何而非重解全部力。280项新非积分检查通过；CI通过属于软件证据。

Actual Claude Opus提供核心task/env/readout草稿及部分测试；root接入真实API、修复奖励边界、完成训练/评估执行和物理核验。后续四次Claude代码请求超时，没有输出代码，其回执见前置readiness公开包。

本轮没有修改77冻结文件、重跑旧三次65k或24G1。原已通过的停车/转向组合范围保持独立记录。新GUI仍未实现，R仍仅simulation reset；可靠跳跃、真实台阶越障、提速和物理自救尚未完成。

## 文件

- `training/`: 新正式训练回执、三个模型及sidecar、逐次PPO更新、episode/boundary记录。
- `training/training_trace_parts/`: 原始94,222,844字节gzip日志的按字节分块，无内容重编码。`parts.json`给出每块及重建后SHA256；用`reconstruct_training_trace.py`写入一个新的输出路径。
- `evaluation/`: 十场T/T+1/5T完整物理记录和冻结评分。
- `provenance/`: 根代理审计、前检、核心代码快照与计数脚本。
- `frozen_contract/`: 训练前冻结的任务、预算、课程及五条件评估协议原件。
- `failure_diagnosis/`: Astra ultra的可复算动作/奖励诊断及下一项零积分资格合同。
- `figures/`: 原始记录生成的训练和评估图；绿色窗是末段稳定性评分区间。

```bash
rtk proxy python3 reconstruct_training_trace.py /new/path/training_trace.jsonl.gz
```

脚本在公开包目录中运行；输出路径必须尚不存在。分块重新拼接所得SHA256已与原件核对。

下一动作由本次失败的任务/动作诊断约束，不自动启动第二训练预算，不据此替换已合格的驾驶控制。

[下一合同](failure_diagnosis/next_contract.md)只核验单一共享腿伸缩动作：请求窗口120拍内把一个标量映射到四条腿，轮残差为零，窗口外全部残差为零。原PD/PI、40mm残差尺度、保护、奖励和真实跳跃门槛均保留。该假设可能仍失败；映射正确和保存态控制代数通过不能证明真实腾空能力。
