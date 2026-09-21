# 2026-09-21 最终交接：本轮归档结束，完整目标继续

用户完整目标仍为 **稳定直行 → 可靠越障 → 提高速度**。本轮已完成固定控制验证、两次全新RL训练及各自冻结验收；跳跃和可靠越障没有通过。用户要求在本轮结束后移交新Codex终端，本轮不再执行后续合同的物理实验。

仓库：`/home/lyh/wheel-legged-control-lab`（下文R）。工作证据目录：`/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920`（下文W）。最新可变状态为W的 `continuation_state.json`；旧checkpoint和旧实验记录保持不变。9月20日文档中的“运行中”“尚未训练”等是历史阶段状态，当前结论以本文及最终回执为准。

## 已验证能力与未完成范围

| 项目 | 当前证据 | 不能推出的结论 |
|---|---|---|
| 固定减速参考停车 | 按原始用户停令评分，两场失败，未采用 | 不能按处理后的参考重判为停车通过 |
| 原生平地停车、转向组合 | 固定八场全部通过，8400控制/42000 native，含两条已停稳后转向历史 | 不是一般驾驶鲁棒性、行进转向或实机验证 |
| 新独立8动作 PPO | 新131072步，最终策略0/5；四跳失败，hold漂移0.156127m也失败 | 软件/记录有效不等于策略成功 |
| 新共享腿动作 PPO | 新131072步，最终策略1/5，仅hold；四跳及晚段速度失败 | hold由动作门控保持zero基线，不是学会跳跃/站稳 |
| 真实台阶、坡道、碎石通过 | 下一真实单台阶合同已冻结，尚未执行 | 不具有可靠越障能力 |
| 提速、新RL GUI、人工试驾、物理自救 | 未完成；GUI生成超时无代码，R仍为仿真reset | 旧GUI的LQR/VMC能力不能转记到新PD/PI控制链 |

若必须给进度百分比，沿用约30%的工程粗估；这不是验收完成率，不能以已用训练步数折算项目完成度。稳定直行阶段也仅通过有限固定域。

停车使用非零forward实际消费到零后启用的固定腿阻尼，转向仅恢复原内层body-rate反馈，组合控制保留原PI恰一次和80Nm腿/12Nm轮保护。组合入口/证据为 `scripts/d1_stop_turn_composition.py`、`scripts/probe_d1_heading_stop_turn.py` 及 [八场公开包](../results/d1_driving_stability_development/heading_plane_stop_turn_01/README.md)。默认GUI/控制入口未替换。不要另造第二套增益或重复叠加航向P/D。

## 这轮强化学习实际做了什么

已经实际执行强化学习。两次均从零启动、各131072控制步及1024次PPO更新/4096 optimization epochs，smoke权重均弃用，仅评最终checkpoint；旧三个65k模型和旧24场G1没有重跑。

第一次seed62001使用95维oracle任务和原8通道残差，175个完整跳跃episode中16个出现训练定义的flight_once，但没有full progress/landing。最终四跳请求窗净空仅0.468/0.278/0.576/0.071mm，卸载10/8/12/4ms；原20mm/20ms门槛失败。静止漂移失败。见 [第一次完整失败证据](../results/d1_driving_stability_development/jump_ppo_formal_01/README.md)。

第二次seed62002仅将策略动作改为请求窗内四腿共同伸缩的标量，保持95维观察、原物理8维历史及原奖励。所有219个含partial episode的flight/progress/landing均为0。完整训练816.195547秒；冻结十场评估6000/30000记录有效，最终策略1/5。四跳实际净空0.225625/0.223874/0.284593/0.276964mm、向上起始卸载8/8/8/6ms；晚段最大body vx为0.040561/0.052524/0.040440/0.051735m/s，均超过0.03。静止全600控制/601状态观察/3000 native与zero逐位一致，四跳请求前前缀也一致。见 [共享动作完整失败证据](../results/d1_driving_stability_development/jump_shared_heave_formal_01/README.md)。

这是两个有限预算的失败实验。不同seed与动作结构不构成单变量因果对照；没有证明强化学习无效、机器人无法跳跃，或已经找出唯一失效原因。共享动作没有改善跳跃，不能把训练回报或初始化净空当成功证据。执行器可达性、接触时间协调及奖励的具体瓶颈仍待证实。

本次续进累计 **315424控制 / 1577120 native**；共享动作新合同消耗137712 / 688560（640 smoke +131072正式 +6000评估）。所有该合同预算均已用尽，没有待恢复的训练或评估进程。完整训练日志可由公开分块重建，实际SHA核验通过。训练保存紧凑记录，不能声称独立重放每个训练native；最终评估保存全部native记录并已审计。

## 优化后的唯一近期任务

下一终端首先执行 [冻结的单15mm台阶合同](../results/d1_driving_stability_development/jump_shared_heave_formal_01/next_terminal_plan/next_contract.md)，工作原件为 `W/next_terminal_handoff_plan_01/next_contract.md`，SHA256 `1ea04dc310486ed7025a73ed1e61a7977fdaafe0545f3de859905835d1b5c238`。实际 gpt-6-astra ultra 已制定此方案，主代理审阅后采用。

1. 新增独立 native plane + 单box plant，box全尺寸0.36×1.24×0.015m，中心在初始base前0.70m。沿原第一阶尺寸，不导入完整course、不使用hfield替代真实碰撞。先做零积分几何、contact frame、地形query、名字重绑定和记录检查；`mj_forward`如有调用须单列，不能混称零求解。
2. 前检通过后仅两场：同seed77301，plane-only与plane+box各1200控制/6000 native，最大合计2400/12000。使用已验证的StopTurnCompositionController、zero8；raw forward在k=200..999为+0.2m/s，其余为0，raw yaw=0，世界高度参考0.455m。原始停令在k=1000立即生效，不加减速ramp，不调增益。
3. 分别报告记录有效、实际轮-box正载荷接触、机械通过及最终稳定。通过须四轮和所有机器人collision geom都到box远侧且姿态/速度/接触门槛满足，不能仅看base中心。正确侧壁/边棱法向不必竖直，平地轮底高度函数不能冒充box距离。
4. 有效但失败的zero轨迹可以作为后续RL任务基线，不要求经典控制先把台阶刷通。记录/plant无效则修具体实现；有效后根据真实接触瓶颈另存一个有界RL合同，再实施一个明确的任务/动作假设。不要自动做第三次平地跳跃增训、扫描经典增益或挑中间checkpoint。

原地四轮20mm净空不是低台阶滚越的必要前提；跳跃作为独立需求继续保留原门槛。低台阶通过后才规划台阶高度/位置/摩擦变化的未见条件验收，随后扩展坡道/碎石和速度、制动距离。此长期顺序是计划，尚无对应通过证据，也未预先填入新训练预算。

用户已授权常规开发、仿真验证及及时上传GitHub。合同中的“尚未授权自动启动下一训练”表示当前固定实验不含新训练预算；下一主代理可在现有用户授权范围内依据新readiness结果制定并记录有界合同，不必例行再次向用户要许可。

## 分工与调用真实性

使用实际 gpt-6-astra ultra 制定/审阅方案；Claude Opus负责小而有界的几何、记录、测试等模块，主代理掌握接口、集成、计步和物理验收。不要让多个执行者各自启动环境或模拟，避免隐藏预算和重复实验。

用户在9月20日约22:40（中国时间）报告Claude额度耗尽、约两小时后刷新；这个刷新时间没有服务端确认。共享动作请求已取消，回执在 `W/claude_shared_heave_qual_01/attempt_01/cancelled_receipt.json`，没有生成代码、工具执行或验证，费用未知。本轮之后未再次调用Claude。下一终端如果额度已经恢复，可按已有授权恢复有界任务；遇限额保留回执并由主代理继续，不反复消耗请求。不能把主代理写的共享动作模块归给Claude。

减少token的方法是传入具体文件/接口/验收条件，只返回diff和简短验证回执；原始大日志留文件，主代理读取指标与失败摘要。Astra方案、Claude执行结果和主代理独立核验分别归档。实际provider记录的Opus模型为 `claude-opus-5`，不声称使用未显示的型号。

## 状态、发布与检查入口

优先完整阅读本文、W的 `continuation_state.json`、下一合同和 [9月14日交接](codex_handoff_20260914.md)；需要历史机理时再读 [9月20日过程记录](codex_handoff_20260920.md)。原9月14日state仍保留在 `recovery-20260912/stability_20260914/continuation_state.json`，不是最新续进状态。核对工作树、GitHub main、运行进程、77冻结哈希与下一合同身份，然后继续新工作。

本轮代码提交为 `3812eb7c6ccfa55d3daa7a1fbf8953e3737f51ba`，[CI 35518616794](https://github.com/LYHrmer/wheel-legged-control-lab/actions/runs/35518616794)已成功。本文和最终证据的发布提交由下列第一条命令定位；最终上传/CI实测回执写入 `W/final_publication_receipt_01.json` 和当前state，不能把代码CI成功冒充后续归档提交CI成功。

```bash
rtk git log -1 --format='%H %s' -- docs/codex_handoff_20260921.md
rtk git status --short
rtk gh api repos/LYHrmer/wheel-legged-control-lab/commits/main --jq .sha
rtk gh api 'repos/LYHrmer/wheel-legged-control-lab/actions/runs?per_page=3' --jq '.workflow_runs[] | {id,status,conclusion,head_sha}'
```

GitHub优先使用已验证SSH。port22此前连接关闭，port443已成功，保留既有可信github.com主机key，不关闭验证、不改持久SSH配置：

```bash
rtk proxy git -c 'core.sshCommand=ssh -p 443 -o HostKeyAlias=github.com -o BatchMode=yes -o StrictHostKeyChecking=yes' push git@ssh.github.com:LYHrmer/wheel-legged-control-lab.git HEAD:refs/heads/main
```

每条shell命令都加rtk；原始诊断用 `rtk proxy`。无 `.codegraph/` 时跳过，不创建索引。Python环境使用R的 `.local-deps`，已有MuJoCo/训练依赖；如需运行新的已批准入口，沿用 `PYTHONPATH=.local-deps:src:.` 和各BLAS/OMP线程数1，不能为查看状态而实例化旧LQR GUI并漏计其初始化积分。

不得修改77份冻结文件及旧实验记录；不得重跑旧三个65k、24场G1、本轮已完成的两次131072训练和各物理批次。冻结77来自 `results/d1_budget_study/protocol.json` 的 `source_sha256`；本共享动作正式输入215项见 `W/shared_heave_formal_preflight_01.json`。新工作使用新目录，exclusive-create，不覆盖失败。R仍为reset，物理翻倒自救未实现，新RL GUI仍无代码。

可直接粘贴的新终端话术见 [codex_continue_prompt_20260921.txt](codex_continue_prompt_20260921.txt)。
