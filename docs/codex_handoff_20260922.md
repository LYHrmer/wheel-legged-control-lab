# 2026-09-22：真实15mm单台阶，两场预算用尽，接触资格未通过

完整目标仍是“稳定直行 → 可靠越障 → 提高速度”，尚未完成。本轮没有新RL训练，没有重跑旧三个65k、24场G1、两次131072跳跃训练或其评估。R仍是仿真reset，物理自救未实现，新RL GUI仍无代码。

工作目录 `W=/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922`。先读本文、`W/continuation_state.json`、`W/final_publication_receipt_01.json`，再读公开包的[结果说明](../results/d1_driving_stability_development/single_step_readiness_01/README.md)与[下一合同](../results/d1_driving_stability_development/single_step_readiness_01/next_terminal_contact_plan_01/next_contract.md)。旧9月21日与9月14日交接及stability_20260920记录均原样保留。

## 已执行的唯一物理合同

原合同SHA256为 `1ea04dc310486ed7025a73ed1e61a7977fdaafe0545f3de859905835d1b5c238`。新plant在compile前加入原box helper生成的0.36×1.24×0.015m静态真实台块，原plane、机器人与控制器保持不变，所有缓存重新绑定。原始world-z参考始终为0.455m，真实地面高度只用于原控制器的相对高度转换。关闭障碍的同版本plant作为对照。

两场顺序为plane_only、single_15mm_box，各完成1200控制/6000 native，总计**2400控制/12000 native，额度全部用尽**。native尝试、返回、时钟推进均12000，无失败native调用；无重试、补齐或额外动态探针。首次box接触前1779个native返回状态/扭矩前缀逐值一致。两场均到12秒时限，无物理提前终止。新runner只用于解释该已完成实验，下一终端不得再启动它。

本次连续续进累计317824控制/1589120 native，不含更早三个65k与24场G1。静态forward与计数mock不计入native；各前检回执逐项保存，正式物理批次另外记录4985次forward、2次setConst。随后法向诊断使用4次kinematics/comPos/collision/geomDistance，**0 step、0 forward**。

## 结果与不能声称的能力

平地原评分record_valid/task_passed均true。台阶原评分record_valid=false，原因被旧scorer归并为 `record_invalid:unhandled_exception:ValueError`；实际报错位于 `d1_single_step_integrity.validate_record_links` 的box法向/几何特征检查。原失败评分、源码、数据与manifest不修改。

11053条box接触里，11032条有正法向载荷且特征检查通过；另21条无正载荷。唯一法向异常是native index3486（t=6.972s）、contact5、后左轮 `RL_foot#geom1`：active约束、正间隙0.676533mm、载荷0，box→robot法向约(0.004160,0.000002,-0.999991)，在box顶面附近却指向下方，cone residual=1.0。它不是0.0021小角度容差的问题，不能扩大容差或翻转该条法向来刷通过。

主代理用同冻结builder重建保存的3485/3486/3487步前姿态，仅运行静态碰撞检查：全部接触geom/frame/pos/dist与保存记录逐值一致；3486步后姿态不同，符合原生接触缓存的求解相位。实际 Astra ultra 又用圆柱support解析算出3486轮最低点z=0.015676532493m，整轮在box top之上，与geomDistance相差约4.83e-10m。这足以确认原生未载荷候选法向异常已复现；具体narrowphase内部机制尚未定位。零载荷说明该条记录的直接接触wrench为零，不能推断删除约束对整个求解无影响。

补充统计显示最终全部collision geom越过box远边，最小余量0.308176m；最大pitch3.891924°；没有非轮地形接触；末100端点body vx峰值0.004056m/s、whole COM vz峰值0.000425m/s、world-z RMSE6.607mm；末500 native四轮正载荷占比均1.0。上述是**未获完整资格通过的保存轨迹观测**，不替代原评分，也不证明可靠越障、跳跃或更高速度能力。

## 实际模型、验证和来源

实际 `gpt-6-astra`、`ultra` 子代理制定并独立复审实现、几何修正、执行条件与失败结果，报告在公开包provenance中。实际Claude调用返回 `modelUsage=claude-opus-5`、`provider=firstParty`，完成几何、环境、评分模块与两份小测试；主代理集成并实际执行测试和全部物理。不能把Claude生成测试说成Claude执行过物理。一次较大的Claude测试请求600秒超时、无代码返回，失败回执保留；缩小任务后成功。

物理前47项纯测试通过；诊断后另7项通过，合计54项，无额外本地积分。14个新增源码/测试文件Ruff通过。源码中原11个实验输入保持执行时哈希；新增2个诊断脚本和1个测试不改变原实验。77冻结输入、前轮215正式输入与本轮193实验输入均需在续进前重核。GUI、旧控制入口及冻结代码未改。

## 下一工作和上传

依据这次实际异常，下一合同只处理零积分接触取证和资格语义分离，不启动RL。不得直接把台阶原invalid改成passed，也不得为了通过修改margin、摩擦、solver、增益、速度或障碍高度。完成具体几何资格问题后，再按证据另立有界任务/动作假设及训练预算；不盲目追加第三次平地跳跃训练。

常规开发、验证、上传已获用户授权，无需例行再问。上传优先使用已验证SSH443：

```bash
rtk proxy git -c 'core.sshCommand=ssh -p 443 -o HostKeyAlias=github.com -o BatchMode=yes -o StrictHostKeyChecking=yes' push git@ssh.github.com:LYHrmer/wheel-legged-control-lab.git HEAD:refs/heads/main
```

最终HEAD、GitHub main与该提交CI实际结果以W的publication receipt为准；不要用父提交CI冒充本提交CI。直接URL push后本地origin/main可能陈旧，应核对GitHub真实main。所有shell命令继续加rtk；本仓库无.codegraph，不能自行建索引。
