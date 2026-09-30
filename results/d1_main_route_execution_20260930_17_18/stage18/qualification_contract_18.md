# C18-C：固定候选七场景资格合同

2026-09-30，Astra。仅在C18-B唯一候选rough/ramp完整通过并满足全部独立闭合前置后生效。此时只预注册资格设计，不发物理GO。保持C17七场景原脚本与数值门；每条新轨迹验证C18控制器，不重跑旧控制配置。

唯一系统为C18 combined：tau0.20s、alpha1/21的共同误差修正z滤波，yaw gain4/内部limit1.2；原grouped final ZIP SHA256 `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`。全部16RL通道和保护保留；zero仅为同一基控上的明确零残差对照。不训练，不换checkpoint，不因开发/资格分数改参数。

`qualification_spec_18.json`固定两worker：

| worker | 固定场次与归档身份 | controls | normal native | compiler |
|---|---|---:|---:|---:|
| primary | floor600 seed181190；原六任务依次flat0.6、flat1.6、flat1.2+yaw、bumps0.4、rough0.35、ramp0.45complete，seeds181201–181206，每项zero→grouped | 20,200 | 101,000 | 2 |
| mirror | floor600 seed181390；flat1.2+yaw seed181403、mirror=true、zero→grouped | 3,800 | 19,000 | 2 |
| 合计 | 两floor、14条正式轨迹 | 24,000 | 120,000 | 4 |

每worker只strict PPO load1/torch.load3及32行probe。合计zero controls11,400；学习正常predict12,600，probe2次×32行，predict API12,602，actor rows12,664；critic/value/train/backward/optimizer/save全0。primary soft/close/hard=1200/1320/1380s，mirror=600/720/780s，各额外≤5s仅回收。每floor clean reader≤60s计入所属host；正式独立reader各295+5≤300s，总≤600s。

C18-B+C合计最多28,000 controls、140,000 normal native、6 compiler、3 PPO/9 torch、16,603 predict API、16,696 actor rows。若开发失败，未执行资格额度不挪作新的候选或训练。C17实际32,600/163,000/+10是已关闭历史账；两阶段合计不是重复授权旧批次。

source/physics绑定、独占归档、初态逐位配对、I与z状态连续性及完整独立5T/contactForce验算沿用开发合同。每case控制器reset清I/z，zero/grouped初态五数组逐位相等。两个worker各先通过独立floor读回。来源或软件错误停该worker；合法任务失败按冻结顺序保存结果。禁止原位重试、延长时限或运行中调参。

七条grouped正式轨迹必须全部通过原记录、安全、horizon、任务速度/yaw、地形和最终停止门，才可写转向、完整坡道和六任务回归完成。特别rough RMS≤0.05；yaw/mirror RMS≤0.12；ramp均速0.45±0.04、RMS≤0.05及原几何；高速1.6和其他任务原门全保留。不得用平均分或zero通过弥补任何grouped失败。

RL贡献只按primary原六对和原两分支/成本算法另判，mirror单列，不更换有利子集。基控修复不算学习收益；若七任务通过但贡献false，分别报告工程完成与额外RL收益未达标准。grouped与global裁剪算法优劣不由本轮单checkpoint推断。

这是固定oracle确定性脚本验收；seed仅是归档身份，当前实现未产生独立物理随机性。原任务参与过开发，不能称未见分布测试，不授予GUI默认、自由键盘、实机或未测地形资格。失败应保留并依据新证据继续下一有界修复，不能修改本合同定义成功。

`plan_go_C18.json`在执行前绑定完整C18-B实际结果/receipt/初态检查、全部新来源、C spec、合同和精确来源GO。运行代码与B冻结一致时复用已完成纯测试，不增加重复检查；新来源修改必须另审，不覆盖旧记录。
