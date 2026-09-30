# C18-B：唯一共同修正滤波候选开发合同

2026-09-30，Astra。按`plan_18.md`的唯一z滤波公式实施，tau=0.20s、alpha=1/21，不提供运行时可选增益。实际运行前必须完成必要纯测试和精确来源GO；用户已授权常规开发与有界验收，不另设确认。C17全部文件不改。

## 固定输入与执行

唯一actor为C15 grouped final，ZIP SHA256 `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`，一worker strict load一次及原32行probe。variant=`combined`，C18独立schema标识新计算；没有新训练、权重、策略门控或地形分支。

`development_spec_18.json`固定唯一arm=`development`，output=`development_filtered_01`，floor seed181090。先执行原600 controls floor，独立clean floor reader全链通过后，按顺序运行rough_0p35 seed181105/horizon1600及ramp_0p45_complete seed181106/horizon1800，均只用grouped。raw/servo、spawn、地图、hold/drive/final窗口及所有门按原冻结脚本。seed作为归档身份，由root核无冲突；不作为物理随机样本。

保存C17对应grouped基线，不另跑baseline或旧C16。开发后的跨worker初态qpos/qvel/ctrl/qacc_warmstart/observation必须与对应C17轨迹逐位配对，再描述干预改善。旧新轨迹分岔后的同tick差值不作通道级因果解释。

## 状态和独立验算

pure math显式输入z_before、输出raw correction d、z_after、tau/alpha及eI；adapter reset清z，每tick只消费一次。common active时z_after=z_before+(1/21)*(mean(ew)−eb−z_before)、eI=ew−z_after；inactive时z_after=0、eI=ew。滤波状态即使PI拒绝commit也照公式更新。新积分候选为clip(Ibefore+3×0.01×eI,±4)，预common候选请求2.2×ew+I_candidate，commit条件仍为幅值≤12或候选请求×eI<0。P、共同P、差动、名义yaw及全部保护不变。

独立checker不得调用live residual18。reader从每case reset的z=0出发，逐tick核z_before=上次z_after，独立用真实pre qpos/qvel/body反馈及servo重算新标量、eI、PI和原扭矩；inactive必须为零。同时保留原I连续性、stop latch、schema、源/模型身份及完整5T/contactForce/native证据。不可只核adapter自报字段。

必要纯测试限于新机制和连接：共同滤波的常值/DC关系及差动保持；启动和inactive/reset清零；有限值/错误状态拒绝；积分限幅及对应退绕；独立checker能拒绝错误z/积分或参数；必要导入/语法检查。root执行并归档，无运行代码差异时不重跑旧大批测试。

## 明确上限

本开发worker：4,000 controls、20,000 normal native、2 compiler；PPO load1、torch.load3；正常predict4,000加probe API1，共4,001 API/4,032 actor rows。value-only/critic、learn/train/evaluate_actions/backward/optimizer/save均0。

host soft/close/hard=600/720/780s，额外最多5s仅kill/reap；600-control floor reader≤60s包含在host内。正式独立读回295s等待+5s回收，总≤300s。归档采用独占输出和reservation，attempted仍计账，不退款、不原位重试、不超时延长。

冻结`plan_go_B18.json`必须绑定C17计划/真实结果和对应旧基线、唯一诊断及receipt、plan/本合同/spec/GO/新测试receipt、新runtime/controller/checker/reader及完整历史输入闭包。冻结后禁止修改。实际读回与前后来源校验、模型/物理账完整闭合才可评分。

## 进入资格阶段的必要条件

rough和ramp均按原任务门完整通过，同时floor、来源、独立记录、安全、停止、初态配对及ramp几何通过。rough原hold RMS≤0.05；ramp hold均速0.45±0.04、RMS≤0.05及三坡面正轮载荷、四整轮清除。只改善rough而丢ramp不合格。

通过后才能另冻结C18-C资格GO；尚未消耗资格预算。未通过则保留唯一尝试，当前合同不准再选tau或其他候选。D分数只能说明开发脚本；最终完成仍须C18-C七场景全过，RL贡献仍单独判定。
