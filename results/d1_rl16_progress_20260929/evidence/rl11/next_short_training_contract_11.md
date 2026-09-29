# 11-S 世界竖直短课程：新训练与预注册持出合同

实际 GPT-6-astra ultra制定；root独占执行，Sol负责新实现，publication_handoff负责新纯scorer和独立读回。此为有界新研究，目标是生成真正完成的最新99D/16D RL模型，并用同任务零残差对照验证能力与贡献；不承诺65536步必能学会全部任务。不把传统基线通过、参数变化、或RL开关冒充学习收益。旧T/10/11-R/M/07及全部冻结源与结果不改、旧预算不复用、旧失败checkpoint和synthetic checkpoint绝不载入或续训。

## 前置证据与唯一改动

11-R真实1800/9000+2及独立readback reference_baseline_passed=true；11-M一次1024合成/1train/16optimizer完整final与严格加载通过。它们分别支持新任务参考和新预算接口。

实际物理复用已验证 WorldUprightCourseEnv、同92geom CoursePlant、固定oracle provider、实际修复引擎、原99D observation与16D joint12+wheel4 residual、原增益/保护/servo/step链。不改模型、接触法向、地面高度、scan、控制器参数、奖励数值或动作幅度。期望世界roll/pitch固定0，task20°/native硬45°；所有资格另按全时世界各轴10°，旧几何相对姿态保留诊断。新task/reference身份严格入checkpoint，不能凭99×16同形状混载旧任务。

相对失败T：本次1000tick训练episode以175tick初稳+825tick正向连续命令代替大量停车填充；固定有限混合课程，不按结果挑选重抽/延长/晋级。真实TimeLimit用post-state99D bootstrap，真实taskfail正常terminated并前进episode序号。训练末尾未完整episode单列，不伪称任务通过。

## 执行与记录预算

唯一新root子进程、新独占run目录、1800s上限、无自动重试或checkpoint选择。训练恰65536 controls（BudgetSpec(65536,1024,256,4)）：64 rollout/train calls、256epochs、1024真实optimizer steps；固定CPU PPO数值沿已实测learning11，显式PPO seed88621，从零均值actor头和log_std=-2新建，禁止load任何旧参数。

正式持出12场，前五任务各zero/policy×1600、坡道各1800，总最多19600controls。总硬上限85136 controls /425680 normal native +2冷compiler；训练327680native、每1600场8000、每1800场9000，段间不得借额，合法早终止余量作废。五native/控制及构造、reset、static、C/ Python/guard/caller账本如既有真实实现；逐rollout durable进度和分块原始记录，最终每场前后独立持久化，异常终止记录实际下界/部分预算，不补造完成。

一次learn，一次唯一final save；save内部strict reload1，随后heldout使用同一final字节strict reload1，共2次PPO.load、不得再load或选checkpoint。内部3批加heldout加载1批共4次确定性probe批（每批<=32真实训练观测）；不构造/步进额外环境。正式final policy每返回control恰一次predict，上限9800；zero actor预测0、exact zero16。训练Gaussian policy.forward与上述probe/heldout predict分列。未完成训练只存明确unqualified failure checkpoint，不进入heldout、不resume。合成模型不得使用。

训练native用既有compact训练记录：每子步finite/几何/真实非轮/计数检查，训练失败contact保存原始快照，不能声称每native完整contactForce。持出用full native qualification模式，真实contact/force和全部pre/post integrator/controller/servo/action/obs/actuator记录；训练与评估字段真实性不同须明确。预算不含任何新GUI/模型预演/额外rollout。

## 训练计划，预先固定

selection seed88622；每episode i用独立 NumPy PCG64(SeedSequence([88622,i]))，避免前场早终止改变后场选项。measurement seed由 SeedSequence([88623,i]).generate_state(1,dtype=uint32)[0]，与计划/ PPO seed分开。不增加物理随机化。每次reset前保存完整choice、seed、1000raw命令/hash；不得因预检查或任务失败重抽。每episode raw ticks[0,175)全0，[175,1000)持续该episode正向目标；vy0、jumpfalse、clearance .455，servoΔvx<=.005/Δyaw<=.006沿原规则。

前8个episode均flat，固定(speed,yaw幅度,首段符号)：(.4,0,+)、(.6,0,+)、(.8,0,+)、(1.0,0,+)、(1.2,0,+)、(1.6,0,+)、(.6,.15,+)、(.9,.2,-)。随后槽 j=(i-8)%8：

|槽|terrain|目标speed m/s|yaw幅度 rad/s|
|---|---|---|---|
|0|flat|U[.4,.9)|0|
|1|flat|U[1.1,1.6)|0|
|2|flat|U[.6,1.2)|U[.15,.3)|
|3|bumps|U[.25,.4)|0|
|4|rough|U[.25,.4)|0|
|5|ramp|U[.35,.5)|0|
|6|flat|exact1.6|0|
|7|ramp|U[.35,.5)|0|

每个i独立RNG先按表抽speed（常量不消耗draw），再抽yaw幅度（常量不消耗），最后仅槽2用integers(0,2)决定首段符号：0→+1、1→−1。非零yaw训练只在[450,550)首符号a、[550,750)反符号a、[750,850)首符号a，其余yaw0。这是对称短转向，不能让高速度无限恒yaw驶出地图。组合限|vx*yaw|<=.8不变。

spawn沿原compiled course lanes：flat(-8,-4.7,.455)、bumps(.35,-2.2,.455)、rough(.35,0,.455)、ramp(2.75,0,.455)。训练flat每次真实reset后、首control前，以纯servo1001位置和真实compiled robot collision bound做footprint clear precheck，确认C/Python计数不变；所有计划先纯名义XY安全预检。terrain命令不能拿空旷检查排除本来需要越过的障碍。预检失败为执行失败，不重抽。允许训练候选terrain到.5但不据此开放GUI资格。

首次32真实训练pre观测用于唯一finalprobe，不从heldout选。训练审核记录raw Gaussian、SB3 clip、effective16、原raw motion permission、reward各项、真实torque与计步，报告有效非零动作率和策略更新，不以最终reward代替能力。

## 正式持出表与窗口

训练完成并唯一final严格加载后按以下任务顺序，每任务zero先、finalpolicy后；同reset seed、模型、初态qpos/qvel/ctrl/qacc_warmstart/99Dobs逐位配对。均不采样探索噪声，不改变参数。前五任务的raw/servo形状继承原400hold设计，独立新seed。第六明确1800，不能走旧硬编码1600 scorer。

|id|terrain|speed|yaw幅度|reset seed|N|H|release t0|F|
|---|---|---|---|---|---|---|---|---|
|flat_0p6|flat|.6|0|88701|1600|[295,695)|695|[1500,1600)|
|flat_1p6|flat|1.6|0|88702|1600|[495,895)|895|[1500,1600)|
|flat_1p2_yaw|flat|1.2|.3|88703|1600|[415,815)|815|[1500,1600)|
|bumps_0p4|bumps|.4|0|88704|1600|[255,655)|655|[1500,1600)|
|rough_0p35|rough|.35|0|88705|1600|[245,645)|645|[1500,1600)|
|ramp_0p45_complete|ramp|.45|0|88706|1800|[600,1000)|1355|[1700,1800)|

每场[0,175)raw0，[175,t0)raw=s，[t0,N)raw0；yaw场仅其H内100tick+.3、200tick−.3、100tick+.3，其余0，实际yaw servo限速不变。ramp用长drive穿越完整up/deck/down，H是事前固定中段400样本，D=[175,1355)包含全过程；它不是旧.35短轨迹改名。所有位置、COM速度、body-frame yaw rate、姿态取当tick实际post状态，目标取当tick已消费servo。原生全时安全包括初态与全部5T返回。

任务资格：record/source/5T/geometry有效，完整N且无taskterminate，全native各world |roll|/|pitch|<=10°、clearance>=.28、无非轮terrain候选、无warning，|x|<=10.5/|y|<=5.8。低速及yaw场H speed RMS<=.05、mean在[s-.04,s+.04]；flat1.6 H mean[1.52,1.68]、RMS<=.08、>1.5比例>=.9。yaw H实际body z rate对servo RMS<=.12，servo正/负tick集合真实yaw积分分别>0/<0。最后F各场meanabsCOMvx<=.05、meanabsbodyyaw<=.05、clearance population std<=.02。

terrain场全[0,N)的spawn-heading横偏<=.25、heading偏差绝对值<=.15rad，最终沿初始heading净进展>=.75*sum(实际servo_vx*.01)。bumps/rough分别要求其真实家族至少一次正wheel contact normal load。ramp另外要求up/deck/down三个指定geom各有真实正轮载荷，最终所有四轮碰撞cylinder实际世界Xmin严格大于所有三ramp box世界Xmax；精确primitive投影基于同期measurement缓存，qpos/qvel/time必须对应真正最后状态。不能用轮中心/名义进展/geometry hit替代负载或形状通过。

flat1.6停车严格沿原05：release pre位置起，最早连续25个post端点各absCOMvx和absbodyyaw<=.05，窗口末点距release<=4.2s；到该端点所有native实际base qpos[:2]沿初始heading最大前冲<=3.2m。其余场同样报告停车量但不新增此专用门。raw释放后actor effective为0，刹停收益归基线不归RL。

## RL贡献与诚实出口

沿冻结08-05精确定义，只有姿态改为新world任务、N/H/t0/F改为上表。D_i=[175,t0_i)：每tick e=((真实post COMvx-servo_vx)/.25)^2+((真实post bodyyaw-servo_yaw)/.4)^2；c=五native各16motor的(真实applied torque/[80,80,80,12]×4)^2先平方后共80样本均值。报告每轴SSE/每pair样本/全部失败，不能调窗口或选子集。

前提：全部12记录有效、同初态配对、policy6场完整且全安全；任何zero task通过的pair，policy也须task通过。A=至少1个zero taskfail→policy taskpass；B=共同完整成功pair集合J至少4，pooled S_policy<=.85*S_zero且至少4pair S_policy<=1.02*S_zero。pooled zero SSE为0则B=false；逐pair零误差双方都0仅算no-worse，ratio null并给原因。

A/B还需所有6pair各有非空共同D交集，pooled真实c_policy<=1.20*c_zero；零成本仅双方0通过，ratio null。合法zero提前任务失败保留，成本注明仅共同窗口并另报policy独有后缀，不能把非法/缺记录算任务失败。RL_contribution_passed=前提 AND成本 AND(A OR B)。能力qualified另需6policy任务全通过（含1.6和完整坡道）。训练完成、参数真更新、能力、贡献和GUI流畅分开判，不把无贡献模型宣传成RL改进。

本单seed研究若完整训练但贡献门失败，仍交付真实final和可复现实验报告，不能盲加同构训练/挑checkpoint；先基于命令误差、真实姿态/轮载荷、有效动作/保护饱和找具体可证伪改进。若record/geometry失败，后续RL/GUI新资格关闭。A/D侧移、Space跳跃和新GUI数值/性能验证不在此预算内，仍需真实下一有限合同，不能伪按钮宣称完成。

## 新源码与有限核验

新文件只在rl11等新写域，旧冻结源只import。root新launcher/worker SOURCE GO前固定所有源/真实依赖/合同/既有闭合证据。新pure总最多8case/round、最多2round、30s/round；Pub6个scorer边界组合与Sol2个新schedule/wrapper接口case，首过停，第二轮仅首败具体修正。0engine/model，不重跑旧16/8/6或11-M。覆盖1600/1800/H/F错边界、真实world参考/旧schema拒绝、完整坡道load+exactshape门、SSE/cost/零分母/合法partial、1.6连续25/native前冲、1000truncation与budget边界/不可重抽。实际case文件与收据进入GO。

训练期间只读分析，新读回器可在非运行依赖文件独立编写；不得改任何已冻数字实现。root执行前明确唯一child PID/进程组与runtime封口，1800s崩溃/中断预留预算全部关闭无补跑。所有新执行以单独精确SOURCE GO为最后门，本文不是已通过结果。
