# C17-B：yaw 请求上限与坡道共同积分的定向开发合同

2026-09-30，Astra。阶段A已经得到保存数据证据；本合同规定下一步实际开发，不停留在诊断。root负责实现集成、唯一预留、必要测试与全部物理；Astra只读审阅。用户已有任务授权，不另设确认。首次真实调用前，root把本合同、新源、选定checkpoint、诊断输入及针对性验证结果绑定到精确plan；这是内部执行冻结，不是用户审批。

## 1. 选择依据与两个因素

输入为 `yaw17/saved_yaw_diagnosis_17.json` 和 `ramp17/saved_ramp_diagnosis_17_v02.json`。旧77冻结源与已完成C15/C16记录不变。ramp第一版保存数据诊断因inactive common项恒等式适用域而停止，属于0模型/0物理的解析失败；保留原输出/错误，v02是新脚本/新输出，不将第一次失败覆写为成功。

**Yaw：只把内部请求上限从0.6改为1.2 rad/s。** grouped hold中51.75%到达原上限；负平台101行全部顶−0.6，平均actual yaw约−0.10198而servo为−0.3，平台SSE占整个hold约58.92%；正平台同样顶限。wheel目标、力矩与保护均未限幅。由此优先检验请求饱和，暂不改变gain或训练。

`u_yaw = servo_yaw + 4.0 * (servo_yaw - pre_body_yaw)`

`effective_yaw = clip(u_yaw, -L, +L)`

baseline L=0.6；yaw候选L=1.2。名义轮速映射、servo、动作、轮PI和common P均不变。1.2为唯一预选对称值，不进行候选网格。旧命令与所有物理/安全/计分门不变；这是内部控制参数变化，不是扩大用户命令或放宽验收。

**Ramp：修改轮积分误差的共同分量及其对应的antiwindup退绕方向。** grouped原hold pre body vx约0.51002，命令0.45，wheel target等效速度0.46158、wheel实际0.47814；平均积分+1.46606Nm、body P−1.22486Nm，合计仍+0.24121Nm。hold第一四分段body约0.49955但wheel约0.44768低于target0.45917，因此原积分仍增加；zero同样存在body超速而wheel误差为正。无目标/力矩/保护/支撑clip，全部积分commit。控制与评分均是base body惯性COM在body frame的前向速度，仅pre/post时刻不同，H4的物理点/坐标误解被排除。

对四轮定义 `ew_i = wheel_target_i - wheel_omega_i`，R=0.087m：

```
if body_common_p_active:
    eI_i = ew_i - mean(ew) + (servo_vx - pre_body_vx) / R
else:
    eI_i = ew_i
I_candidate_i = clip(I_before_i + 3.0 * 0.01 * eI_i, -4.0, +4.0)
```

P仍是 `2.2*ew_i`，积分仍只有原4维、每tick更新一次；在候选clip/commit前，差分 `eI_i-eI_j` 与原 `ew_i-ew_j` 相同。`candidate_request=2.2*ew+I_candidate`；commit条件为 `abs(candidate_request)≤12` 或 `candidate_request*eI<0`，作用点仍在后续common P之前。**退绕方向随实际积分增量eI更新，不能继续用可能与它异号的旧ew阻止退绕。** 这是同一个积分误差坐标重构的组成部分，应完整披露；不声称原antiwindup乘积输入逐字不变。不改积分上限、Ki、力矩阈值或停止控制。common inactive时eI=ew，严格保留原wheel积分定义和停止分支。

新共同积分以任务servo为参考，因此会补偿持久的common残差偏置；这属于基础速度闭环的作用，必须如实归因。全部RL动作继续进入原leg target/wheel target/P力矩路径，不改变scale、不删除维度、不按yaw/ramp关闭policy。该修复的成功不能证明RL贡献。

## 2. 三个固定 worker 与配对场次

唯一模型为C15 grouped final，SHA256=`1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`，deterministic predict。不load global或旧final，不train/backward/optimizer/save，不挑checkpoint。三个独立cold worker顺序运行，各只load该模型一次并核对冻结32行probe；每worker一个始终不变的控制配置，不在场次内部或中间改参数。

| worker / 配置 | 场次（固定顺序） | controls上限 | normal native上限 | compiler上限 |
|---|---|---:|---:|---:|
| baseline：L=.6、原wheel积分 | floor600；yaw1600；ramp1800 | 4,000 | 20,000 | 2 |
| yaw_only：L=1.2、原wheel积分 | floor600；yaw1600 | 2,200 | 11,000 | 2 |
| ramp_only：L=.6、新共同积分 | floor600；ramp1800 | 2,400 | 12,000 | 2 |
| **B总上限** | **7场** | **8,600** | **43,000** | **6** |

development seeds固定：floor=171090、yaw=171103、ramp=171106。root在首次调用前核查这些seed/配对从未执行。跨配置共用相应seed、raw脚本、spawn和初态；全部初态qpos/qvel/ctrl/qacc_warmstart/observation逐位相同，积分归零。它们属于开发，不充当C独立资格。

脚本数值沿用C15，仅seed与独立场次身份更新：floor 175 settle+250 raw vx=.4+175 stop；yaw为1600步、vx1.2、hold[415,815)、raw yaw正100/负200/正100各|.3|、release815、final[1500,1600)；ramp1800步、vx.45、hold[600,1000)、release1355、final[1700,1800)，保留全部几何与停止要求。任何development task failure都保留，不改脚本到通过。

baseline必须跑相同新seed，为两个最小因素提供当批对照；不能用C16历史seed代替。每worker的floor独立读回通过后才能运行其正式开发任务。floor失败时该worker剩余场次废止，不自动换seed。合法任务失败可按固定脚本继续记录同worker剩余场次；软件、归档、来源或计步错误停止对应执行。三worker各自预留、各自关闭，不用一者未执行的额度增加其他者。

## 3. 模型与时间预算

总PPO load≤3、torch.load≤9；probe predict共3次batch×32行。正常predict≤8,600次/行；总predict API≤8,603，总actor rows≤8,696。critic forward/value-only、learn、train、evaluate_actions、backward、optimizer、save均为0。实际attempted/returned分别记账；提前终止按实际记录，未用reservation不退款或重用。

各worker host均计入import、hash、construct、load、probe、floor reader、正式任务和归档。三个worker均为soft/close/hard=600/720/780s，三个host预留合计≤2,340s，额外各最多5s只用于kill/reap。root若采用更紧上限可在plan明确减少；不得执行后延长。每个floor clean reader≤60s（含回收），计入所属host。最终独立保存数据reader按worker各≤300s、合计≤900s，0模型/0物理；若失败，保留失败，不自动把已有数据视为通过。实际配置键`baseline/yaw/ramp`分别对应本表`baseline/yaw_only/ramp_only`，无额外配置。

独立reader不得在live worker直接导入。复用C16 bridge/CLI保护的新17副本，指向C17 reader；全局import禁令、去loader injection、owned child、来源与输入/输出hash、超时回收保持。所有底层runtime/native/archive防护保持原样。

## 4. 必要实现、记录与独立验证

保留exact `WorldUprightCourseEnv`、CoursePlant与原physics/archive路径，只在never-reset阶段一次安装固定C17 adapter并重建同一WorldUprightLoop。将新adapter schema、variant、L=0.6/1.2和积分模式写入reset/每场identity；不以旧controller_schema冒充新法。baseline在新接缝下的控制算术必须与旧法相同。

yaw记录真实pre body yaw、servo yaw、未限幅/限幅请求、配置L/gain，以及用于名义轮速映射的实际足横向偏移。ramp记录servo/pre body vx、原wheel_error、实际eI、integral before/candidate/commit/after、base wheel/common/final/safe torque。保留所有原policy/raw/clipped/applied action及5T motor/native/contactForce字段。

新增独立checker显式接入C17 reader，不能以另一个模块的同名global误以为替换了被继承函数。独立算术不得调用live新compute。新输入绑定到真实pre qpos/qvel与消费servo；按配置独立重算nominal yaw和积分公式，核对reset零积分及每tick before/previous-after连续性。保留完整requested/delayed/applied torque→native→contactForce→post metrics链，原数值评分门全保留。

root执行与本次改变直接相关的一组纯验证：baseline新接缝与旧控制数值一致；yaw未饱和/正负饱和对称与上限正确；ramp wheel慢但body超速保存状态下common eI转负、差分误差不变；commoninactive/stop走旧积分；超限candidate且candidate_request*eI<0时允许实际退绕；clip/antiwindup阈值与完整RL动作映射保留；clean reader能按新checker复算一个fixture。测试及静态检查不构造模型/仿真，不重复C15/16已完成批次。测试失败修复后保留失败凭据，冻结最终源后仅执行一次各物理worker。

## 5. 开发判定与立即后续

对yaw，报告完整任务pass、hold总RMS/SSE、固定三raw子段与servo plateau/transition子集、旧±.6区外的实际反馈请求比例和新上限占比。主要可证伪预测是平台误差下降；仅饱和比例下降而RMS不改善不算机制修好。

对ramp，报告完整任务/几何/停止pass、hold均速/RMS、固定4×100步子段、实际common积分误差与积分、目标/实际轮速和body速度、所有保护占比。主要预测是body超速时共同积分不再因wheel偏慢增加，持续速度正偏差下降；公式按预期执行但任务失败仍是失败。

两候选分别通过完整定向开发门后，立即冻结其组合进入C独立资格准备；若其中一个失败，先用该次记录明确失败环节，再在新短合同下选择下一单因素，不盲追加训练或参数扫描。B的结果不直接取得新default GUI、完整六任务或总体RL贡献资格。

C保持已接受的最小资格结构：一套新六任务，另一个独立seed的原yaw、镜像yaw与ramp，对照同一修复基控下zero及唯一学习策略；共每actor14,800 controls，floor与精确C预算另冻。所有yaw/ramp与六任务回归逐场通过原绝对门后，才能报告“转向、坡道跟踪完成”；任务资格与总体RL贡献分别裁决。若C未过，明确失败并持续推进下一有界修复，不在本诊断或B开发后收尾。

本合同的Astra编写动作0模型/0物理/0项目测试。以上为B的预注册上界，真实实际消耗以root各reservation/receipt/独立读回闭账。
