# 拟定合同：独立plane固定停车腿阻尼，最多新增4000步

制定者gpt-6-astra / ultra，2026-09-20。状态：**规划，未授权或执行本合同物理**。本合同以新plane零残差两停车的离线证据为依据，不把旧hfield失败当作新plane根因。诊断见同目录README/report；b来源仍是预先固定名义模式，不进行调参。

## 唯一变量、来源和边界

在已验证的native z=0 plane plant上，将原wheel_leg controller替换为本轮真实Opus已实现的`StopLegDampingController`；只启用固定停车后的纵向腿相对速度阻尼：

```text
Jx_i = base_rotation[:,0]^T @ foot_jacobian[i,:,:3]
u_i = Jx_i @ joint_velocity[4*i:4*i+3]
delta_tau_leg_i = -126.4374005337902 * Jx_i^T * u_i
delta_tau_wheels = exactly 0
```

无release governor、无1.0 yaw cap、无PI reset、无积分/位置锚点/接触权重/滤波/额外残差、无PPO、无训练。原heading servo外环2/.4/1和内层yaw cap .6保持，wheel PI、support、腿目标、原姿态反馈和全部保护顺序保持。不得把本方案自动加入默认、GUI或转向模式。

`b=126.4374005337902`固定来自`stop_kinematic_diagnosis_01/report.json`名义模式；新plane复算完全一致，当前诊断`report.json`保存数值与来源hash。模型相同仅允许复核推导，不足以直接采用；本次新plane实际支撑/腿回摆/低滑移证据补足了进入候选测试的依据。候选失败后不改变b或追加第二机制。

## 最窄接缝和身份

新增薄候选env继承`D1FlatPlaneHeadingEnv`；若需完整复用native/endpoint/PI/partial日志，可继承现有`PlaneDiagnosticEnv`再记录damper字段。先设置自身启用字段，`super().__init__`完成尚未步进的plane构造，首次reset前只将`self._controller`换成原`D1WheelLegControllerAdapter(StopLegDampingController(enabled=True, **asdict(self.wheel_leg_control)))`。不再换plant、不复制reset/step/control loop、不改冻结或default。

原`StopDampingEnv`仍继承hfield heading env，旧runner仍包含两条件8k集合，**不能直接启动旧入口**。新runner只4条candidate，没有bypass loop或checkpoint参数。

新candidate task_schema、controller_schema、配置中stop_leg_damping块必须如实记录；保留已验证的collision_terrain/plant身份。注意plane父类`_build_heading_task_config`硬写plane-zero task常量，candidate覆写必须在super返回后显式同步自己的task_schema。该方法在父构造内调用，因此只读预置enabled字段及常量，不能依赖尚未替换的controller。旧heading及plane-zero checkpoint均在反序列化前拒绝；85维观察、8维zero动作保留。

## 激活和保护规则

以每次**实际compute消费的forward命令**推进latch：reset inactive；相邻执行nonzero→exact zero触发；连续exact zero保持；任何非零立即关闭；初始+0或−0 settling不触发。prepare/preview不得更新latch或重复消费命令。不能写tick400/case名/epsilon触发器。

无release时，heading只替换yaw，因此每拍断言`damping.forward_command == servo.forward == raw.forward`。停车在执行tick400首次active，冲击case从未停令，始终inactive。compute调用父类恰一次，保留PI更新；delta加在unlimited request上，再执行原限矩/位置外向/速度外向保护。inactive必须返回原结果，禁止把加零的新数组路径替代原结果导致逐位漂移。

记录未保护delta瞬时功和保护后`candidate_safe-base_safe`增量瞬时功，两者不要混称。前者应为`-b sum(u_i²)<=0`；后者可能受逐分量保护影响，不把未保护代数性质当安全保护后或整个保持区间的能量证明。若要计算native子步实际作功，须增加只读native入口qvel及相同相位实际delta记录；没有该记录就不报告区间耗散功。不能用native force乘同步端点速度冒充同相位功。

## 必需的非物理前检

先固定来源清单、版本和helper/report/合同hash，复核77冻结源及plane preflight。所有测试guard`mj_step/mj_step1/mj_step2`，时间0；不夹带smoke integration。

检查zero8拒绝非法action；初态和metadata；新controller reset绑定；latch/preview幂等；正负u符号、四轮严格零增量；父compute一次；inactive/disabled精确透传；unlimited相加后原保护；新task/config双层checkpoint拒绝；正常/部分区间/异常归档。复用现有Opus模块的已通过纯测试，但新增plane接缝仍需测试。若固定b公式、有限值或基本保护检查失败则停止，不能减少b继续。

## 四条存量基线及四条新candidate

须先等本轮plane-zero续跑完成并封存相应四条基线，核对每条complete manifest、native/state/trace完整性及统一case/seed/gate/plant/default controller身份。基线可能由`flat_plane_01`原停车carry到`flat_plane_02`，来源、原件hash、复制件hash及旧/新runner有限改动必须可追溯。已有停车三项失败保留；不能选择别的零策略运行或重跑baseline来改善比较。

| case（只candidate一次） | 最多控制转移 | 最多native子步 |
|---|---:|---:|
| flat_forward_stop | 800 | 4000 |
| flat_reverse_stop | 800 | 4000 |
| forward_positive_yaw_impulse | 1200 | 6000 |
| forward_negative_yaw_impulse | 1200 | 6000 |
| 合计新增 | **4000** | **20000** |

seed55101，dt=.01/physics=.002，5子步不变。只读取复用baseline，不再执行其4000历史步。原±.5Nm world-yaw外扰在执行tick600..619通过已验证native桥施加，实际signed impulse应±.1N·m·s。固定duration、raw profile和原gates直接读G1协议，不复制后修改。

每case第一次termination/truncation结束，不补时；门槛失败可按固定列表继续。plant不变量/记录/逐位前缀/数值失败则停止余下批次，保存实际clock和已进入的native步，包括不足一控制区间的部分；不得reset重试。同一控制计算之后若native失败，不能在已有latch状态上重试。所有修复再运行必须另记已有预算并先立补充合同。

## 配对标准：同case，不需要signed-zero豁免

全部同case reset的qpos/qvel/85D obs/PI memory与对应plane-zero baseline字节完全相同。这里raw正负零的来源相同，不需要跨正反case的zero规范化；不要放宽本合同任何观察、轨迹或力矩字段。

两停车candidate对各自baseline：

- 执行tick0..399，共400拍：raw/servo命令、action、原controller unlimited/safe request、全部5子步applied torque、PI before/after、native ctrl/wrench逐位相同。
- 状态和观察索引0..400，共401项：qpos/qvel、truth body-origin position和85D observation逐位相同。obs400此处双方都含同一个raw归零命令；preview不推进latch，因此仍应相等。
- delta全0、inactive至执行tick399结束；执行tick400起才允许candidate动力学分岔。raw停令、heading目标和原评分索引始终相同。

两无停车impulse case：全1200转移的命令/action/request/applied/native ctrl/wrench/PI与baseline逐位一致；T+1=1201项状态和观察逐位一致；latch始终inactive、delta逐元素严格0。metadata中的新身份和显式zero damper诊断字段不作为数值轨迹相等对象。若只能比较部分前缀，就标比较未完成，不能称全程通过。

## 原评分、证据和判定

继续使用原raw-goal评分：执行区间端点真实base inertial-COM body-x减该区间raw forward；不要换servo/governed target、轮速或body-origin速度。原停车门槛全部保留：完整8秒、无非轮接触/翻倒/边界、实际roll/pitch≤10°、height RMS≤.015m、heading peak≤5°且RMS≤3°、全程raw vx RMSE≤.05m/s、endpoint tick500..799最大|vx|≤.03m/s、position tick500..700累计平面路径≤.05m。冲击case沿用原forward_and_impulse全部门槛。

保存T转移/T+1状态、prepared额外decision、raw/servo/action、base unlimited/safe及damped unlimited/safe torque、5子步真实actuator torque、PI前后、u_i、delta和两种瞬时增量功、latch/active、原保护、body COM/轮速、pitch和回摆描述量。保持plane原native接触证据和同步端点Jacobian分解，并标采样相位；plane运行时wheel-floor法向水平分量≤1e−12、|nz|与1误差≤1e−12，缺接触如实记录。native observer不得改变live data、warmstart、输入或额外forward。

分别报告`plant_semantics_valid`、`execution_evidence_valid`、`pairing_passed`、四case原任务门槛。只有两停车全部原门槛通过，且完整无停车冲击控制行为逐位不变、原性能门槛保持、所有plant/证据检查通过，才称该固定候选在此plane开发集合中通过。不得把“改善”替代“通过”，也不得把有限开发通过升级为默认稳定驾驶、转向、跳跃、障碍、提速或真实机器人有效。

失败则冻结结果，保持b不变，不追加参数扫描、第二机制或训练；先离线诊断新失败签名，再由独立新合同决定后续。当前合同不预支这一步预算。
