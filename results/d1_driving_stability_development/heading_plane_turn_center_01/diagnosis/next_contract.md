# 拟定合同：纯转向脉冲轮中心运动学校正，固定3200步

gpt-6-astra / ultra，2026-09-20。仅规划及离线证据；本文件没有执行物理或授权额外预算。真实Opus负责新的有界实现，root独立集成/前检/执行。只测试以下一个机制，不叠加已另行通过的停车腿阻尼，不调整yaw cap或增益。

## 决定和证据范围

新plane左右转向均完整800步，唯一失败项是heading peak：endpoint250，raw参考±.300rad时实际yaw仅+.065866/−.065787rad，绝对误差.234134/.234213rad。原脉冲是执行tick200..249、raw yaw±.6rad/s，保持原样。

同相位pulse端点201..250：轮中心r*qdot的几何yaw-fit约±.469，实际body yaw均值仅±.133；完整接触点速度分解的轮自转fit约±.4536，其中wheel/body gap约±.3207，腿项约±.3129，滑移项仅±.00784。轮中心相对纵向速度约解释该完整gap的76.66%，其余包含腿关节带动轮体转动和接触几何。实际native法向竖直、normal load约472N，驱动与侧向摩擦yaw moment抵消；没有原轮力矩饱和。

这使系数1的轮中心补偿成为可证伪候选，但不证明heading gate会通过。`report.json`同时保留全接触点剩余项、晚段滑移和实际native opposing moments，不把所有wheel/body差距归于可消除的轮中心平移。

## 唯一公式：水平heading轴，原effective-yaw保持

只在当前raw用户命令 `forward_velocity_mps == 0.0 and yaw_rate_rps != 0.0` 时启用。零比较为精确比较，+0/−0均为零；不引入epsilon、timer、case/tick判断、历史latch、速度阈值或渐入渐出。

使用当前同一个已发布 `D1StateEstimate`：

```text
ex_h = [cos(psi), sin(psi), 0]          # 原yaw/lateral使用的水平heading frame
ey_h = [-sin(psi), cos(psi), 0]
y_i = foot_offset_world[i] dot ey_h
u_i = ex_h dot (foot_jacobian[i,:,:3] @ qdot_leg_i)
r_eff = clip(servo_yaw + 4*(servo_yaw - measured_body_yaw_rate), -.6, +.6)
omega_target_i = clip((servo_forward - r_eff*y_i + u_i)/.087, -30, +30)
```

系数严格为1，来自滚动近似 `r*omega_i ≈ v_body - yaw_rate*y_i + u_i`，符号必须 **+u_i**。保留原forward/heading servo输入；不能把raw yaw代替servo yaw用于r_eff。由于wheel与body间有腿运动，补偿后轮差动的等效yaw目标可以超过.6；必须分别记录它和仍限于±.6的body effective-yaw request。不能将其宣传为“轮差限幅完全不变”，也不能把它直接当实际body yaw。

完整四腿u都使用，不另加mean removal、空间投影、contact weighting、full-contact inverse、增益乘子、filter、轮轴补偿或接触flag切换。mean(u)必须记录以观察共同前向运动；它目前pulse RMS约.00253m/s。轮中心近似遗漏的wheel-carrier/contact/roll-pitch项保持可见，禁止乘1.29或用结果拟合其比例。数据支持的是这个固定近似的试验依据。

gate=false时原样返回父类nominal result，不重新计算等价公式或加零；gate=true只替换 `D1WheelLegTargets.nominal_wheel_speed_rad_s`，保留joint targets、extension、effective yaw。compute复用原父类恰一次，原PI/积分限值±4、轮±12Nm/腿80Nm、关节限位/速度外向保护均保持；本任务control_dt=.01、physics_dt=.002、5子步，不改变采样率。controller本身不得读取plant/contact真值或运行MuJoCo。

## 时序与最窄接缝

新增controller子类，只扩展pure `nominal_targets`及有界raw-decision信息/诊断；compute可薄包装一次父compute保存当前gate/目标/PI收据。保留旧文件；不更改冻结controller、heading `_prepare`、模型或全局factory。

新增candidate env继承已验证的native-plane heading环境（或其现有diagnostic子类），首次reset前仅替换controller adapter。不要实例化停车damper。明确要求传入external `command_source`，拒绝schedule-only入口：原G1 runner本来就使用该callback。

包装callback每次只调用原callback一次，校验并**原样返回同一个D1MotionCommand**，同时把该次raw command及time绑定到candidate controller的当前decision信息。之后原heading `_prepare`照旧计算reference/servo并preview；nominal_targets对preview和compute看到相同gate。重复 `_prepare`沿原缓存返回，不再次调用source，不重复改变reference。不能另行query callback，也不能复制整个 `_prepare`或monkeypatch模块级 `_servo_command`。

绑定信息是当前准备决策的stateless输入，不是控制模式latch。reset清空；preview不得更新PI、integral或compute计数。compute检查输入时间与published state/decision一致，并保存不可变**执行时gate快照**。`env.step`返回前已经准备下一拍：执行tick199后当前flag会指向tick200=true，执行tick249后会指向tick250=false；日志必须读取保存的compute记录，不能读取这个已更新flag错标区间。

启用执行tick200..249恰50拍；tick250立即恢复原nominal公式，保留该关断瞬态。不得因关断后反弹改为latch、延长脉冲或加斜坡。原raw参考±.300rad、左保持积分、原score窗口完全不变。

candidate单独task_schema/controller_schema/config块，同时保留actual plane身份与原编码/source/reward意义。`_build_heading_task_config`须在super返回后显式设置candidate task_schema（plane父方法使用常量），构造中只读取已预置常量。旧heading、plane-zero、stop-damper checkpoint经两层metadata检查均拒绝反序列化；runner无checkpoint/PPO路径，action严格zero8。

## 前检：不积分，不假称被动或已稳定

单元测试须guard`mj_step/mj_step1/mj_step2`并检查time0。包括：水平heading旋转共变；左右符号；u=0；pure preview幂等；原compute一次；raw gate正负零/正反yaw/非零forward；**raw yaw=0但servo yaw明显非零**仍不启用；callback每拍一次和缓存；reset；first/last pulse decision与compute快照；zero8拒绝；目标±30及原PI/antiwindup/保护；inactive原对象/数值逐位透传；metadata与三类旧checkpoint拒绝；T/T+1及partial异常归档。

先在原保存状态进行单拍目标/PI代数核对。当前100个左右pulse样本的影子目标最大2.52409/2.52287rad/s，增量最大1.02553/1.02439rad/s，同一原PI memory下单拍请求最大3.34747/3.34303Nm，无目标clip、积分clip/reject或±12Nm超限。它们不是反馈轨迹：每拍重新使用原baseline memory，不把候选memory传给下一拍，不能用于预测gates或未来饱和。

补偿是轮—腿交叉速度反馈。未保护新增轮力矩近似 `(2.2 + 3*.01)*u/.087`；它与wheel velocity的功没有固定符号，**不是阻尼，也没有被动性保证**。腿—轮惯性耦合、轮PI、支撑摩擦、10ms sample/hold可使该反馈放大运动；不能靠单拍力矩余量或名义公式宣称离散稳定。当前oracle同步无延迟，未来估计器/延迟不在合同内。若非物理检查失败则停止，不改系数/采样率/滤波继续。

## 固定四条candidate，复用对应plane-zero

基线只引用完整封存的`flat_plane_02`相同case；两停车carry provenance继续保留。不要把`plane_stop_damping`通过轨迹当inactive基线。核对case/seed55101/raw profile/gates/plant/source/default controller/模型输入以及complete manifests。原baseline不重跑。

| 新candidate case | 控制步上限 | native子步上限 | 用途 |
|---|---:|---:|---|
| stationary_turn_left_hold | 800 | 4000 | 正转唯一机制 |
| stationary_turn_right_hold | 800 | 4000 | 反转唯一机制 |
| flat_forward_stop | 800 | 4000 | 全程gate=false noop |
| flat_reverse_stop | 800 | 4000 | 全程gate=false noop |
| 新增合计 | **3200** | **16000** | 训练0、参数扫描0 |

不新增impulse物理；其raw yaw=0而servo yaw可大的gate区别由上述纯测试明确覆盖。本合同不声称已执行新candidate的扰动/延迟/GUI泛化验证。两停车仍是原plane-zero控制，预期原3失败项不变；它们验证隔离性，不要求凭空变成通过。

第一次termination/truncation结束case，不补跑。原门槛失败可依固定列表继续；数值、plant、日志或逐位隔离检查失败则停止剩余批次，保留失败区间已经进入的native步和末态，禁止reset重试。修复后如需物理须另立补充合同并计入已用预算。

## 精确配对与评分

同case reset qpos/qvel/obs/PI memory与baseline逐位一致，不引入跨正反初态的signed-zero豁免。

两turn：

- 执行tick0..199的raw/servo/action、原unlimited/safe请求、每拍5个实际torque、PI before/after、native ctrl/wrench逐位相同，gate=false。
- qpos/qvel/body-origin truth position的状态0..200（201项）逐位相同。
- observation只要求0..199（200项）逐位相同。**obs200已preview候选目标，允许不同**，即使本次增量很小也不能把该项塞入严格prefix。tick200才开始不同物理。
- 活跃50拍逐拍验证原公式+u/.087（clip前/后分别记），原effective-yaw表达式/限幅仍相同；这里“公式相同”不要求candidate在不同state上得到与baseline相同数值。tick250起gate=false，但轨迹已经分岔，无后缀逐位要求。

两stop：全800转移原请求/实际5子步torque/PI/actions/raw/servo/native ctrl/wrench、801状态/观察逐位相同；gate始终false、目标增量严格0；原gates、failed名单和metrics完全相同。其原velocity_rmse/late_speed/late_path失败保留，不能因新停车damper通过而替换基线或掩盖失败。

两turn成功条件：全部原turn gates通过，包括8秒完整、无翻倒/非轮接触/边界、roll/pitch≤10°、height RMS≤.015m、**全程heading peak≤5°**、最终参考±.300rad、endpoint450..799 heading error≤3°和|COM body-vx|≤.03m/s、全程平面原点位移≤.1m。原raw与reference不变，不以reward、晚段最终到达或candidate轮差拟合替代heading峰值门槛。

记录T转移/T+1状态、raw/servo、计算时gate/time、u_i/mean(u)、原nominal和修正wheel target、effective body yaw与等效wheel差动yaw分列、误差/PI/请求/安全/实际5子步torque、actual body yaw和全Jacobian中心/接触分解、真实native normal/tangent force和yaw moment、接触样本相位。native只读observer不得额外forward；缺接触不记零滑移。native force与另相位endpoint velocity不得相乘算能量。

最终分别报告plant/执行证据/隔离配对、两turn原gates、两noop stop原失败保留。机制预测是：腿中心运动出现后，新增目标让wheel PI继续看见body yaw缺口，增加同向驱动并减小原raw heading lag；若只增加wheel spin、腿形变或滑移而body yaw/原heading峰值不改善，即不支持该机制。改善但仍超5°依然失败。

失败即封存，不追加cap/系数扫描、latch、filter、yaw腿阻尼或训练。即使两turn通过，也不自动合并停车damper；组合需要独立合同/验证，再进入人工驾驶、真实净空与落地稳定的跳跃/台阶阶段，最后提速。旧heightfield障碍几何还需单独审计，不能用平地plane结果代替障碍证据。
