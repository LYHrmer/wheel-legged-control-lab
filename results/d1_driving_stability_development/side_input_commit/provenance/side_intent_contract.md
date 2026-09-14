# A/D 单步提交意图契约

方案：GPT-6-astra / ultra，2026-09-14。核心交实际 Claude Opus；本文件只冻结方案，不修改源码。本阶段先生成外部工作目录候选并完成纯逻辑检查；当前里程碑提交后再由根线程集成、实测。

## 问题与范围

已实测完整侧步约在场景t=12.56s完成，其中前2s为settle，实际步态约10.56s；左右位移约±3.703cm，交接4s后保留98.3%。短按1s仍处于重心支撑移动阶段，A会先向右约2.864cm；旧drive把release的direction=0与当前侧步方向不等视为cancel，因而取消尚未完成的动作。这里修正输入语义，不把已验证保留位移的交接逻辑当作回退bug，也不声称加快步态。

本版用户行为：**空闲时短按A/D提交一次完整侧步；按住则在完成边界重复；松键停止重复，当前已提交周期继续；X请求安全取消。** 反向输入不改变正在执行周期的方向。活动周期期间不积累离线点击队列；若要下一周期反向，完成边界时仍按住反向键。这样松键后最多继续当前一个周期，不会延迟执行多个排队动作。

不改FastSideStepController、SideStepController、低层增益、步长、步态时长、IK、力分配、MuJoCo步进或77项冻结源，不调用reset/写qpos伪造移动。完整落脚后的现有handoff保持。经典GUI仍无PPO；与正在训练的85维任务分开记身份。

## 已核对的接缝

- `scripts/d1_course_side_step.py`：`CourseSideStepDrive.step`中direction不同或jump_pending时调用side.cancel；每拍唯一torque owner、唯一plant.step；只有side.done才交接，failure字符串本身不足以交接。现有factory和brake_reference_mode保留，制动默认仍legacy。
- `scripts/d1_course_controls.py`：当前side_direction仅表示A/D held差值；`_cancel`同时处理X、R、失焦、超时等，必须为新意图显式区分取消原因与普通松键。
- `scripts/d1_keyboard_viewer.py`：GLFW事件先经handle_key_event，再按实际held采样update_pressed。Shift已有同poll短点击去重逻辑；A/D事件也必须只消费PRESS一次，不能让REPEAT或held采样重复提交。
- `scripts/run_d1_course_drive.py`：先poll与离散reset/jump事件，再取commands并drive.step；zone/R reset创建新segment；只有侧步完成后下一拍才能回legacy；GUI的Space在侧步期间目前被阻挡且不排队。
- `scripts/d1_fast_side_step.py`：start要求四轮接触、实际轮接触几何接近world z=0、姿态/线速度满足门槛；drive另检查角速度。全部保持，不按zone名称绕过实际guard。

## Opus唯一核心：纯意图状态机

新增候选模块 `d1_course_side_intent.py`，最终拟路径 `scripts/d1_course_side_intent.py`。无需NumPy/Gym/MuJoCo依赖，不读时钟、文件、全局键盘或plant，不计算任何torque。公开一个类与一个不可变返回记录：

```python
@dataclass(frozen=True)
class SideStepIntentDecision:
    drive_direction: int
    cancel_current: bool
    finish_current_cycle: bool
    repeat_direction: int
    blocked_until_release: bool
    reason: str

class CourseSideStepIntent:
    def reset(self) -> None: ...

    def resolve(
        self,
        held_direction: int,
        *,
        keys_released: bool,
        active: bool,
        active_direction: int = 0,
        press_direction: int = 0,
        cancel_reason: str | None = None,
        controller_failed: bool = False,
        start_inhibited: bool = False,
    ) -> SideStepIntentDecision: ...
```

`held_direction`、`active_direction`、`press_direction`是-1/0/+1的非bool整数；active=True要求active_direction非0；keys_released=True要求held_direction=0。布尔项要求真正bool；cancel_reason为None或非空str。先验证全部输入，任何无效输入不改变内部记忆。`keys_released`是A、D**两键实际都已释放**，不是两个键相消所得held_direction==0。两个键都按住时held=0但keys_released=False，应停止repeat且不得把冲突当成紧急取消后的释放。

`press_direction`是本拍消费的一次新PRESS意图，用于PRESS/RELEASE发生在同一poll、held采样已经为0的短点击。REPEAT不产生press，已经消费的事件不能再次传入。多个同poll A/D PRESS只保留最后一个；若采样时A/D均按住则视为冲突，不从press强行选边。这个值只是当前输入包，核心不建立动作队列。

`active`和`active_direction`来自上一次实际控制器状态；不能来自按键期望。`controller_failed=bool(side.failure)`允许旧failure在done后持续为True：核心只对False→True的新失败边沿建立一次抑制，不得每拍重新锁死；之后观察到释放可解除锁定，即使旧failure仍保留。下次成功start会清除failure并自然重新武装失败边沿。

内部所需记忆只有取消/失败后的release barrier和上一controller_failed值。不需要步态阶段计时器、累计点击数、隐式恢复队列、位姿目标或地形判别。

## 精确转移语义

1. cancel_reason非None是最高意图优先级：立即清repeat/新press，drive_direction=0；若active则cancel_current=True，交旧drive调用side.cancel并继续安全abort_land/abort_hold，不直接移交legacy。设置blocked_until_release=True；同一个取消输入包中的零键值不算重新武装。
2. 新controller_failed边沿同样取消并抑制。即使后续side.failure保持True，也不重复产生新失败边沿。所有取消/失败状态必须保持可报告reason；不得因此把failure写成success。
3. release barrier存在时丢弃新press/held请求。只有一个后续无cancel、模式允许且keys_released=True的输入包能清barrier；该包只负责重新武装，不同时提交短tap。A+D同时按住不清barrier。模式仍start_inhibited时不重新武装。
4. active且没有取消：drive_direction始终等于active_direction，不受release或反向键影响。repeat_direction取当前无冲突held方向；held=0时repeat=0，finish_current_cycle=True。活动期间的短tap仅记输入，不排队；若它已经释放则不会在完成后迟发。`start_inhibited`本身只限制新周期，现周期需要取消时caller必须给明确cancel_reason。
5. inactive且start_inhibited：drive_direction=0，不缓存tap；设置release barrier，防跳跃/恢复结束后仍按住的A/D自动触发。只有模式结束后的真实释放才重新武装。
6. inactive、未blocked、模式允许：held非0则drive_direction=held；held=0且keys_released=True时，可用本拍press_direction启动一次。冲突键不启动。输出仅表示**请求**，旧side.start仍有最终物理guard。
7. 同拍tap被物理guard拒绝后不得保留延迟请求：下一拍press=0/held=0就输出0。持续held可按旧行为重复尝试实际guard，界面显示需要静止平地；guard通过前不得声称已提交正在执行的步态。
8. 成功完成那一拍仍由side controller负责。下一拍active=False时才根据当前held决定是否开始新周期；此刻仍按反向键可开始反向周期，已经松键则回legacy，不强制执行任何旧方向排队动作。
9. reset只清意图记忆并建立release barrier，不接触simulation。显式zone/R reset后，需要下一次真实键盘采样确认释放；不可用commands.reset合成的side_direction=0假装物理键已释放。构造初态可无barrier，首次用户按键正常工作。

reason至少区分 idle、start_requested、continuing_held、finish_current_cycle、reverse_after_cycle、cancel_requested、controller_failed、blocked_until_release、inhibited、conflicting_keys；名称可保持上述固定值，具体cancel_reason另由caller原样记录。输出字段必须反映当拍真实决策，不把请求启动标为已成功启动。

## 根线程后续薄集成

保留旧CourseSideStepDrive以维护原控制器接口和旧直接调用测试。新纯状态机在runner/adapter调用旧drive之前解析意图：

```text
真实raw held + 一次性PRESS + 明确cancel/mode状态
    -> intent.resolve(...)
    -> drive.step(requested, intent.drive_direction)
    -> 实际side.start/compute/plant.step
    -> 新active/done/failure用于下一拍
```

新GUI配置可名为side_input_mode=commit_cycle_v1；旧direct drive/release_abort路径仍明确可测试。根线程决定最终mode默认时，必须同时改界面帮助和协议，不通过伪造operator_side_direction隐藏松键。telemetry同时记录raw held、press、resolved drive_direction、finish_current_cycle、repeat、blocked、cancel reason、实际side phase/done/success/failure。事件cursor/键盘日志顺序保持。

controls在side active期间仍屏蔽W/S、Q/E与高度请求，直到落脚交接；不得因release后raw方向为0提前恢复前进。显示“正在完成当前侧步，松键后停止重复；X安全取消”，反向held显示“本步完成后换向”。说明当前fast周期实测约11秒，属于会继续执行的已提交动作；不承诺11秒硬截止或更快横移。

X、失焦、输入超时、fallen/恢复需要明确cancel通道，不能继续用direction=0兼任release和cancel。X按住期间一直抑制；X释放但A/D仍held不能立即重启。__call__发现输入超时也要产生cancel状态，不能仅清普通motion command。Esc/窗口关闭保持runner当前停止推进行为，不为“落脚”强行延后退出。

外部R/zone reset仍拥有显式simulation.reset权限，先执行现有reset与新segment，再清drive/commands/intent记忆；不跨segment计位移。fallen/恢复不得启动侧步，活动侧步按原安全abort路径处理；不因意图层而跳过身体/接触失败检查。

已接受的jump_pending或正在执行的jump必须阻止新side，丢弃尚未启动的短tap并抑制到释放；不要在跳跃结束后补执行。GUI侧步期间Space保持现有blocked事件、不排队跳跃，当前步继续依其既有路径；直接传给旧drive的active期间jump_pending仍走其现有cancel/安全落脚逻辑。不要混淆这两个既有入口，也不要让新意图层提高Space或A/D的优先级越过现有互斥guard。

## 有界验收

纯逻辑候选先验：空闲短tap只请求一次；held重复；active release不cancel；反向不改active方向；busy tap不排队；X与失焦/超时抑制及真实释放重置；A+D冲突不能解除锁；sticky failure不会每拍重新锁死；jump/recovery抑制不延迟启动；reset后真实release门禁；非法参数不变状态。不要写只镜像每个if的测试，覆盖这些用户可见转移序列即可。

里程碑提交后，再做有限真实MuJoCo验证，原配置/步态不调参，全部记录qpos/qvel、真实torque、controller source、contacts和cycle边界：

| 场景 | 要验证的行为 |
| --- | --- |
| 平地settle2s，A按1s再释放；D镜像 | 当前周期成功完成且只完成一次，释放后未cancel；从start起15s内完成为本版拟验收门槛；body-relative有符号净侧移2.5–5.0cm；交接4s后保留≥90%，回退≤5mm |
| held到第二周期开始后立即release | 第二周期继续完成，绝无第三周期；逐周期测位移，不能只看最终净值 |
| A开始1s后改held D，第二周期开始后release | 第一周期方向/支撑不被中途切换；第一成功done后才能开始D；两周期分别验证，最终相消不是无效动作 |
| X于支撑shift；X于实际lift/swing | 走cancel/安全落脚，不再走完整目标；四轮接触与稳定条件满足前不移交legacy；不瞬间切torque owner，不把abort当成功；A/D仍held不得重启 |
| 已有非平地guard、jump/恢复、显式zone reset | 非平地start仍拒绝；跳跃/恢复不生延迟side；reset后没有旧段pending intent，所有位移按segment分开 |

每拍唯一compute、唯一plant.step、同一真实16维有限限幅torque、无state teleport保持原测试要求。完整侧步gate数值均为**提议，尚未通过新语义实测**；现有约3.7cm/98.3%是旧完整周期基准，不能提前转写成新功能已验收。若某安全abort不能收敛，保留失败状态/证据，不强制done、不为了通过数值门槛调快步态或重置物理。

这一改动解决“短按取消未完成动作”的操作问题。它不等于可靠横移已跨地形泛化，更不完成稳定直行、越障、跳跃或速度提升全部目标。

## 实现前澄清（最终契约的一部分）

- release barrier存在且active=True时，输出drive_direction=0、cancel_current=True，继续要求原控制器完成安全abort，不用active_direction隐藏取消。若本拍真实释放清除了barrier，本拍仍不启动新周期；已经发生的failure/abort不会被清成成功。
- finish_current_cycle精确定义为active且本拍无cancel要求，并且(held_direction不等于active_direction，或者start_inhibited)。因此普通release与active反向均为True；同向held且无模式抑制为False。这只是UI/日志语义，不改变当前扭矩所有者。
- start_inhibited=True时repeat_direction必须为0，即使active周期仍由原控制器继续；不得在日志里显示被模式guard禁止的repeat计划。与其他取消不同，单独start_inhibited不触发活动周期的强制abort；需要取消时由caller给cancel_reason。
