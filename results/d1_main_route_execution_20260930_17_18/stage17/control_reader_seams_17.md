# C17 最小控制与读回接入点

Astra 只读源码后的实现建议；没有执行导入、测试、模型或物理。该文档不选择尚待诊断的增益/公式，不是物理运行 GO。

## 保留原环境与物理链

`WorldUprightCourseEnv` 必须保持原 exact class，避免改 `heldout15.record_case` 的 `type(env) is WorldUprightCourseEnv` 保护。原 `FullDriveCourseEnv` 在 production constructor 内直接 import 原 adapter，不接受 production factory；不需要为了新控制器复制整份环境。

在 `runtime.construct(...)` 返回后、首次 reset 之前做一次显式 C17 安装：检查 `env._steps is None`、`_episode_index == -1`、`_active is False`、`decision/last_transition is None`、原 loop 未 initialized 且无 decision/receipt、原 controller 无 preview；构造新 adapter，将 `env.controller` 替换，并用同一 plant、同一 provider 重建 `WorldUprightLoop`。确认 `env.controller is env.loop.controller`，`env.plant is env.loop.plant`，mode/caps未变。既有 `WorldUprightCourseEnv.__init__` 自己就在同一 never-reset 阶段做同型 loop 替换，因此无需改 env 或 physical loop。

新 adapter 调用原 nominal constructor 只复用已经热好的 `_nominal_geometry` 缓存。仍需让 worker 实际确认 construction/native账未增长；不能凭静态判断替代计步。原先 compiler=2 的 cold construct、CoursePlant、actuator、runtime、archive13、native guard 和 reset 逻辑均保留。

## 新 adapter 与控制算术

可用 `FullDriveController17(FullDriveControllerAdapter)` 继承 `reset`、`preview`、`terminal_observation_proposal` 与 `action_gate`，在 C17 中显式定义 `compute`（原方法约75行）。新 `compute` 调用独立命名的 `control_math17.compute_*`，记录新 adapter/control schema、精确控制参数与全部新增输入/中间项。

**不能只在子类所在模块导入新 compute 函数，然后继承旧 compute 方法。** Python 旧方法继续读取 `full_drive_controller_08` 的 globals，会静默走旧法。同理，仅在新的纯函数里声明不同常量不会改变被继承的旧函数。不要修改冻结模块的全局变量或 `sys.modules` 影子替换来接新实现。

名义 yaw 已提供 `yaw_feedback_gain` 构造参数，`yaw_request_limit_rps` 初值0.6。若诊断最后只选择 yaw gain/limit，可在新 adapter 初始化时显式配置 `_nominal`，保留相同 preview/observation机制，并将参数和原/新 nominal wheel/effective yaw记录到新schema。这改变的是基础 yaw 反馈，不是学习到的收益。若新公式只需现有纯函数输入（nominal wheel、body前向速度、wheel integral等），不额外改观测、环境或状态提供器。

现有99D包含wheel integral四维。若修复可通过这四维的共同/差分更新实现，就不引入第二个隐藏速度积分器；若必须新增状态，应在合同声明reset、更新、停止和独立连续链，而不是借放任额外memory缩短实现。

阶段B需要baseline/候选对照时，采用预注册的少量固定配置；配置只能在场次开始前切换并完整reset，场次内部不可切换。安装函数本身只允许never-reset时调用；不能每步替换controller。是否同worker切换固定配置由root明确冻结，所有旧配置/新配置身份写入每场schedule/receipt。若无现成安全配置入口，用两个固定worker优先于临时热替换。

## 独立 checker 的真正接入点

`read_eval16.py` 的 `check_controller` 导入自 `rl11.verify_short_rl16_training_11_04`。这个函数内部的 `recompute_controller` 又绑定在该旧模块的 globals，来源为 `verify_course_e_08_03`。**在 `read_eval17.py` 导入一个新的 recompute 函数不会替换旧 checker 的内部调用。**

最小做法是新建纯 `verify_control17.py`：复制旧约50行的 policy→clip→raw-motion gate→support 检查与约100行 independent numerical recompute，仅修改实际选中的公式。`read_eval17.py` 显式从这个新模块导入 `check_controller`。独立复算不能直接调用 live `compute_*` 来自证。旧 nominal support、常量、contactForce、几何和数值比较函数可继续只读复用。

`read_eval16.heldout_case` 已有以下链，复制为17时完整保留：真实 policy input 与 observation[k] → clipped/gated action → 独立safe torque → requested/delayed/applied16 torque → 每control5个native前后状态 → solver actuator force →每条contactForce与地形归属→post qpos/qvel重建评分。新增控制输入要同时绑定至真实pre状态和实际servo，不只检查calc自报值。

如果修改积分，C17 checker/reader需要加逐tick连续性：reset初始integral为零；`before[k] == after[k-1]`；calc新积分等于proposal/下一obs的对应值（在99D既有字段定义可查时）；终端不多更新一次。C16 `heldout_case` 没有原E reader的 `previous_integral` 连续链，不能因复用了旧reader就假定这项存在。停止latch也要用保存servo时间线独立推进。

新参数从冻结session/recipe查表取得，保存record中的值必须与它相等。若改名义yaw，应从pre body yaw、servo和足横向偏移独立复算effective yaw及wheel target；不得只信新calc里已加工好的nominal。原源证明速度反馈与评分均为base body inertial COM的body-frame前向速度，差别在pre/post时刻；不是世界X速度，也不是整机所有连杆的系统COM。

## floor 不能遗留旧 reader

C16 `offline_floor16.py` 明确导入 `read_eval15`，并检查其绝对来源；`floor_bridge16.py` 的receipt也绑定该文件。新控制器若只改final reader，floor仍会按旧算术复算并失败。

复制bridge/CLI到17，变更CLI名、目标reader绝对路径与阶段schema，指向 `read_eval17.read_floor`。保持禁物理/策略模块、去loader injection、owned child、超时回收、输入/输出hash和actor/checkpoint身份核验原逻辑。对应一项必要pure seam检查即可：live側已import假Torch时仍只启clean reader，child解析一份新控制算术fixture。不要重复旧整套测试或为新公式另写归档/nativeguard。

## source 与 recipe 的小范围复制

root 的host/worker可从16复制，只改C17身份、选定checkpoint/配置、控制安装点、场次列表及真实预算。heldout recorder若需新actor/variant标签，复制为17并仅扩展明确身份；保留exact env和writer/guard类型保护。每场另记录`controller_variant`，不要把旧`global_continue`名称悄悄解释为新基控。

新seeds和镜像场景要求新recipe/score身份。可复用原score算术，但让recipe拥有`scenario_id`（唯一场次）、`task_kind`（六种原任务类型）和实际seed；reader按完整scenario验证，scorer按task_kind调用原速度/yaw/ramp门。不要把真实seed改写为151xxx来骗过旧评分器，也不要让镜像场景丢掉自己身份。主要窗口与门是原数值，仅身份/scene映射新增。开发批次和资格批次可以共用同一套实现，由不同已冻结recipe控制。

最小改动面是：新control/math、独立checker、C17 host/worker/eval/heldout/recipe/score/reader及floor桥接身份。原environment/plant/runtime/archive/nativeguard不动；身份变化的薄副本只做必要差异。诊断结论到达后即可在这些明确接缝上实现，无需重新设计基础设施。
