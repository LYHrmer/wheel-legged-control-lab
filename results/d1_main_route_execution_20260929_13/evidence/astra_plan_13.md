# 13：保存数据诊断、原子归档和异步 GUI 短脚本合同

审阅日期：2026-09-29。审阅角色：Astra；实际执行唯一 owner：root。本文件只记录本轮新合同与源审门，不运行引擎、策略、训练或测试。用户已授权按主计划继续开发及有限物理验证；主计划中“本次不执行”描述的是其已结束的发布轮次，不能把旧 12 的候选预算直接继承成新执行。

## 当前结论

可以立即做保存数据诊断和新源码开发。原子归档纯故障注入通过、最终新源码及依赖哈希被冻结、下列接口审阅通过后，root 可执行唯一新短脚本配对：headless 600 控制，再 GUI 600 控制。此时每臂各 3,000 normal native + 2 compiler native，预算不共享、不退款、无自动重试。不得重训、重复原九场、续写旧 rough partial 或消费缺失三场的 5,200 控制候选。

本合同的设计审阅结论为 **可实施**；在新 archive13/gui13 源码交付前，物理执行状态为 **SOURCE REVIEW PENDING**。最后的具体 GO 必须由本文件的后续源审记录或 root 的引用审阅收据绑定确切文件字节，不能把本段当作未见源码的预先批准。

## 证据边界与已阅读输入

- 完整阅读 `R/docs/main_plan_20260929.md`、`R/docs/codex_handoff_20260929.md`、`R/docs/rl16_engineering_followup_20260929.md` 和 `W/gui12/next_gui_contract_12.md`。
- 已审阅 `gui12/build_gui_capabilities_12.py`、`launch_gui_validation_12.py`、未完成的 `run_world_upright_gui_12.py`，并检查 mailbox/renderer、`R/scripts/d1_rolling_engine_runtime.py`、`course_impl08/course_native_guard_08.py` 和 `upright11/world_upright_course_11.py` 的相关接口。
- `R=/home/lyh/wheel-legged-control-lab`；`W=/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01`。R 无 `.codegraph/`，未创建索引。77 冻结源、旧 08–12 源、所有旧 records/receipts/model 保持原字节。
- 唯一模型为 `W/rl11/training_run_01/final_checkpoint/final_model.zip`，SHA256 `6cf2db80be7b990efc8be40eff307e58351eae839970b0c78ce5c9b5193f8e70`。65536 控制训练完成；九场完整且任务通过，四对 SSE/力矩成本退化；rough policy 与两场 ramp 未完成。branch B 已被四对结果否定，总贡献未完成。
- 部分独立读回是 `W/rl11/partial_readback_20260929_05.json`；schema 为 `d1-world-upright-short-rl16-partial-readback-11-v1`。不得用 GUI12 要求的完整读回 schema/字段硬挡新开发验证。
- `individual_complete_case_scores_not_matched_pair_aggregate` 是评分表；`complete_full_native_heldout_cases_verified` 是完整物理记录表。按 `(case_id, actor)` 唯一交叉绑定；不得仅信任手填 capability 的 SHA 或任意布尔。
- 旧 host `exit -15`、1800 s timeout 与 partial gzip 是已保存历史，不要求旧运行变成 exit 0。接受其 `source_hash_mismatches=[]`、worker 已退出、归属清理无 orphan、预算闭合的真实事实。

## 第一步：零引擎诊断及归档故障注入

root 读取已有四对完整轨迹，按原已保存命令、servo 和评分窗口拆分加速/保持的 vx/yaw SSE、误差均值/符号、共同轮速残差、腿姿态残差、目标/力矩限幅。可报告时间相关性与缺失字段，不做反事实因果结论，不加载模型或重放物理。后续训练假设不在本轮预算内。

新归档事务必须覆盖一个 block 的全部成员（例如 native gzip 与 npz），不能只把 gzip 写原子化后继续直接写最终 npz：

1. 在目标目录独占创建临时 `.partial`，完整 gzip close/npz close，文件 fsync；最终名必须不可覆盖原子发布（同文件系统 link 等）；发布后 fsync 目录。
2. 仅在全部成员及其摘要可验证后发布 block manifest。先发布一个成员后失败时，该成员仍是未登记 orphan/partial，不可计为完整 block。不能删除旧证据或覆盖已有最终名。
3. 首异常永久锁存；归档失败后不再接受新 block，不重试同一 block。`finish_segment` 重入只能给原失败/已封存收据，不能再写出同名文件或用第二个异常替换首异常。
4. SIGTERM/SIGINT 只锁存 stop Event；owner 在当前真实 tick 返回、当前归档边界完成后停止发新引擎调用。异常封存记录 Python attempted/completed、C/native attempted/returned、完整 block 与未登记文件，不能把五个 native 返回冒充 Python control 返回。

root 的归档回归按明确的红→修正→绿执行：一次红色故障注入复现真实旧问题，修正后一次绿色套件验证；各轮 wall 上限 30 s，0 engine/0 model/0 optimizer。不以绿色结果覆盖红色收据。使用 stdlib/fakes；覆盖成功提交与重复 finish、写入中断、gzip close、文件 fsync、不可覆盖 link、成员间失败、manifest 发布故障、已有最终文件碰撞。每项检查最终完整性、manifest 不越级、原异常保真和不重复提交。必要的纯 mailbox/事件 ack 检查可并入绿色套件；不重跑旧测试。绿色首过即止；若仍失败，停止依赖的物理阶段并保存具体故障，不盲重试。此为 root 在本轮明确选择的针对性回归，不是物理重试额度。

## 第二步：精确能力来源与开发 profile

能力候选只来自 partial05 已通过的四个 final_policy 离散组合：`flat_0p6`、`flat_1p6`、原命令脚本的 `flat_1p2_yaw`、`bumps_0p4`。这些组合是候选范围，不推成连续速度区间、任意 yaw/速度组合或自由键盘资格。rough/ramp、真实 15 mm 台阶、A/D/Space、倒车、静止转向均不加入本轮。

每个候选必须绑定 actor、case_id、terrain、原 source/compiled geometry/模型身份及两个表中的精确记录。最低正向证据是评分 `record.record_valid=true`、`task_passed=true`，物理记录 `full_horizon_controls=true`、`completed_controls=1600`、`native_returned=8000`、`physical_record_integrity_passed=true`、无合法任务失败；同时读回 `saved_records_rescored=true`、`training_complete=true`、`budget_reserved_and_closed_without_retry=true`。检查 reader/source closure 与 checkpoint manifest/文件摘要。`all_twelve_heldout_verified`、总体 RL contribution 与 `qualified_for_default_GUI` 继续为 false。

本轮只选择 **final_policy × flat × 新 0.6 m/s 直行开发脚本**。此短脚本用于查数值一致性、归档和异步开销，名称须含 `development_600`，绝不称重做/通过原 1600-control `flat_0p6` case。两臂 seed 固定为 **88813**；证据来源原 case seed 仍为 88701，两字段不得混用。原 flat spawn `(-8.0,-4.7,0.455)`，相同世界竖直 99D/16D env/控制器/servo/保护/几何。不得因结果不佳自动改 seed、速度、脚本或 renderer 参数。

固定一个 600-control segment，无物理 R/reset 测试，不设置额外准备积分。初始 reset 一次；env horizon=600（作为新开发脚本显式标注）。控制周期 0.01 s，每 control 恰 5 native；175 tick settle。命令最早只能由合法的下一 tick prepare 接收：

|已完成 control 边界|逻辑事件|最早消费的 raw|
|---:|---|---|
|0|focus true；全部松开；选择精确 0.6 档；reset|tick 0–174 为 0|
|174|W press|tick 174 已准备为 0；tick 175 开始 vx=0.6|
|424|W release，S press|tick 424 仍消费 vx=0.6；tick 425 起 raw=0|
|429|S release|tick 430 起仍保持 raw=0|
|600|恰第 600 次返回后关闭，不加 control|不再 prepare/积分|

因此 tick 0–174=175 settle，175–424=250 drive，425–599=175 raw-stop；不要求在此缩短的 1.75 s stop 窗口完成原任务的刹停资格，不人为补步。每 tick 保存 raw/servo 和真实 COM，以实际 servo 与速度报告观察值。

按 root 确认的最小范围，脚本只需固定逻辑事件输入与 ack 路径，明确标注 `logical_script`；不需要扩建完整 manual 输入系统，不冒称真实硬件/X11 键盘验证。GUI 实际 GLFW poll/focus/callback 与逻辑事件分列；不得把逻辑 held 状态与 GLFW held 混写。普通 tick 不等 UI；只有三处稀疏事件边界允许 owner 等待 main 的事件 request/ack（源审接受每个 wait 上限 5 s，均计入 active wall），两臂同样执行。命令在合法 prepare 处消费，逻辑请求/ack/首次实际消费时间和 tick 均保留。headless 也保留 main consumer + owner 的邮箱和 display copy 路径。

## 每臂硬预算与调用口

|项目|headless|匹配 GUI|
|---|---:|---:|
|fresh worker/root reservation|1|1|
|control 上限|600|600|
|normal native 上限|3000|3000|
|cold compiler native 单列上限|2|2|
|严格 load|1|1|
|现有保存 probe|1 批，按可信清单且 ≤32 样本|同左|
|正式 deterministic predict 上限|600|600|
|learn/save/optimizer|0/0/0|0/0/0|
|soft-stop / worker close deadline / host total hard wall|95 / 115 / 120 s|95 / 115 / 120 s|

总计不超过 1,200 controls / 6,000 normal native / 4 cold native；load 2、probe 2 批、正式 predict ≤1,200。正常完整臂恰 600 predict，包括 settle/制动期间，effective gate 仍由冻结控制语义决定。计数使用明确入口轻量适配器，不使用全局 `sys.setprofile`；load 内部 probe 单列。源/模型修改或未列出的额外模型入口立即 stop。soft/hard 限时从 worker launch 算起，包含 import、构造、加载和封存；主线程阻塞/崩溃时独立 host 必须仍能关闭自己绑定的进程组，不伤其他进程。完整预留不因提前结束退款。

先完整闭账并独立读回 headless。若该臂无效、提前终止、计数不符或不具备配对输入，则 GUI 不启动。仅 GUI 性能失败时记录失败，不能降分辨率后重跑、借另一臂预算或称 headless RTF 为 GUI RTF。

## owner/快照/原生链审阅门

- 唯一 owner 负责 runtime 创建/enter、真实 construct、严格 load/probe/predict、reset/step、guard 和全部封账；UI 只读模型和持租约的快照。构造后禁止变更 live model/几何和额外 model/data 动力学入口。
- 六份 MjData 身份为 live、measurement、三槽 snapshot、display；两臂相同分配。只允许 owner `live→measurement`、owner `measurement→持 WRITING token 的槽`，main `当前 READING lease→display`。WRITING/READING 校验必须包含活租约/调用线程和对象身份，不能仅做“destination 属于 slots”。
- 错线程只记录 violation 并设置 Event，不让违规 UI 线程写 runtime C phase/target、执行引擎或修改 live `_fatal`；owner 见 Event 后封存。renderer 禁止 forward/step/collision/kinematics/comPos/setConst，禁止使用 live/measurement 引用。
- 开始和结束由 owner 暂停的握手期间各渲染一次，证明 live/measurement 数组及 C/Python 计数不变；不要在并发帧中要求全局物理计数不变。
- compact native mode 可以选旧 guard 的 `mode='train'` 记录分支；本固定开发脚本的 env 使用 `mode='eval'`、学习零次，不声称 manual 已验。收据显式 `force_sampling_performed=false`、`full_contact_qualification_claimed=false`。保留每 native 的有限性、integrator、真实几何、非轮候选及 native 帐；原生数组和 control trace 只做有限分块原子提交。
- 逻辑事件带 monotonic 时间戳，未来/超期/ack 超时即锁存 stop，并由 owner 在完整 tick 边界封存；GUI window close 与 soft-stop 同理。control 仍遵守当前 tick prepared 不可回写。限定 profile 外的命令 fail closed，不接受 max-speed 范围推断。本轮不声称完成键盘 TTL/失焦刹停的 manual 验收。

## 验证与停止门

两臂同源码/依赖/模型/脚本哈希，初始 qpos/qvel/ctrl/qacc_warmstart/99D observation 逐位相同；每 control 的 raw/servo/obs/action/torque 和每 native integrator 状态逐位配对（wall/input seq/render 序号除外）。不得靠轨迹重放替代第二臂推理/积分。

成功记录要求 C、Python、native guard 三账闭合，compiler=2、600 controls/3000 normal native，零 warning/fatal/thread/copy violation，源摘要无变化，guard 末 phase/target=0，进程退出且无 orphan，非零 effective policy action 真实发生。姿态绝对 roll/pitch ≤10°、clearance≥0.28、非轮候选为 0、无地图越界；任一物理/记录/线程异常立即停发新物理，保存已有前缀，无自动重试。早停或冷构造失败不能宣称配对通过。

GUI 固定 private software Xvfb，800×500、MSAA=0、shadow/reflection off、vsync=0、最多 15 FPS。真实 active RTF≥0.8、实际 draw≥8 FPS、poll interval p95≤250 ms、snapshot age p95≤250 ms；输入逻辑事件发出到实际消费的 latency 单列，不能用 poll 间隔冒充 latency。active wall 从第一个 control 前到第 600 次返回，包含中途 barrier、记录、截图和所有慢段；import/构造/严格 load/首末暂停截图/最终 fsync 各自另列，总 wall 也报告。可保存初始、运动中、最终各一张真实非空帧并绑定 snapshot seq；运动中 capture 成本留在 active wall。

最后分别判定 record valid、headless/GUI numeric match、短脚本物理观察、GUI 性能。成功只支持此 exact 600-tick development profile 的数值/性能结论；不升级 1600 case、不称全部 GUI 已 qualified、完整键盘驾驶通过或 RL 有收益。HUD 简明显示目标/真实 COM、策略与 terrain、按键范围，并标注“RL 策略运行；尚无已证收益”；详细哈希留日志。

## 最后源审待填

等待 Sol 交付新 archive13/gui13，root 提供单次纯测试收据、最终运行计划与冻结哈希。审阅重点：事务失败路径与首次异常、真实父类退出顺序、所有隐藏模型入口、owner fence、snapshot lease、partial05 字段绑定、600 对称脚本准备时序、硬 watchdog 与独立闭账。未审阅完成前不启动物理。

### 集中源审 13-1：归档通过，GUI 修正后再绑定

已完整审阅交付的 archive13 与 GUI13 五个文件，未执行其源码。归档回归红色收据保存了 `TerminationRequested` 被 finally 的 `FileExistsError` 遮蔽；绿色收据记录 13 项纯故障/封存测试通过（unittest 0.143 s，host 0.239 s，0 engine/model）。独立读取当前源哈希与该收据相符：

- `archive13/atomic_archive_13.py`：`047fd765fc9213aaa346c0fe47393c65377ab38b04eae4a8cfa36c489f2732d7`。
- `archive13/test_archive13_fixed.py`：`d7d3bc88bbb4866f9329397c6467b904d28e906c42cfe9d493519aefde971504`。
- `archive_green_receipt_01.json`：`caf002640401316a830c699551209a7b5826b2be555c7d80dd6147c8b9fdbc5f`。

**归档源码审阅通过**：gzip/npz 同一 manifest 事务、不可覆盖发布、失败 partial 保留、首异常锁存、finish 幂等、不重试同一 block、独立失败收据均有源码与绿色测试支撑。

GUI 的本次集中审阅提出六项有限修正，尚未 GO：真实跨 main/owner 的稀疏逻辑事件确认（现稿仅 owner 自记 ack）；首末快照可能已被普通消费取走的竞态；poll 时间与三帧截图缺失；10° 本轮门和异常控制前缀封存；旧训练源与原冻结哈希比较、reader/历史闭账/合同/绿色收据绑定；115 s worker close 与 5 s host cleanup 的硬限时一致性。`mode=eval` 可作为本固定脚本标签保留，报告不称 manual 已验。修正仅处理本合同现有门，不增加物理预算或扩新输入技能。

### 同轮六项 delta 读回：具体源码 GO

已只读核对上述六项改动。main 线程实际 `schedule.submit`，owner 在合法 prepare 消费，main 再读回 ack；headless 初始消费门和 final_pending/已复制 exact control index 消除首末丢帧误判；poll 时间、active monotonic 边界与三截图入口存在；每 tick 返回后保留 row/state 再检查 10°/非轮/clearance，writer 健康时异常 finally 保存已有数值前缀；旧训练源逐项比原 session SHA，旧 partial/host/合同/绿色收据及新配对 reader 来源纳入输入冻结；95 s 发 soft stop、115 s 启动进程组清理，120 s 为硬收尾门。实际 host 总耗时（含收据与源复核）须如实核验，不能把仅有 deadline 常量当作已满足墙钟门。

Root 的 `gui_bridge_green_receipt_01.json` 记录五项纯测试通过（0.010 s，0 engine/model）：真实 main/owner 事件、拒绝 owner 自注入、wrong-thread 不改 native phase、exact index 暂停复用、当前 mailbox 租约拒错；测试源码 SHA 为 `d4fd0746f8912fcb2309dfa283ffa7325efd5965ee00388b5a78a3696b3e363a`。未由本审阅者重跑。

**SOURCE GO：允许 root 在最终输入冻结与零引擎 preflight 通过后，执行一次 headless 600-control 新开发臂；其独立只读验账通过后，才执行一次相同预算 GUI 臂。** 两臂预算与所有失败停止边界不变。绑定当前确切 GUI 源码：

|文件|SHA256|
|---|---|
|`gui13/gui13_bridge.py`|`706babf0ac2296c48d382babeedf41d0307db2c9338abb843e403a608a85181a`|
|`gui13/gui13_contract.py`|`3d59bb9870ac55dd22166280e4ff0685da6f128bf3f365cccff1e402108c2bac`|
|`gui13/launch_gui13.py`|`b78ee010b465ec7833def1adeb2f8657a5beda14a56d0c5ec5195f6a3418ff60`|
|`gui13/prepare_gui13_plan.py`|`4d32ee8f0b344364c8f91f4536ac1939412f77d2be8da5b6e9b522312615df19`|
|`gui13/run_gui13.py`|`e375e6e43bdd336aa664d7c35d5e93ba0b29a8df0ead0e89b3c68a2ae2f19c80`|

归档绑定仍为上段 `047fd765…` 源码。最终计划必须冻结实际配对 reader、主计划、本合同、归档与桥接绿色收据、旧可信 partial05/manifest/模型等依赖。此 GO 不包含重试、训练、缺失 heldout 补评估或后续 critic 模型诊断。结果仍需完整独立读回；不能预先称 GUI 合格。

## 保存诊断审阅与后续研究候选

已只读审阅 root 的 `analyze_tracking_13.py` 与 `tracking_diagnostic_01.json`。该次分析核对 8 场、139 个保存输入，未运行 engine/model/optimizer。四对的加速窗口 SSE 均改善，保持窗口均恶化。flat 1.6 的保持 vx 均偏差为 0.03240473 m/s，97.10% 的 vx SSE 来自均值项；此比例不能泛化到 bumps，后者保持 vx SSE 均值项约 10.75%。

共同轮速残差乘冻结半径 0.087 m 后，与 policy-zero 的保持速度差在符号和量级上吻合，支持“持续正共同轮速残差是候选直接作用通路”。腿姿态、接触和反馈同时变化，尚不能宣称去残差的反事实因果验证。四对保存案例的 drive/hold 总奖励均降低，未支持这些评估状态下奖励偏好稳态超速，不外推训练状态分布。`residual16_math_08.py` 的 rate mask 在 wheel joint target 重置与 wheel leg-PD 清零之前记录，0.25 不能解释成 25% 有效残差被限幅。

训练 reward 均值 3.94358、零尾折扣和均值约 397.922 是尺度线索；后者不是历史 GAE/bootstrap target。保存的 postclip 总梯度范数约 0.5 表示全局 clipping 饱和，不证明 critic 梯度挤压 actor。负 EV 与高 KL 仍不能直接定为速度偏差原因。

下一研究合同的最小 **零物理候选**，本轮不执行：先静态审 TimeLimit terminal observation/bootstrap 恰一次和 rollout 边界；然后严格加载同一 frozen final 一次、既存 probe 一批，事前固定 512 个训练内段起点（按 terrain/加速/保持分层），每个 64-step 窗不跨 episode/truncation。至多 1024 状态的 value 前向，求当前 final 的 64-step bootstrapped target、bias/RMSE/EV 及常数预测基线。4 个 128 样本 minibatch 仅做 critic loss backward，0 optimizer/learn/save/actor 更新，报告 vf_coef 加权 preclip 范数、各层范数和相对于全局 0.5 上限的理论缩放系数，结束权重哈希不变。只描述当前 final 在旧数据上的 critic 诊断；历史 logprob/value 缺失时不重建历史 PPO actor loss 或伪造 actor/value 梯度比。若此后确实授权新学习，才记录各损失真实 preclip 梯度与实际 clipping 系数，价值尺度和 target_kl 分开检验。

上述 64-step rewards 来自历史行为策略；该 target 是保存行为轨迹加当前 final bootstrap 的诊断目标，存在离策略差异，不能充当 final 真实 on-policy return 真值或精确历史 value loss。此候选只查固定保存数据上的尺度、梯度和拟合线索，不能自动进入 critic 训练。
