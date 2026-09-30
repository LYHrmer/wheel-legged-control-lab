# C23 GUI / C24 真实15 mm：有界执行合同草案

制定角色：实际 gpt-6-astra ultra，`/root/astra_mainplan19`。用户本轮明确要求交付 B22 GUI 与真实15 mm单台阶，两项均须实施；本文件把已授权工作变为可审阅、可执行的有限合同，不要求用户再次批准常规实现。源实现、必要纯测试及静态资格完成后，由确切 SHA 的 source GO 开启执行，不能把本草案当作未见源码的 GO。root 独占执行全部模型、物理、GUI和测试；本制定者只读旧资料、写新合同和终审，不调用模型/物理/servo或项目测试。

基础状态为已发布 C22 commit `6c0b7b44717eec8475888186a7c58ba671e4fafa`（保存 publication receipt 的 CI 3.10/3.12 均成功）；B22 唯一 SHA `7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691`。B22 有限课程收益已成立、原六 RL>zero 仍 false。本轮不训练、不改旧77/C18/C20/C21/C22源码或记录、不重跑旧物理批次，用户已有结果目录不动。

## 分工和最小架构

- root：新版本化入口、环境/控制接合、source freeze/原生与模型 guard、宿主与 GUI自动输入驱动、实际所有执行、归档/独立读回集成及发布。
- Sol：C23 新 input/GUI安全接口与必要纯测试，复用已有 owner/mailbox/display 架构；实现范围以 root 的文件分配为准。
- 独立 Astra：C24 几何、明确版本化 single-box 环境/运行归档/纯评分与 reader 所有源；不实际运行。
- 本 Astra ultra：本合同、接口/来源/预算审阅、source GO、两项实际结果科学与交付终审。不得以只交合同或纯输入测试替代两个实际交付。

仅复用已证的 C13 revision02 owner runtime、三槽 mailbox、copy fence 及 render-only display 设计，版本化最小必要改变。C20 input20 是状态机依据，不修改其冻结原件。当前99D观测、16D残差、C18 combined controller、servo/动作门与 B22 数学保持不变；调用真实策略，不借 test_actor、假动作或旧85D入口冒充B22。

全进程只有一个控制 owner。env构造/reset、prepare、读取99D、唯一predict、动作门和一次control_step（恰5 normal native）均由该 owner 顺序执行。main thread只接收真实输入、提交不可变snapshot、消费独占已发布快照并渲染 display MjData；不访问 live/measurement做任何物理，不 forward、不 step、不直接改qpos/ctrl。可共享只读 model；camera/overlay只影响显示。六份数据对象和允许 copy 边照旧显式绑定，producer不得覆盖READING槽，renderer不得占用WRITING槽，地址/线程违规立即锁存失败。

## C23 可用入口和诚实的资格范围

交付稳定的新入口（由root确定脚本/命令名），有零模型/物理的 `--check`、独占新输出目录、明确错误，不覆盖旧 d1-rl。UI/日志同时显示 B22模型短SHA、C18、99D/16D、RL/zero、当前命令/实际速度、剩余控制预算与暂停/制动/reset状态。默认或醒目入口提供 `flat_1p6`，不能交付只有旧低速窗口却称已接高速策略。

`script` 模式可选择 C22已过六profiles：flat_0p6、flat_1p6、yaw_1p2（原/镜两个离散方向）、bumps_0p4、rough_0p35、ramp_0p45_complete，actor为B或zero。严格复用对应封存命令、terrain、spawn、servo、seed语义和完整1600/1800 horizon，释放后保持基控零残差制动。每次启动只有一次有界任务；选择profile/actor须在启动前，运行中不能热换权重或命令脚本。用户后续主动启动完整脚本有自身独占运行计数，不能混入本轮已预留的600步验证账。

C22控制能力可作为上述固定脚本的继承证据；本轮只对新flat.6/600脚本做GUI与headless同轨验证。其他profile在UI/说明标注“继承C22固定脚本控制资格，未在本轮重复GUI物理验收”，不宣称全部已重新跑过。C24完成后可读取保存记录显示台阶结果，禁止为演示额外重跑一次台阶。

`keyboard` 模式本轮验证profile为yaw_1p2：W为固定1.2目标的deadman，Q/E为±.3偏航，Q+E相消；未按W时raw forward/yaw为0。短按可覆盖Q/E到控制链的实际路径，但1200步事件验收不等于完成原两条1600转向任务，也不授予任意高速驾驶、倒车或地图任意连穿资格。A/D横移、跳跃不可用且界面明确说明；Space/X是Stop，R是仿真reset，不是物理自救。另有可点击Stop/Reset和启动入口，其事件进入同一owner命令路径。

## 输入、制动、reset和渲染

Snapshot至少含单调sequence、真实monotonic采样时间、immutable held keys、focus、close及离散按钮事件；退出请求不可被过期时间戳屏蔽。owner100Hz每tick检查TTL≤.25s与时钟/来源有效性。正常renderer/poller低于100Hz时，尚未过期的同一snapshot可复用上一次已验证held状态，离散R/按钮edge不可重复触发；不能直接每tick调用会把相同sequence当stale的旧input20.update。新sequence才进行新输入消费；乱序/未来/时钟倒退拒绝，确实过期、失焦、Stop均raw0并锁存必须W释放后才能重新前进。纯测试必须覆盖100Hz owner/低频poll的重复snapshot情形。

释放W、失焦、TTL和Stop通过同一C18 servo减速及动作门执行，不能简单把电机ctrl清零冒充稳定停止。close/Esc锁存退出：owner在完整tick边界停止新物理并闭合已发生记录，不因为渲染已退出仍留后台物理。硬预算/物理异常立即停止；剩余额度不足时不能为刹车偷偷加步。GUI声明的Stop完成须有实际末窗停稳证据；主动close则可以是清晰标记的仿真终止，不伪称已物理停稳。

R最多一次，在owner边界处理，清理所有控制/servo/provider/PI/z/previous-action状态并保存前后完整初态；global预算、日志sequence、宿主时限不重置，episode局部tick明确重新开始。R后需释放W重新arm。reset不作额外积分，GUI不得把跳变连成真实运动轨迹。

GUI上限800×500、max15 FPS、至多1000个render调用/worker；初始和最终各至少两个无控制推进的稳定显示帧，保存首/中/末截图及相应snapshot sequence/hash。headless render=0。保持C13既有性能门：active controls的RTF≥.8、实际活跃渲染FPS≥8、poll间隔和snapshot age p95≤250ms。RTF按global controls×.01/实际active wall计算，R不使分子倒退；仅预先声明的无控制初/末显示暂停分列，不能事后删慢帧或排掉普通输入/渲染耗时。人工、OS注入和合成脚本的来源分别记录。

## C23 唯一三段验证与上限

1. `headless_script_23`：B22，flat_0p6，600 controls。raw vx=.6仅[175,425)，其余0，yaw=0，clearance=.455；新独立episode，不重放C22原1600场。
2. `gui_script_23`：相同seed、初态、命令、600 controls。只增加GUI快照/渲染。与第1段完整初始五数组及控制reset逐位相同，逐控制观测/raw+servo/动作/控制力矩/pre-post state及全部5T native记录应相同；渲染不得改变模型或数值链。第1段坏记录/物理失败则不启动第2段。
3. `gui_keyboard_23`：仅在前两段有效且同轨成立后，真实GLFW/OS输入路径≤1200 controls。必须实际覆盖W按下/释放、Q和E各一次、Space/X停止及可点击Stop、失焦→恢复不自动重启、TTL过期→不自动重启、释放W重新arm、R仅一次、退出锁存。可由xdotool/XTest等OS事件驱动并保存事件→poll→owner消费的时间/序号链；不能把C13合成LogicalScript当键盘硬件/人工驾驶证据。

键盘建议在global前650步完成W/Q/E、停止、focus及一次≤.40s的显式input-poll停顿（只用来使真实snapshot过期；不是篡改timestamp），再在已停止状态R一次，末段执行短W、按钮Stop、释放并留至少100个控制点观察停止，最后Esc/close。自动驱动按实际owner计数和ack推进，每个事件有有限等待，不依赖冷启动时长猜wallclock。完整事件表/等待上限在source GO前由root冻结。合法提前close记录实际controls，不强求补满1200；漏验必需事件不能记总交付通过。

每段1 cold worker，PPO load≤1、torch.load≤3、一个32行probe、每control最多一次单行predict，critic/train/evaluate_actions/backward/optimizer/learn/save全部0。script每段actor rows≤632；keyboard≤1232；合计GUI≤2400 controls/12000 normal native/+6 compiler native、3 loads/9 torch loads、3×32 probe、actor rows≤2496。probe必须有限/确定且99D→16D，动作有限且真正来自B22；zero用户模式不假计RL调用。本轮不为了zero模式再开一轮物理验证。

沿C13三阶段时限：每worker纯来源/显示preflight≤240s，readiness后soft95/close115/hard120s（含冷构造/load/probe/控制/归档），退出后全源postcheck≤180s；外层host≤600s、独占清理≤5s单列；每段独立reader≤300s。source hash以完整字节为准且每阶段仅遍历一次，避免重复扫GB级闭包；不以stat代替SHA。渲染/模型/物理/进程调用全部在独立host账本闭合；失败不退款、不原地retry。

## C24 唯一台阶时空方案

C24与GUI预算完全独立；不等GUI全速/所有profile重复物理验收才准备源码，但真实C24开始前C23已形成有效当前策略控制接合与来源证据。台阶是MuJoCo真实碰撞box，不是height-query伪高度。

固定spawn=(-3.8,0,.455)，boxcenter=(-3.1,0,.0075)，half-size=(.18,.62,.0075)，前缘x=-3.28、后缘x=-2.92，真实顶高.015m。唯一yaw0、raw vx=.20于[200,1000)，其余0；horizon1200/actor，clearance命令沿C18 .455语义。zero→B，各1200，总≤2400 controls/12000 normal native；不得换高度、速度、出生点试搜，不补失败actor。

静态来源推导的初始robot包络前缘距台阶约.052128m；servo .005 m/s/tick下，到release名义位移1.561m、停后1.600m，servo在1039归零；release时末轮和整个robot包络均已有正后缘余量。该推导只证明时空可容纳，必须在实际reset后、动作前核编译geom/初态/包络，不把推导当真实通过。最终endpoint1101..1200共100点及native5500..5999共500点足够放在servo停止后。实际速度、轮形状、接触和停车由本轮物理记录判定。

新SingleStepWorldEnv24显式构造single-box plant、绑定真实ground query的oracle、FullDriveController18 combined与WorldUprightLoop，继承/等价复用已证reset/step/99D/reward数学；新类型诚实标识。旧exact-type installer不适用时必须明确新constructor并审阅，不猴补旧factory、不先造旧plant再暗中替换、不借test_actor通道。robot/关节/执行器/惯性/阻尼/摩擦/solver/timestep和obs/action/controller配置与C18生产路径一致；唯一新增地形是plane+此box。静态geom/query和真实native collision的边界/高度需相符，不能把旧85D单台阶env直接接B22。

同一compiled模型可按owner完整reset运行zero/B，初始qpos/qvel/ctrl/qacc_warmstart/99D observation以及全部控制隐状态必须shape/dtype/bytes匹配。保存各自完整初态，不仅比seed/COM。实测compiled geometry与唯一预列box/robot完全绑定；对robot collision sphere/box/capsule/cylinder按实际shape求AABB，不用轮心越过代替整机越过。compiler额外次数由C24具体constructor和nominal-cache静态路径在source GO前列清：当前实现预计≤2 compiler native；若必须单独构造参考模型核物理参数，须在任何执行前明确另立只允许一次的静态构造上限，不能运行中自行加调用。

## C24 原物理门与评分

完整1200/6000、无warning/非法调用/坏归档/物理termination、每control恰5T；zero/B原始命令相同，B动作真实且C18门保持。全程初始+endpoint+native world roll/pitch≤10°、相对初始yaw偏差≤5°、侧偏≤.1m、clearance≥.28m，无非轮terrain接触。

台阶接触要求至少一次真实active wheel-box接触、正normal load、明确geom pair/wheel index/接触位置/法向与box face/edge几何一致；保留实际触碰轮列表与全force记录。旧门没有要求四轮都接触box，不凭空新增这个门，也不能以静态overlap替代正负载。最终必须四个wheel collision geom均在记录中，且所有robot collision geoms的min_world_x > 后缘+各自margin；不能只验COM、轮心或轮中心投影。

保留旧严格停稳窗：endpoint1101..1200的max|body vx|≤.03m/s、max|whole-robot COM vz|≤.03m/s、base world-z对.455的RMSE≤.015m；native5500..5999每轮正负载fraction≥.95。同时报告当前策略末窗yaw，mean|yaw rate|≤.05rad/s；当前clearance总体std≤.02门单列。旧06已使用的endpoint275..875共601点速度窗维持mean≥.18m/s、对.20 RMS≤.05m/s，actual applied servo与raw两种参考均报告，不凭空删掉踩台阶时的慢点。

所有门是source GO前固定条件。先判每actor物理任务资格，再报告zero/B接触、穿越余量、姿态、速度、停车和真实力矩成本。B通过而zero失败可提供这项能力转换证据；二者皆通过只证明均有资格，不自动宣布RL整体超过zero，原六贡献门不因此改写。合法失败与记录无效分开，不因失败换更容易的box/seed/时窗再试。

C24最多1动态cold worker、PPO load1/torch.load3、probe1×32、B单行predict≤1200（zero不predict），actor rows≤1232；critic/train/optimizer/learn/save全部0。纯preflight≤240s，worker soft240/close270/hard300s，postcheck≤180s，outerhost≤900s、cleanup≤5s，独立reader≤600s。时间窗按完整2400步归档及新接触reader预留，不能后加。全部compiled调用与native预算须在source GO明确并由C/Python双链闭合。

## 两项“完成”的证据

C23须有可启动新入口/--check、真实B22 SHA与99D16D、完整脚本GUI同轨、实际键盘/按钮/focus/TTL/reset/退出路径、性能和截图、可读运行说明与有限范围标签；纯测试或单张窗口图不足。C24须有实际15mm compiled几何、同初态zero/B完整原生接触与越障/停稳评分和独立reader；仅名义路径、旧06或C22bumps/ramp不足。成功与失败都必须发布据实结论，不能预写CI/推送成功。

本草案尚待：Sol/root的最终输入事件表与入口文件名、C24具体constructor compiler闭包、所有新源SHA/必要纯测试、预留回执和source GO。这些是实现收口项，不是需要用户再确认的授权项。两项实现与实际验收由root按本轮已有授权持续推进，不停在方案。
