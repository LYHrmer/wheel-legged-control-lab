# B22 tracking reward 42：有界对照与执行准备

状态：**RESOURCE_REVISION02_SOURCE_VALIDATED_NOT_EXECUTED**。资源修订02的五组pure/fake组合测试**208项PASS（3.23秒）**，10份Python源Ruff及diff-check通过；原77保护输入共21730336字节逐SHA核验PASS，所有者已解冻experiments.py仍核当前固定SHA。原release01已准备但未执行，正式output01不存在。用户进一步明确允许本机慢、无需如此保守，因此仅修订资源准入与进程组限额，不增加训练或评测次数。尚未执行真实依赖bootstrap、加载策略、构造环境、积分、训练或评测。旧77、用户9结果、v27及31～41源和原回执不改，42数值worker/reader/helper保持原字节。

旧release01保留SHA `7128e72d0fcfde9c0322d185e188f0648ef350c90911705fdb52e5fc8de597f0`，来源commit `4c55bbb2baeca84dc3db897ffeaeb07329faa38e`。修订02以`resource_revision=owned_cgroup_v2`登记新的`*_plan_20261018_02.json`、`*_20261018_02`输出和`*_admissions_20261018_02`，prepare/execute均拒绝已存在的原output01。旧release原字节不变且不执行；02使用同一份尚未消耗的96136controls预算，不是续跑或再加一轮。

## 单一干预与可证伪目标

[已有奖励分析](b22_reward_signal_analysis_20261018.md)提示tracking与height/attitude等项的相对取舍值得检验，但未证明PPO信噪比或失败因果。42比较control α=1与deficit8 α=8。令：

- qx=((vx−consumed_vx)/0.25)²，tv=2 exp(−qx)；
- qy=((yaw_rate−consumed_yaw)/0.4)²，ty=exp(−qy)；
- **仅训练时**改为tv′=2+α(tv−2)、ty′=1+α(ty−1)。

总奖励变化为(α−1)(tv+ty−3)。其余七项tracking_vy、height、world-upright attitude、actual_torque、action_change、leg_speed、termination原值保留，仍是原九项。分母、指令时刻、5T实际力矩、终止和安全边界不改。α1特判原值返回，避免减再加丢失微小浮点值；完美tracking时两臂相同，α8两项最小值为−14/−7。

这是tracking deficit相对权重的一项干预，不改指数形状、vx:yaw=2:1、动作空间、执行gain或课程。它不同于已有Fx/Fz/低维试验及gain0.5缩放。旧平地反事实0.000645/拍增量代数乘8为0.00516，只是公式敏感性，不是新策略收益。如果真实配对SSE未改善或成本受损，保留阴性结果，不更换系数、补训或挑checkpoint。

## 共同起点与final-only

两臂均独立加载完整B22及同一Adam，不续接彼此：

| 固定身份 | 值 |
|---|---|
| ZIP | 7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691 |
| policy_state | 59440de0c5beb102cec0d292838fb9b125eca7eb56bbffc06d1eaec46ec7aa48 |
| optimizer_state | 7ff761525a0b679deb266091746f15992c715de181740673d4e7a00f2fb5b82f |
| 当前本阶段计数 | num_timesteps32768 / _n_updates104 / Adam step1635 |
| 共同下一阶段seed | 201011；course_arm=B，另设reward_arm=control/deficit8 |

learning21.load_warm_start20的stage2合法路径先核policy/actor/log_std/Adam及计数，设seed后重核hash，只清新阶段计数。两臂不能以旧A标签绕过parentage，不重初始化optimizer。历史档案声明B22之前有81920祖先controls，再加本阶段32768；当前32k不是从零训练。未重新审计的祖先声明不能升格为本轮累计actual。

共同原参数为单进程单环境、99D/16D、128×128 Tanh、n_steps1024、batch256、最多4epochs、LR3e-4、gamma.99、GAE.95、ent.001、vf.5、actor/critic各.5梯度裁剪、target_KL.03。batch KL>.045正常跳过该批更新并结束train，>.30/非有限值硬停。每臂仅32768controls/32rollouts，唯一final保存并严格reload/probe，禁止中间权重选择。

共同B课程、offset0、command seed201002、measurement seed201003与1000拍episode计划不改。[spec22原件](../results/d1_main_route_execution_20260930_22/spec22.json) SHA为f061b64320e8fae35dd8ca0d186242579b4640bd5fecdd3e6da63d2f6983f8f4。迁移runtime目录旁没有spec副本，新适配必须显式绑定原件，不能直接import按相邻路径找spec的recipes22。两臂保存并核最初动作前完整初态/课程/随机源；更新后轨迹不同是结果，不能强求后续obs相等。

## 两臂都评、原六贡献门保留

固定顺序：零积分依赖bootstrap→训练control→训练deficit8→共同原奖励评测。每个数值worker之后独立保存读回；仅完整训练、保存、来源和清理门通过的final可评。工程异常停止并保存partial，合法任务失败记录实际计数、不补齐或重跑。不能因一臂较差而只评另一臂。

评测继续原reward/schema、控制器、**gain1**及task/cost数学，训练reward/α另以新C42 metadata真实绑定。每actor原六任务共9800controls（5×1600+ramp1800），seed181201～181206。新zero只跑一次，不加载网络，与两个final共享完整reset/命令/模型/几何身份；不用旧gain试验的zero冒充新配对。

原600拍floor在C22 development用于决定A/B能否进入后续评测；它不是score18.score_pairs原六公式的输入。42为两final各留600拍，沿原floor_schedule、固定seed201090和原数值门，**仅单列短能力结果，不再据此筛掉后续评测臂**。合法floor失败仍收齐两臂原六，工程/安全硬门照旧。

每臂与同一zero独立走原六门：来源/完整窗口/安全前提、cost≤zero的1.20倍，且原branch A或B通过。当前zero皆task-pass，不能预期靠转换失败任务获益；branch B仍要求≥4共同task-pass、pooled SSE≤zero的.85、≥4项SSE≤zero的1.02。保留所有单项、停车/地形、成本与control/α8直接比值；不同reward总回报不能直接当可比优劣。

原六是已有多轮结果的回归/贡献比较集，不称未见holdout。单一共同seed的有限对照不证明显著性、跨seed/地形泛化或实机，不自动改默认。

## 新增上限与资源

| 阶段 | controls | normal native | compiler native |
|---|---:|---:|---:|
| 独立依赖bootstrap | 0 | 0 | 0 |
| control唯一训练 | 32768 | 163840 | 2 |
| deficit8唯一训练 | 32768 | 163840 | 2 |
| eval：两600 floor＋三actor原六 | 30600 | 153000 | 2 |
| 总上限 | **96136** | **480680** | **6** |

三个数值worker各仅一次course+nominal冷构造；所有compiler另计。父/保存reader/compare为0模型/0MuJoCo物理。每训练臂保留原1learn/32train/≤512optimizer、actor/critic/API计数、final reload/probe，不藏额外NN smoke或积分；新增任何验证调用须先登记预算。

训练worker的每次reset含课程名义路径与有限几何资格路径两遍纯Python servo，不能计作零计算。worker上限为160000＋160000＋30600＝**350600次advance**；saved-reader独立复算为80000＋80000＋30600＝**190600次**，合计**541200次**，与MuJoCo/native账分开。两侧均在真实advance调用前计attempt、返回后计return；异常时actual未知保留null并单列已返回下界，不从计划填零。

bootstrap是独立额外进程，硬wall300秒、64MiB写额度，所有模型API、Env构造、servo、normal/compiler native预算均0。它只做真实依赖导入、显式MuJoCo `_functions`/`_rollout`加载、原ELF/GOT绑定和适配函数/类定义，不创建环境或策略。依赖导入边界由固定源码审核与原生零计数约束，不能称导入期间已全程instrumented；随后适配定义区由ModelCounter及四类构造拦截器共同约束。拦截器记录Env、PPO、BaseAlgorithm、nn.Module构造尝试，即使内部吞掉异常仍失败。EngineGuard不arm，先绑定并解析`original_step`指针再取等值baseline；允许解析地址，不允许任何积分。几何缓存必须保持cold、native计数/phase/target/limits全0，四个GOT槽保持原证明。bootstrap不声称运行过独立saved-reader，也不授予控制资格。

修订02保留启动6GiB/运行停止4GiB，将原8GiB连续60秒工程门改为**6GiB连续10秒、5秒间隔至少3点**，间隔异常重置稳定计时；下一请求/nonce未建时最多等300秒，末次来源检查后仍须6GiB。4GiB运行门与准入均指host及本任务上方外部祖先的可用量，**不把本任务自身3GiB硬上限剩余量同4/6GiB比较**。原31的min-own-cap读法在此限额下会误拒绝，因此只在新父资源接缝采用stdlib guard；冻结31及数值worker无需修改。

整个任务必须位于唯一systemd用户unit `wheel-b22-tracking42-owned02-20261018-01.service`的同一cgroup，guard实际核对应`MemoryHigh=2048MiB`、`MemoryMax=3072MiB`、`MemorySwapMax=0`的内核文件及`memory.oom.group=1`；启动unit另登记`OOMPolicy=kill`，组外systemd记录实际配置与终态。父、执行期数GiB输入逐块hash、bootstrap、所有数值worker、saved-reader、最终比较均受同一组限额约束；不能只限制训练子进程。父监督每阶段、保存reader与worker启动门都读同一登记scope，记录host/外部祖先与owned current/high/max/events/peak，未知或OOM事件失败。单CPU线程、Nice≥10、串行，无自有残留child。

3GiB是否足够这次wheel训练尚无实测峰值证明。历史可用RAM跌幅不是本任务RSS/PSS；输入hash每次只持1MiB读块，但多GiB文件页缓存可能记入组memory.current并被内核回收，不能把它称为私有内存泄漏。MemoryHigh可使回收变慢，90分钟总wall不增加；MemoryMax可能导致本轮INCOMPLETE，不能自动重训。若整组被内核OOM杀死，进程内finally和最终回执不保证执行，须由组外systemd终态/事件与已落盘阶段记录闭账，缺失实际计数保持null，不能把RUNNING旧回执当成功。

来源核验、等待、bootstrap、训练、保存读回和清理全部计入**5400秒/90分钟**。bootstrap soft/close/hard为240/275/300秒；数值worker为1140/1180/1200秒，每次saved-reader另留600秒、最终比较300秒。不足以保留读回/清理窗口时不spawn。旧B22 host训练571.56s+读回328.22s；按旧eval吞吐投影全轮约45～50分钟，只是历史估计，不承诺完成或过关。

### 完整归档容量依据

只读完整本地原C22 metadata和文件stat，未读/解压大native/control归档：

| 原完整数据 | 文件/块数 | 字节 |
|---|---:|---:|
| train_B_1全部regular文件 | 1405 | 843918960 |
| native transaction payload | 32块 | 384854132 |
| full-control声明 | 164块 | 235317345 |
| numeric声明 | 32块 | 165420096 |
| Gaussian声明 | 32块 | 13158400 |
| 梯度及其manifest | 128文件 | 31670784 |
| development_1完整49600拍输出 | 1422文件 | 868625619 |
| final_1完整25600拍输出 | 728文件 | 452323930 |

本地原件根为/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01/continuation22/。公开training_blocks_manifest为49407B、SHA a2d8c7c7605fe87cdbdbf02a65c8beec48f73e76da064f2a621e591a66877b0b；它包含全32/32/164块声明，公开首块不是完整payload。

用历史最大块native15605446B×32、full-control1525330B×164及其余实际字节投影，每train约973275875B；原final每拍输出投影30600eval约540668448B，合计约2487220198B（2.32GiB）。这是容量估计，**不是新轨迹压缩率的数学上界**。完整成功产物预估3GiB。每worker在写前执行硬额度：训练各3GiB、eval2GiB，含partial和1MiB回执保留；bootstrap64MiB。原ArchiveWriter事务、NDJSON事件和checkpoint/metadata直接保存均接入额度。直接模型保存预留每文件16MiB并由RLIMIT_FSIZE限制实际文件长度，保存失败仍保留部分文件，不返还已占额度。ArchiveWriter失败回执仍走原事务，但只能用保留额度。此实现的源与fake已验证，真实训练尚未执行。

**全轮8GiB是父每秒采样及worker每事务检查的监测停止阈值，不是原子全局硬quota**；日志、reader和临时硬链接可能在采样间隔内暂时越线，按路径stat的逻辑字节也可能保守双计。磁盘启动至少10GiB、运行保留2GiB。超额INCOMPLETE且不删旧证据。历史df可用113053859840B，正式执行重采；依赖输入只绑定、不复制数GiB文件。

## 最小源集合和真实接缝

原科学源集合为9文件：纯helper及其测试、本页、三个生产入口及其对应parent/worker/reader测试。资源修订02新增stdlib `scripts/resource_guard42.py`及其纯测试，合计11文件；只修改父、parent测试和本页，不复制旧physics/controller或整个runtime：

| 新文件 | 最小职责 |
|---|---|
| scripts/run_b22_tracking_reward42.py | stdlib父，默认PLAN_ONLY，新plan/release/四个唯一worker（含零积分预检）/来源、资源、空间、总wall、partial/清理/最终门 |
| scripts/worker_b22_tracking_reward42.py | 新实际argv/GO/source；钉旧worker22/train22、learning21.stage2与archive/engine主体，仅版本、两reward臂、spec重定位、env/curriculum、预算和final元信息适配 |
| scripts/read_b22_tracking_reward42.py | saved-only训练/评测/比较；先重算原九项再按绑定α验证，原obs/action/native/controller/Adam/梯度/KL/coverage数学保留；评测仍原reward/评分 |

实际train env须在原WorldUpright._reward之后**恰好一次**调用helper；eval必须原env。旧worker22锁C22/stage1/B；Curriculum20锁精确env类型/reward_schema；read_training22锁九项公式；checkpoint21/rl16_learning11锁元信息；recipes22锁相邻spec路径。因此不能只改reader标签或外层reward wrapper假装接好。新final要真实记录训练reward provenance并由新schema验证，不能谎写旧reward过strictloader。初始B22加载继续原strict规则。

每个适配以旧完整sourceSHA和inverse AST列明，module globals/实际worker身份为C42，不全局替换旧SHA。闭包还钉实际原spec、B22 ZIP/metadata/probe、policy/Adam证明、recipes/curriculum/learning/checkpoint/heldout/readers、controller/99D builder、engine/GOT/ELF、archive、原评分及实际依赖栈。

freeze前完成集成审查与pure回归：错父代/错臂、α漏接/重复、spec迁移、reward总和→Monitor/rollout buffer真实接线、floor不筛臂、两final采齐、partial/早终止实际计数、final-only、零模型reader、wall/RAM/字节清理。三个入口已实现，尚无真实bootstrap/训练/独立读回结果；不能把source准备完成写成数值验收通过。

新worker明确钉原适配源码、B22四文件、原spec和实际MuJoCo完整双根包库存；父再绑定v27现有全部控制/依赖输入。`required_inputs42()`当前纯stdlib冷核返回**287项，共87886584字节**，saved-reader另钉3份原源码；四种阶段的实际默认会话均通过冷接口核验，没有导入NumPy或模型/物理框架。有限几何部分仍依赖本机历史work目录内原C21报告及12份模板，原件身份由固定spec/报告核验；这是本机可复现训练输入，**不等于fresh clone独立训练**，也不继承v27推理的`external_research_work_directory_required=false`。此轮不迁移或重写历史输入。

worker以原完整sourceSHA＋inverse AST锁保留真实物理owner主体，声明的接缝包括实际C42入口/回执、固定spec路径、训练Env/九项reward schema、warm-start stage2、独立final metadata/loader、真实两个actor/floor及额度计数。函数共享同一适配module globals，严格loader不借旧schema伪装。最初1024拍仅排除reward/reward_terms9后逐字段比对numeric，gaussian全字段比对，并核初始policy/Adam/actor/log_std与随机源；它发生首次更新前，不额外请求NN或积分。

当前四组组合 **171项pure/fake测试PASS（3.47秒）**：helper26、parent43、worker39、reader63；八份新Python文件Ruff PASS。没有模型推理或物理积分，不能据此称全链实训通过。helper测试只编译两份完整SHA钉住源码中的_reward，以synthetic五条actuator trace计算，不import原模块/创建环境。反例是“准确tracking但高度误差6mm”与“高度准确但vx误差0.005m/s”：原奖励偏好后者，α8偏好前者，证明它能区分这一取舍，不证明训练会找到改进行为。worker纯测试覆盖完整inverse AST、共享globals、实际额度写前拒绝/部分文件/失败回执、两floor失败仍采齐六任务、bootstrap构造异常吞没与懒绑定指针；父/reader另覆盖bootstrap回执与三份数值读回的实际接口、原奖励重算接缝、初态/首批配对、原六任务门和已接受文件身份。checkpoint测试从冻结保存函数提取实际manifest表达式：三份payload逐字节身份校验，metadata单独由SHA绑定，最终读回身份仍覆盖四文件和manifest；修复了首次训练后必然拒绝有效checkpoint的问题。这些不替代真实bootstrap或训练反馈。

~~~bash
rtk proxy env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest --noconftest -o addopts= -q tests/test_b22_tracking_reward42.py
rtk proxy .local-deps/bin/ruff check scripts/b22_tracking_reward42.py tests/test_b22_tracking_reward42.py
rtk proxy env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest --noconftest -o addopts= -q tests/test_b22_tracking_reward42_worker.py
rtk proxy .local-deps/bin/ruff check scripts/worker_b22_tracking_reward42.py tests/test_b22_tracking_reward42_worker.py
~~~

父入口默认仅打印计划：`rtk proxy python3 -B scripts/run_b22_tracking_reward42.py`。11份源全部提交并通过检查后，`--prepare`一次性写02独立release，仍不启动训练；`--execute`由上述唯一unit内Nice≥10的受控父进程调用，先核实际cgroup限额并完成短内存准入。正式输出一旦创建即禁止重跑或续训，保存读回失败也不重做物理。release与输出均为固定独立路径，不覆盖31～41或旧42release01。
