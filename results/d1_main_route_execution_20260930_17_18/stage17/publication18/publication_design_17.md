# C17 增量 GitHub 发布设计（待资格读回后填写结果）

2026-09-30，只读设计。父包为 `results/d1_main_route_execution_20260930_15_16/`，其 `publication_manifest.json` 已列出 792 个精选 payload、原路径、字节数和 SHA256。C17 不覆盖父包，不重命名旧 grouped checkpoint，也不把正在执行的 primary 当作已通过。建议新建一个 C17 结果目录和一篇新中文报告，以父包 manifest 的 SHA256、仓库提交号和 grouped ZIP SHA256 连接两阶段。

## 发布时必须同时给出的裁决

首页先用两个独立字段报告：`fixed_script_task_qualification` 与 `RL_contribution_passed`。前者只在 primary 六任务及 mirror yaw 的 **grouped** 全部跑满且逐场通过原记录、安全、任务跟踪、真实坡面载荷/整轮清除、末段停止门，并且两个 worker 的完整独立读回及来源/预算闭合后，才可写“转向、完整坡道跟踪和六任务回归完成”。任何一场失败都原值显示，不四舍五入或跨场平均补偿。zero 在同一 combined 基控下逐场并列报告，但不代替 grouped 资格。

`RL_contribution_passed` 另用 primary 六任务的 zero/grouped 配对 drive SSE、逐任务/hold 和归一化力矩平方成本门判定；mirror 只列绝对及相对成绩，不事后改变共同任务集合。基础控制器的 yaw 上限及共同积分修复收益归基础控制，不记到 RL。即使固定脚本任务全过，贡献门仍可为 false。默认 GUI 资格保持单独的 `qualified_for_default_GUI=false`，因为 C17 没有把新控制器和 checkpoint 集成、验证到 GUI；既有阶段 13 旧 final 的限域 GUI 记录按原范围保留。不能用 C17 推出自由键盘、实机、15 mm 单台阶、侧移、跳跃或物理自救能力。

报告应提供每一场的 case/actor/variant、完整 horizon、pass/failure reasons、hold 均速/RMS、yaw RMS、坡面三个真实轮载荷与四整轮清除、停止、安全与成本，附原始窗口和阈值。B 的 baseline/yaw/ramp 是开发与单因素证据；C 的 combined 是确定性资格，不把 B 任务通过数并入 C。若 C 未闭合或失败，按真实状态发布失败/进行中，不生成成功标题。

## 最小完整增量选集

建议在新结果目录保留以下文件组；最终只选择已产生、哈希稳定的真实文件，缺失项不以占位成功记录替代。

1. **入口与身份**：新 `README.md`、新中文 `docs/main_route_execution_20260930_17.md`、逐场 CSV/紧凑 JSON、`publication_manifest.json`、本地全档 inventory。manifest 对每个复制件写相对路径、原始路径、SHA256、字节数；额外指向父包 manifest 的内容哈希和父包仓库提交，写明 grouped ZIP `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`。若父包同仓库可用，只链接其 `stage15/grouped_01/final_checkpoint/final_model.zip`，不重复一份同权重；若做独立附件，则只复制一次并重新核哈希。模型 probe/metadata 链接父包原件。
2. **决策与预算**：保存诊断结论和 v02 坡道诊断成功凭据，同时保留 v1 解析失败说明；`intervention_contract_17.md`、`qualification_contract_17.md`、B/C spec、精确 GO/plan、各 worker reservation、host/worker/clean-reader receipts、执行计数与来源闭合汇总、最终 Astra 审阅。B `development_summary_17.json` 和三个独立读回结果应作为 C 预条件证据，而非资格成绩。
3. **新增可审计源码**：`controller17.py`、`residual17.py`、`verify_control17.py`、`read_eval17.py`、`score17.py`、`recipes17.py`、`worker17.py`、`host17.py`、`eval17.py`、`offline_floor17.py`、`floor_bridge17.py`、实际调用的 `readback_host17_v02.py`、纯测试及运行器/测试凭据。记录每个 C17 文件的来源哈希；C17 源仍从父仓库的 `heldout15.py`、`rl16_heldout_score_08.py`、`verify_course_e_08_03.py`、`rl11/verify_short_rl16_training_11_04.py`、`upright11/world_upright_course_11.py`、`course_impl08/*` 与 D1 底层模块取依赖。用父包 manifest 或新 source-closure 清单逐个指向实际版本，不把一份单独的 C17 脚本称作完整可运行源。
4. **C 原始证据精选**：primary 和 mirror 各自的 session/identity、floor gate 与 clean-reader receipt、worker/host receipt、独立 readback JSON 和 stdout/命令 receipt；14 条正式轨迹各保留 schedule、case/reset receipt、initial_state、states、geometry manifest、ramp 的编译几何末态投影以及逐文件 manifest。至少保留支撑关键结论的少量真实 control/native 原始块，明确它们只是样本；缺少其余完整 gzip 时，不能声称公开子集可重新运行逐 tick 力矩、原生积分、contactForce 全链核验。其余全量原档留本地，inventory 给逐文件 SHA256/字节数及取得方式。
5. **父包关系**：公开页面链接现有 15–16 结论与 C17 新结论，保留旧 4/6、5/6、转向/坡道失败和旧 `qualified_for_default_GUI=false` 作为历史事实。C17 仅对本次 combined 控制器的固定脚本下新资格给出新结论，不回写旧报告或旧评估记录。

若以精简发布为目标，优先保留决策、代码、模型身份、逐场紧凑分数、真实初态/末态、独立读回与源/预算凭据；大体量原生块通过完整本地 inventory 明示保留。这样读者能核对公开的数值与文件身份，但仍需取得未上传的全量原档及冻结环境才能重复独立全链读回。发布包的大小和文件数应由最终 manifest 实测，不沿用 15–16 的约 97 MB 或 1.06 GB 数字。

## 可重现说明需写清的边界

- C primary 固定六个原 case，zero→grouped 配对；mirror 只反转 yaw 命令顺序为负 100/正 200/负 100 tick。两 worker 均用 combined（yaw 内部上限 1.2 rad/s、gain 4，common active 时共同积分误差与相应退绕方向）；观测 99D、动作 16D、deterministic grouped checkpoint 不变。B 的 baseline/yaw/ramp 与 C combined 的身份和预算分开列。
- 同一 case 的 zero/grouped 初态字段须经独立读回逐字节核对；通过后才可称为固定状态起点的配对比较。不同 seed 不能称为独立随机物理样本：`FullDriveCourseEnv` seed 只产生 measurement seed，plant 采用固定 spawn，当前 oracle provider 无噪声且 policy deterministic；B 换 seed 后原 yaw/ramp 成绩曾与 C16 对应 grouped 逐浮点相同。mirror 因命令符号改变构成新的固定脚本，但也不证明随机初态、地形分布或统计可靠性。
- 评分速度是 base body 惯性 COM 投影到机体系的前向速度，控制用 pre-control 值、评分用 post-control 值；不是世界 X 速度或整机所有连杆 COM。坡道几何通过和速度跟踪通过分开。力矩平方是归一化成本代理，不称能耗。
- 复核路径分层说明：公开摘要/CSV 可核结果；已公开 states、schedule、receipt 可核配对、窗口与部分数值；完整 `read_eval17.py` 需原 control/native gzip、编译绑定与冻结源/依赖布局，并在干净子进程运行，禁止在 live physics worker 内导入。环境变量、绝对路径与旧依赖版本记录在 session/source closure，现有工作台脚本不是即下载即运行的通用启动器。不要给出未经实际验证的一键复现实验命令。
- 发布前逐项核：C 两 worker 及 floor 全部结束、独立读回成功或失败状态准确、实际计步与 reservation 分离、原始 archive 哈希闭合、模型与 32 行 probe 身份一致、所有复制件与 manifest 一致、父包未改、中文/英文 README 与逐场 CSV/JSON 数值一致。任何缺失直接写入限制，不把成功的纯测试、仿真 reset 或 floor 当成任务资格。

本文件只设计增量公开内容；没有复制、提交、上传或运行模型/物理。实际结果、GitHub 地址和提交号应由 root 在 C 结束并复核后填入。
