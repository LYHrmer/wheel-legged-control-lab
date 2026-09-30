# C23 v02：同轨证明与停止资格的范围纠正

制定/审阅：实际 gpt-6-astra ultra，`/root/astra_mainplan19`。本修订在原 headless 600 控制完成、GUI 两段尚未开始时固定。原合同、source GO、worker、reader、运行记录和已冻结 runtime_manifest 全部保留，不覆盖、不重跑 headless。

原 GUI GO 将末 100 点 raw/servo 全零与 C22 停止阈值同时加给 600 点脚本，这是本审阅者未完成排窗推导的合同错误。脚本 raw vx=.6 于 [175,425)，release425，servo 每点最多下降 .005；满速释放约需 120 点，到 tick544 才归零。因此 ticks500..599 并非完整零指令停车窗。该不相容性由已冻结命令和 servo 算术决定，与观察到的误差大小无关。

原 `last100_stop` 阈值、窗口、原 `arm_qualification_passed` 判法全部不变；原 headless 的该门必须公开失败，后续 GUI 同脚本也不得宣称该窗停车通过。不得挑选最后 50/55 点替代，不修改 release、不续步，也不把旧失败回执改为通过。

本轮交付证据分为两项：

- **渲染数值独立性**：原 headless 与唯一剩余 script_gui 使用相同 seed231001、完整五数组/隐藏控制初态、raw .6 [175,425)、600 controls。各自完整记录有效、物理安全、真实 B22/99D16D 及全部 native 计数闭合；GUI 性能保持原门。逐点比较全部状态、观测、动作、C18 控制、力矩和 3000 native 记录。此项不授予末100停车资格。
- **实际输入与停止交付**：保留原 keyboard_gui seed231002、原不超过1200 controls、原27项 XTest 表及唯一530起 .35秒 publisher pause。W/Q/E、Space、X、按钮Stop、focus、TTL、R一次及第二次拒绝、退出均须实际记录闭合。原表850后停止、860释放，1000发送Escape，留出原定末100停车窗。末100必须全raw/servo forward/yaw零，并满足 mean|body COM vx|≤.04、mean|yaw rate|≤.05、clearance std≤.02。任何不足/失败均不能声称GUI停止交付完成；不补步、不换窗。

执行顺序：先用原 reader 保存原门结果（若 reader 软件错误，保存错误并据实标明）；再由新 `reader23_v02.py` 完整重核保存数据，分列原资格结果、记录有效性、同轨执行前置和停止资格。headless 记录/来源/安全失败即不启动 GUI。script_gui 同轨、性能和记录均通过后才能启动原 keyboard 表；最终键盘所有门通过才能认定本轮GUI交付完成。

原 headless 的实际 active wall 为 7.609643493 秒，RTF=.788473，原 ≥.8 性能门为 false；不能因其无窗口而抹去这一实际结果。原 reader 的首次执行在来源名单 10 项与实际 15 项不符处失败；五项新增实际来源均已存在于原冻结 bundle 闭包，v02 改为精确 15 项逐一校验，而非放宽为子集。原错误回执保留，完整物理记录有效性等待 v02 完整读回。

剩余 GUI 使用独立文件 `gui23/run_gui23_v02.py`，唯一运行差异是 `live_status` 进度 JSON 从每 control 原子发布改为每 5 controls 或达到 control_limit 发布；线程共享 completed 仍每 control 更新。控制数学、模型推理、native 调用、输入/准备时序、原27项 XTest 表、全部逐点记录和窗口均不变。所有 XTest 目标 tick 为 5 的整数倍，实际驱动延迟仍由保存时间戳核验，不宣称零延迟。此版本差异必须经剩余 script_gui 与原 headless 的全部数值/native 字节同轨证明，不能仅凭静态差异断言数值相同。

纯 I/O 证据 `io_diagnostic_01/receipt.json` 只调用已冻结 worker 的 `replace_json` 200 次，平均 3.9734 ms、中位数 3.9318 ms、p95 4.0986 ms；600→120 次发布估计减少约 1.907 秒。此估算仅支持一次有界工程修正，不是实际 GUI RTF 或 FPS 通过证据。新 GUI 两段仍须实际满足原 RTF≥.8、FPS≥8、poll/snapshot p95≤250 ms，除原预注册 initial/final 静止展示外不扣除运行耗时。若失败，不增加本轮模型/物理预算。

新 reader 的来源由新 scope GO/reader宿主回执独立绑定，不能要求其伪装成已存在于旧 headless session 的冻结输入。其报告同时给出自身 SHA 与原 reader SHA，原末100和 RTF 结果不覆盖。新 `runtime_manifest_v02.json`、新入口 `scripts/run_d1_b22_gui.py` 与新 worker/reader 全部独立命名并封存；原入口、manifest、已执行 worker/reader 及其来源闭包字节不改。

总预算继续包括已经发生的 headless：GUI最多2400 controls/12000正常native/+6 compiler，最多3冷worker、3 PPO loads、9 torch loads、3×32 probe、2496 actor rows；剩余仅script_gui600及keyboard≤1200。所有训练/优化调用为0。C24已独立完成，其2400/12000/+2账本与GUI分开，不为演示复跑。

本修订改变的是错误的阶段依赖和证据范围，不改变任何已跑命令、门值或窗口，也不把旧失败解释为成功。最终必须披露原排窗错误、原 stop false、原 headless RTF false、原 reader 软件错误、单一 I/O 修改、同轨结果和键盘停止结果。此文不预写GUI通过、GitHub或CI成功。
