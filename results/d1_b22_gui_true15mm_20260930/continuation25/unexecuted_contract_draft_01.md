# C25：一次有界的 GUI reset 归档修正

主审：实际 gpt-6-astra ultra，`/root/astra_mainplan19`。用户仍要求完成 B22 GUI 与真实15mm单台阶交付。C24已经在MuJoCo真实几何仿真中完成zero/B两者16/16门，2400控制/12000正常native/+2compiler，不再运行台阶。C23实际2205控制/11025正常native/+6compiler、3个物理worker，原键盘资格失败；剩余195控制作废，不转借、不改旧上限。本合同另立C25的一次物理预算。

已保存数据证明：C23键盘1005控制、active15.672893989秒、RTF=.6412344783。reset在global723前后prepared时间相差4.511624040秒，其间54次GLFW poll及54帧持续重用control720；旧active185帧中55帧age>250ms，完整reader给snapshot age p95=3880.873ms，原门失败。该间隙定位了owner同步reset边界，但没有足够旧时间戳将全部耗时归为gzip。root实际对两份完整归档的纯重放给gzip9=3.590260522秒、gzip1=1.943354468秒，解压字节和事务完整性相同，节省1.646906054秒不足达到RTF门所需3.110394秒；压缩改级方案据此否决，不做物理试跑。

采用唯一修复：新增worker保留原gzip9及原atomic archive事务，在active区间把有限、独占所有权的纯保存payload任务放入FIFO，active_end之后由原owner同步完整写入。无新线程、无第二物理owner，队列不接触live model/data；物理每步采集、全部native/contact/controller/初态证据保留。入队回执必须明确pending，不冒充已落盘；native guard可记录未来manifest名称，但最终独立读回必须核每个实际manifest/payload、FIFO、pending0、无错误和partial，方可资格通过。最多64任务、2episodes、control rows≤session cap、native rows≤5cap；资格cap1200，用户手动运行cap仍≤6000。队列容量或flush失败立即作失败闭账，不重试。controls的JSON转换也移入延迟payload写回阶段，以免R仍阻塞于整段转换。所有原数据需按所有权转交或不可变快照封存，不得后续原地修改。

reset核心控制和物理状态仍同步执行，不减去其wall时间、不移动active时钟，也不人为刷新陈旧snapshot时间戳。RTF≥.8、FPS≥8、poll p95≤250ms、snapshot age p95≤250ms原门全部保留。结束后的有限归档写回仍受原close115s/hard120s及外层600s限制，归档未全部持久化不算运行成功。必要时仅可依据执行前纯测试修复新源码，实际开始后不能改冻结源或追加物理尝试。

输入修正限两处并在新物理前注册：原27项OS表新增505点W按下，共28项；原最后Escape按下从1000移到1100，其余事件、seed231002、yaw_1p2、530起.35秒publisher pause全部保持。原因一：focus_lost470已自动产生W释放，510再次释放没有GLFW回调；旧贪心reader误将590的真实释放（实际延迟42.095848ms）配给510，最终旧事件门仍失败。新reader以原[-50ms,+1s]时间界做最大基数、全局顺序、不复用匹配，同数量最小时差；保持同一poll中的400/405两个真实边沿。原因二：旧raw最后非零856，servo最后非零906，旧最后100窗905..1004恰含两点非零，停车门失败正确。新Escape1100增加固定停车余量，cap仍1200；末100定义和.04/.05/.02阈值不变，不从旧数据改选98点。

新物理仅一次B22/C18、99D/16D键盘worker，最多1200控制/6000正常native/+2compiler、1PPOload、3torchload、32probe+最多1200控制actor rows（总≤1232）；无训练/优化/保存模型。沿用原host/隔离X11/单owner/三槽snapshot/焦点TTL与真实按钮路径，运行旧相同模型ZIP。先由root完成纯归档与输入回归、冻结source GO，再唯一运行。全部28实际callback、所有输入安全路径、一次真实R reset和第二次拒绝、全部数值/原始native、物理安全、原性能与末100停车、真实按钮PNG均须通过。

C23已有v02脚本600/600的24项逐字节同轨证据只通过经审阅的保存序列化变更继承；不宣称C25另外实测脚本同轨。C25旧键盘诊断必须如实输出event/performance/stop false；source/完整性破坏仍硬拒绝，不把失败任务改为通过。最终可用入口新增`scripts/run_b22.py`，旧入口/manifest/worker/reader/GO/错误和截图保留，所有运行直接使用交付bundle。

分工：Sol新增最小延迟归档worker及纯归档验证；本Astra拥有合同、spec、独立reader与终审；root拥有所有测试、模型、物理、GUI及发布操作。冻结前纯保存数据测试不冒充实际GUI成功。本合同不预写GitHub/CI或C25通过，不扩大RL收益主张；台阶zero也通过，B22仍无已证明的普遍RL优势。
