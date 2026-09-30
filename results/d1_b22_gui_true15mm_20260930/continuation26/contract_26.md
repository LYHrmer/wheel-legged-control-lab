# C26：完成原有限归档后的 owner 退出同步

主审：实际 gpt-6-astra ultra，/root/astra_mainplan19。用户仍要求完成 B22 GUI 与真实15mm单台阶交付。C24已完成zero/B两者真实15mm几何MuJoCo仿真资格，不再运行台阶。C23及C25所有已运行来源、预算、失败和记录保留。

C25执行了1102控制、5510正常native、2compiler。最终deferred归档pending为0，实际flush为11.251018191秒；主线程仍沿用旧正常退出路径 `thread.join(timeout=8)`，8秒后抛出owner仍活跃异常，跳过主线程负责的输入、owner决策、render及performance完整归档。preparation失败分支另有5秒join，并非本次失败的直接触发。C25的active时间13.623432071秒只可作为保存回执诊断；缺失完整输入/渲染和最终worker结果，不能恢复成GUI通过。此缺陷是延迟归档与已有退出同步未完整整合，源审阅和测试遗漏由本轮协作承担。

唯一修复：新worker把owner退出等待统一为真实preflight_ready.monotonic_s加原close_s=115秒的绝对期限。等待时只用剩余时间，绝不从join开始额外增加115秒；原soft95/hard120及外层600秒不变。只能在owner线程确认退出后序列化共享证据，若期限到仍alive则明确失败，禁止并发序列化。准备失败路径也使用同一绝对deadline。成功worker主回执新增有限join起止/期限和alive=false证明，独立reader核其与实际preflight及完整归档闭合。没有第二物理线程、新框架、控制或GUI数值改动。

执行前必须先在真实非daemon线程上验证这个实际join helper：以超过原8秒的8.2秒（超过原8秒）延迟写入/完成，确认主线程等待并只在完成后写证据；另以短绝对deadline覆盖alive超时及禁止序列化。测试不得加载模型或物理。C25 Writer类AST保持相同，复用已经通过的完整保存数据回放和归档故障测试，不重新执行大回放。新增reader只改C26来源/合同身份和实际join证明；全部独立数值、native、归档、事件、性能和停车算法保持。

本合同新设一次独立C26键盘worker：B22同一冻结ZIP+C18，99D/16D，seed231002，yaw_1p2；最多1200控制、6000正常native、2compiler、1PPOload、3torchload、32probe加最多1200控制actor rows，无训练/优化/模型保存。既不使用C23剩余195控制，也不续跑C25。实际开始之后源冻结、无自动重试；全部未用预算关闭。

C25最终预注册27项OS事件完整继承：Escape仍1100，510 W释放仍唯一OS bookkeeping清理，要求26个有意义实际callback及完整焦点/自动释放/重新arm证据；不新增505、不再运行OS探测。530起0.35秒publisher pause、单次R reset及第二次拒绝保持。原RTF≥0.8、FPS≥8、poll及snapshot age p95≤250ms，全部实际active wall包括R，不减归档以外的实际控制时间；最后100实际控制raw/servo forward/yaw全部0、mean|bodyCOM vx|≤0.04、mean|yaw rate|≤0.05、clearance std≤0.02。无事后选窗或门调整。

归档仍单owner、gzip9、原atomic事务，≤64任务、≤2 episodes；active结束后全部持久化，pending0、无错误、所有manifest/payload完整、flush前后物理计数不变。新worker必须在原close期限内退出，host必须关闭所有自有进程、来源无漂移；独立reader完整通过并检查真实PNG按钮可读，才可交付资格通过。继承C23真实600/600脚本24项逐字节同轨证据，仅作经版本化序列化/退出变化审阅后的继承，不声称C26另跑了同轨脚本或人类键盘。

分工：Sol仅新增worker退出同步helper及必要纯测试；本Astra拥有新合同/spec、C26 saved-only reader和source/result终审；root独占所有实际测试、模型、物理、GUI、打包与发布。C24无需等待GUI再次运行，也不因GUI失败撤销已经成立的台阶结果。最终入口由root新增版本化入口/manifest，旧冻结入口保留。此合同不预写C26成功、GitHub或CI通过，也不扩大RL优势主张。
