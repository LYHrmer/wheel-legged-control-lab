# C38 NumPy actor 两场闭环兼容验证

C38 是新的独立诊断预算，不重跑 Torch，不重贴 C35/C36 标签，不授予完整矩阵、RL contribution 或实机资格。默认只给计划，0 NN/0 环境/0 native；源码先提交、输入 SHA 冻结后，根线程按现有授权串行执行。D1 重任务未结束前不启动。

```bash
rtk proxy python3 -B scripts/run_b22_numpy38.py
```

固定新输出为 `results/b22_numpy38_20261018_01`；显式 `--execute` 才进入执行。已有输出拒绝覆盖和重试。新 source closure 包含全部 8 份 C38 源码/测试/文档、冻结 v27 运行时、C36 完成导出束，以及选定 C35 参考回执与所有原始文件。C36 数值证明是已存固定输入的 SB3/NumPy 对齐范围；C38 不额外计算逐观测 NN oracle。

| 顺序 | 范围 | policy/control | normal native | compiler native | NN |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | 独立 cold binding bootstrap | 0 | 0 | 0 | 0 |
| 2 | `flat_1p6_B`，seed 181202 | 1,600 | 8,000 | 2 | 1×32 probe + 1,600×1 control |
| 3 | `yaw_1p2_B`，seed 181203 | 1,600 | 8,000 | 2 | 同上 |
| 合计 | 0 新训练 | 3,200 | 16,000 | 4 | 3,202 calls / 3,264 rows |

每个脚本都预测全部 1,600 拍，包括关门拍；不混入 C32 的 skip 优化。PPO load、Torch load、critic forward、训练调用均为 0。NumPy 使用已完成 C36 的原 B22 99→128 Tanh→128 Tanh→16 actor 数组，原 clipped16 再由原 gain seam 乘 float32(.5) 一次。PI、z、last effective action、stop、控制器、传感器、原生 owner 和归档数学沿用冻结 C34；派生源码逐 AST 锁住明确的新身份、NumPy 预测/计数/来源及 bootstrap 接口。

独立 bootstrap 直接启动新 worker `--bootstrap`，不经会预留 2 compiler native 的旧 host。session 的 compiler budget、NN budget 和实际计数全部为 0。它先验证当前父 PID、RUNNING journal、nonce、精确 argv、固定输出与完整 source proof，再载入固定 EPA/MuJoCo extensions 并检查实际 ELF/GOT binding；从不 arm、构造模型/环境、重置、调用 step 或预测。前后 native counters/target/phase、geometry cache 均为零/冷态，`rtld_next_original_step` 是已解析地址，不是一次物理积分。

回执保存实际 session 文件 identity，父与 reader 均用自己读取该文件的 identity 对比，拒绝外来冷回执。

两场实际运行也各自先做相同 cold proof。新 NumPy counter/receipt/origins/strict-load schema 明确与旧 SB3 ModelCounter 区分，禁止导入 Torch、SB3、OpenCV、JAX、TensorFlow。保留原 32 输入 probe，clipped16 和 half-gain 两项绝对误差都必须 ≤1e-6；`probe_actions_byte_exact` 只记真实值，不作为 gate。C36 导出 proof 的 source policy、数组、manifest、两次隔离数值 worker 和工件 SHA 必须相连。

运行启动可用内存至少 6 GiB、运行至少 4 GiB，外部 Nice 10，整轮最多 900 s；bootstrap 60 s、每场 worker/host 240 s、每个 saved reader/比较 100 s，并预留整轮 10 s 清理。所有 Popen 使用自有 session，先设 subreaper，任何失败都停止后续、清理自有后代；Popen 返回句柄前失败也执行 cleanup(None)。清理期间的 INT/TERM/ALRM 被保留为停止原因，不静默启动下一场。父总时长包含 source/provenance 核查、子进程与清理。

阶段间只在上个子进程已同步清理、再次确认无自有后代，且下一 stage 的目录/nonce/请求尚未创建时等待 RAM：4–6 GiB 每 5 s 采样，最多 300 s，全部计入总 900 s。到 6 GiB 后重验完整 source/provenance 与最后 RAM/时间样本，再唯一启动；低于 4 GiB、读值未知、信号、等待耗尽或父总时限均立即停止。每次均预留本 worker 加 reader 的完整窗口（240+100 s；bootstrap 60 s，纯 reader 100 s），剩余不足即停，不用等待越过时限。已经创建/启动的 stage 不重试，child 自身的资源拒绝也不会回到准入循环。

只读 reader 不再跑网络、不创建模型、不积分。原 C26 native/owner ledger 仅将旧 load=1、Torch load=3 的预期改成新 scope 的 0，其余 C/Python/native、线程、数据身份、拷贝租约检查逐 AST 保留；额外严格验证 NumPy schema、逐阶段实际 request/row、probe/control phase、无框架来源及新 load proof。原 C31 每拍 qpos/qvel、99D 输入/16D 实际 action、safe torque、PI/z/stop、5×native、传感器/接触/终止和任务评分全部保留；保存输出只做 float32 half-gain 算术和实际 action 的精确连接核验，不新增 NN forward。

比较只使用固定旧 C35 同两场完整 B 读回。旧 C35 外层仍为 INCOMPLETE、总 actual=null；这两份已验证场景只是历史对照锚，不变成新的 Torch 执行，不补齐旧总支出。新旧必须同 seed、初态数组、完整 reset state、编译 robot/course 和控制/评分窗口；只在完整时域比较 SSE 和归一化 torque cost（不是电耗）。

预注册 `backend_compatibility_passed` 要求：两场新 NP 记录都完成 1,600 拍，原 task/safety/native/controller/stop 门全过，且每场 drive-window SSE 与 torque_cost_sum 各 ≤旧同场景×**1.001**（0.1%兼容容差）。原指标完整列出，不承诺轨迹逐字节一致，不修改原 RL contribution 门。不完整时域仍保留失败记录，不能因窗口缩短得到小 SSE 而声称改善。工程 COMPLETE 与这个结果 flag 分开，始终 `qualification_granted=false`、不自动切换默认控制器或模型。

失败时 actual 未知保留 null；可信已完成场景及合法部分 native interval 单列 known，按唯一 stage 重建小计，不能重复计费。NN 未闭账时也保持 unknown，不能以 0 代替。新旧物理支出不相加冒称历史总量已知。

准备阶段只运行 stdlib/fake 及小型合成 NumPy 测试，未实际执行 native bootstrap、真实模型或任何物理。
