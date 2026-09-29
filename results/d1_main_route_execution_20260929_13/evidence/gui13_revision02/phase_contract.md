# GUI13 revision02：哈希与运行分阶段

原 GUI13 的纯 --check 实际通过，但观测到 76 秒时仍在读取源文件；其 sources() 对约 7.5 GB 依赖重复计算摘要。原源码、GO 及预检收据保留；原两臂尚未预留或执行。

revision02 只改哈希及计时边界：1 MiB 流式摘要，每次 sources() 使用同一个 actual 表核原训练冻结值，不重复全读。worker 仍独立重算全部 SHA，不以 stat 替代。

三阶段均有独立界限：从最外 launcher --run 开始至 worker readiness 的纯预检最多 240 秒，GUI 的私有 Xvfb 启动也在此窗；worker 完成纯预检后独占落盘 readiness（PID、session SHA、monotonic 时间、source 数量），host 核归属。readiness 在 import repair、策略/引擎导入前；其后执行 soft/close/hard 为 95/115/120 秒，包含导入、冷构造、load/probe、600 步和原子归档。worker 退出并清理后，全 SHA 后检最多 180 秒，单独报告。root 独立外层 host 总上限 600 秒，包括退出与私有显示清理；595 秒 TERM、599 秒 KILL，只清理出生身份匹配的本轮后代。

不改控制数学、seed 88813、600 controls / 3000 normal native + 2 compiler native 每臂、load 1、probe 1 批、predict 600、learn/save/optimizer 0，以及匹配、姿态、接触、RTF>=0.8、FPS>=8、poll/age p95<=250ms 门。headless 先独立读回合格才启动 GUI；无物理重试。不把纯哈希耗时排除出总 wall；它们只从执行窗口中分列，并仍有独立硬界限。

外层 host 在启动 launcher 前设 Linux child subreaper；被中途退出祖先遗留的 setsid 子孙会被该 host 收养，并按 PID 与出生 tick 纳入清理。240 秒预检起点由外层 host 统一传递，覆盖 Xvfb 阶段；worker 的 deadline 保持到 readiness 写入与 fsync 完成。超时触发停止后，只为已归属进程留至多 4 秒清理，单列于总 wall，不能再开始新控制。
