# 39：冷准入与首场完整执行，保存结果读取中止

39 的 MuJoCo 来源修复已通过实际冷准入，第一项 `flat_1p6_B` 完整执行 1600 个控制拍。随后保存结果读取器失败，第二项 `yaw_1p2_B` 未启动。父流程保持 `INCOMPLETE`，用时 347.437 秒，实际物理与 NN 总量仍为原回执的 `null`；尚无两场后端兼容性结论。

失败来自旧读取器的一项 Torch 前提。`worker22.mapped_libraries` 仅记录路径含 `/torch/` 或 `/nvidia/` 的映射；NumPy-only worker 保存的两份文件均为精确的 `{"paths": []}`。原 `script_sources` 要求 `mapped and all(path in sources ...)`，空列表在第一个条件被拒，异常文字为 `actual DSO left closure`。没有查到越界的动态库；不能按异常文字推断依赖清单再次缺库。

本次源码为 `bb23ef1d08435108e8bef70ca81e277175984671`。双 MuJoCo 根的 250 个代码文件及 6 份安装记录纳入来源校验；实际冷准入回执为 `COMPLETE`，环境、策略加载、NN 和 native 计数均为零。首场 worker 的保存账本记录了 1600 controls / 8000 normal native / 2 compiler native，以及 1601 次 NumPy predict、1632 行、零 PPO/Torch 加载，worker failure=null。父流程的物理已知小计为 1600/8000/2；其 NN 已知小计仍为 0，因为成功读回尚未生成。以上 worker 账本单列，不用它覆写父流程的未知总数。

180 条资源样本的最低有效可用内存为 6,573,592,576 B（约 6.12 GiB），没有触发 4 GiB 运行停止线。没有阶段等待，清理记录为 `remaining={}`，服务已退出。两项映射文件、父回执、冷准入、worker 及全部 60 份已保存工件的 SHA 见[机器报告](../reports/b22_numpy39_interruption_20261018.json)。本分析重核五份运行源与运行提交一致，未声称重新验签整个依赖清单。

修复应使用独立的保存结果读取流程：锁定旧 39 来源和完整工件，仅将 Torch 映射前提改为两份列表必须精确为空，保留模块来源、GOT、零框架加载、native 账本、完整控制轨迹及原评分检查。旧 39 源码和回执保持原样，首场不重复运行；缺失的转向场景需要另有明确预算的独立执行。本文不授予任务、完整矩阵、RL 贡献或实机资格。生成本分析新增 NN 和物理调用均为零。
