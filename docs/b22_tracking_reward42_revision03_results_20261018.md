# B22 42 修订03：控制器修复通过，延迟导入配置失败

03实际运行51.42秒，bootstrap通过，修复后的控制器安装也实际通过。随后进入`runtime_support22.atomic_training_io`时，它延迟导入了原`curriculum22`，后者再导入原`recipes22`，从迁移后的runtime目录寻找不存在的`spec22.json`，因此在策略加载和学习前退出。

[紧凑回执](../results/b22_tracking_reward42_revision03_summary_20261018.json)绑定原source `7623e655ee615f6dc6e21f06858a1d0d44870643`、release与失败文件的SHA，并保留实际成功的`controller_installer42_proof.json`身份。这次失败不同于02的Env精确类型错误；不能把前一个修复说成无效，也不能把仅定义模块的bootstrap说成覆盖了完整训练上下文。

03的model API ledger为空，已知0controls / 0normal native / 2compiler native，整轮actual_counts继续为null。父与监督cleanup remaining均为空，worker退出码1。02与03的已知构造消耗合计4次，所有尝试精确总数仍未知；两个旧输出均保留，不续接partial。

04针对真实lazy import接缝，使训练I/O上下文使用本次私有适配的课程模块与同一配置，不向冻结runtime补拷配置，也不改旧源码。新增回归要实际进入和退出原上下文，检查写入接管及异常恢复。新执行上限96136controls / 480680normal native / 6compiler native；旧02/03已知额外4compiler另列。

03仍使用6GiB/10秒准入、4GiB外部余量运行线、整组2GiB high/3GiB max/swap0。外部观察器首次采样时进程已经结束，因此外部完整峰值和OOM事件未知；父最后可读owned快照保存在摘要中。本次直接异常是配置文件路径，不应归因为内存。
