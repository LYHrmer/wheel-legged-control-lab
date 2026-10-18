# B22 42 修订04：真实训练后触及旧内存停止线

04已通过Env安装与真实训练I/O上下文，进入完整B22 warm start与PPO学习。约531.90秒后，父进程读到主机可用内存3,687,108,608B（约3.43GiB），低于旧4GiB运行停止线，整轮以INCOMPLETE结束。第二组训练和评测未开始。

[紧凑摘要](../results/b22_tracking_reward42_revision04_summary_20261018.json)绑定原source `c2353be6e0c8a63055e58a13b7438a7188b1f73d`、release与保存回执SHA；原输出`results/b22_tracking_reward42_20261018_04`不改，不恢复partial权重。

## 已有多少真实计算

最后完整保存的rollout为0025，明确记录26,624controls / 133,120normal native / 2compiler native。之后的失败学习回执报告27,000transitions、26次train与387次optimizer step，segment回执报告135,000native returned；相应事务manifest逐SHA匹配。policy、actor和Adam状态均已改变，这次不能称为学习前失败。

worker最终回执缺失，完整actual_counts仍是null。原父known_actual_counts为0，含义是未收齐它认可的worker终态回执，**不是没有消耗**。本页分别披露已保存rollout下界与更后的失败回执前缀；没有重做完整native/训练读回，不能将这些数字改写成整轮精确总账。

原02/03学习前已知构造积分共4次，04又观测到至少2次，旧尝试均单列。05是一份新的独立有界执行，最多96136controls / 480680normal native / 6compiler native，从完整B22与Adam开始；不沿用04未用余额，不将旧消耗归零。

## 为什么放宽资源门

触发时owned整组current为1,456,291,840B（约1.36GiB），已读kernel peak为2,032,533,504B（约1.89GiB），high/max/OOM/OOM-kill事件均0。停止来自主机可用量门，不是本任务触及3GiB硬限额。最后内核快照不等于退出后完整峰值。原父minimum字段4,676,329,472B只涵盖成功检查的样本，真正失败样本的3,687,108,608B单列，避免漏报。

用户明确要求“不用那么保守”。结合实际整组占用，05采用**4GiB连续10秒启动，运行保留2GiB**，继续使用整组MemoryHigh2GiB / MemoryMax3GiB / SwapMax0、Nice10、单线程及串行重任务。这给正常桌面负载波动更多空间，同时保留明确的停止线和任务硬限额；不会停止其他应用。

## 中断回执修复

04的supervisor回执存在，但只记了第二轮空清理；原监督器第一次清理的信号明细没有保存。原清理实现反复TERM，2秒后转KILL。worker已经打印soft-stop并写出失败学习/segment信息，最终回执却缺失；现有证据不能确定最后退出信号，也不能将“2秒收尾不足”的解释写成已证明事实。

05在新父入口中增加一次TERM及有界收尾窗口，再调用原有硬清理；记录首次与后续清理，异常也尝试写监督回执，避免第二次空清理掩盖第一次动作。另修正最低可用量先记样本再判断。原监督器、worker、物理与PPO数学不修改；完整计数未知时仍保持null。
