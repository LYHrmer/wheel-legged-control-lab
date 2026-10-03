# C32/C33：价值诊断与固定侧移时序验证

- [C32 value 诊断](../../docs/ad_lateral_value_diagnostic_20261003.md)：原 54 维空间可拟合这批实际回报；旧 value 数值欠拟合，具体训练原因尚未分离。
- [C33 物理结果与耗时诊断](../../docs/ad_lateral_timing_20261003.md)：8 场完整独立通过，最快 30 mm / 9.37 s，仅比原固定参考缩短 3.30%–3.50%，工程提速门未通过，没有新的 RL 收益证据。

C32 真实数据只执行一次拟合诊断。C33 实际新增 10,837 控制拍、54,185 常规 native 子步，另有 2 次冷构造调用；新侧移 actor/value/optimizer 为零。环境启动与首次 reader 来源检查失败均原样保留，修复后没有重跑已经发生的物理。

[最终物理审阅](continuation33/final_review33.json)、[描述诊断审阅](continuation33/diagnostic_review33.json)、[C32 最终审阅](continuation32/final_review32.json)分别给出结论与限制。初次合成诊断检查的预算偏差和代理执行只读检查的分工偏差均有记录，不隐去失败或事后回填未知时间。

这是字节保持的研究证据子集，含源码、拟合系数、预测、报告和失败回执，不含新的策略 checkpoint。完整 controls/native/states 留在本地；[清单](publication_manifest32_33.json)给出全部本地文件的大小、SHA-256 及未公开原因。公开子集不能独立重放完整 reader，也不是便携运行包。当前 B22 GUI 的 A/D 仍未开放，明显提速目标尚未完成。
