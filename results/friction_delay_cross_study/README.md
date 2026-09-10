# 摩擦、延迟与校准激励交叉实验

54 次位置辨识、270 个延迟候选、432 条预测指标、324 个单轮闭环案例，正式运行耗时
363.31 s。所有选中候选报告收敛，所有闭环运行至 6 s，未触发速度保护；
这些状态不代表辨识参数或运动质量通过。

额外 Stribeck 摩擦为 0 / 0.04 / 0.08 N·m，真实执行器延迟为 0 / 2 / 4 个 2 ms 物理步；
比较 standard 与 low_speed 两类校准激励，测量噪声 seeds 为 53 / 67 / 79。
两族共用未参与拟合的 validation/holdout 轨迹，PI 与控制中的独立合成速度噪声也配对。

无额外摩擦时，18 次拟合全部选对延迟；全矩阵有 24 次延迟选错。
42 次选择搜索边界不等于 42 次错误，因为实际延迟 0 和 4 本身就在边界。
失配对象上的 low_speed 校准没有改善留出预测或闭环误差，相关负结果全部保留。

协议见 [protocol.json](protocol.json)，计数和状态见 [summary.json](summary.json)。
生成时产物清单为 [manifest.json](manifest.json)，实际源码哈希和运行期间一致性分别在
[source.json](source.json)、[source_consistency.json](source_consistency.json)。
本说明为实验结束后添加，不属于原生成清单。

各 `friction*_delay*_seed*_standard/low_speed` 目录保存四条位置输入、全部候选的校准预测、
标称/拟合整段预测及三类命令的两臂闭环记录。复算和公式见
[学习文档](../../docs/friction_delay_identification.md)，独立分析输出见
[analysis.json](../friction_delay_cross_analysis/analysis.json)。

![延迟选择和留出预测](../friction_delay_cross_plots/identification.png)

![同 PI 的前馈对照](../friction_delay_cross_plots/control.png)

图中先逐条件配对再计算比值，线为噪声种子均值、误差线为观测范围。
三个噪声种子不是三个独立机器人。此处仅辨识输入为 position-only；控制反馈仍使用独立
速度通道，没有真机、整机或纯延迟补偿的证据。
