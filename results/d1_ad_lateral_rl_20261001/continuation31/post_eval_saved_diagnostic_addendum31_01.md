# C31 保存诊断输入补充 01

本补充仅允许读取唯一 `train_01/batch_00.json` 至 `batch_07.json` 八个既有批次记录，用于原 64 周期、256 条真实宏转移的 reward 分布及采集时、更新前的 value explained variance 描述；不得读取其他批次、重放策略或产生新样本。

八批目标随采集数据及当时价值估计变化，不能称为固定验证曲线，也不代表最终 64 次更新后的价值拟合指标。它们已用于 `saved_learning_diagnostics_31.json` 的描述统计；本补充不增加训练、模型、优化、物理、GUI 或 native 读取权限，不改变原一次执行、900 秒、64 周期及其他输入/记录帽。

取消场只报告实际安全落脚/交接耗时和带符号位移；不输出 30 mm/取消耗时的技能速度，不将负位移或负 retention ratio 描述为成功侧移保持。原正式取消资格仍由冻结独立报告决定。

原 `post_eval_saved_diagnostic_contract31.md/json` 全部保留，本补充与其一同绑定诊断 source GO。
