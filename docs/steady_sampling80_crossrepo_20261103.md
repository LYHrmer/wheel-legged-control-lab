# 80：D1转向稳态采样修复的条件对照

D1已将转向任务的actor展开起点110/260拍移到175/325拍，进入两个稳态转向窗口的一部分；前进、后退及停车起点保持配对一致。三个冻结模型、两臂各27个固定起点，共完成12次策略头更新，模型更新0。

120条短物理分支均完整、安全，15组原策略重复重放逐位相同。但三个模型的新采样Q相对旧采样分别增加 **0.004360% / 0.008247% / 0.009834%**，短筛查0/3通过。Q是两个32拍片段中较大偏航MAE的三reset均值，不是完整控制验收指标；按预注册条件，63个完整评测回合全部`NOT_RUN`。

实际总计9464 policy / 94640 native / 47320 PD；正式CPU59.237秒，峰值内存约0.751GiB。42项合成测试通过，源码及证据保持完整；未追加预算或选择seed。

- [D1完整结果与失败项](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/steady_sampling80_results_20261103.md)
- [D1机器摘要与原始文件SHA](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/reports/steady_sampling80_results_20261103.json)
- [批准的80方案](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/steady_sampling80_proposal_20261103.md)

完整模型资格仍为NOT_REQUALIFIED，没有新增完整控制性能结论。wheel只同步结果入口，新增训练、模型调用及物理步均为0；48不重跑，v27不重绑，冻结来源与默认策略不改。
