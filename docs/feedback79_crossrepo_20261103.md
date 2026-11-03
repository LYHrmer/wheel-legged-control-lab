# 79：D1模型—策略反馈先导实验

D1独立仓库已实际执行模型梯度支持的策略网络更新和新数据反馈：三个配对学习器，9次策略头更新、300次世界模型更新，准入加正式共39824 policy / 398240 native / 199120 PD。六份最终策略先封存，再执行54个候选与9个共享基线回合；63回合全部满600拍，满足原任务、停车和安全阈值。

唯一变量为新策略数据到达后是否更新世界模型。refresh对frozen三seed等权平均J比为 **1.0000469901**，即略增0.0047%；两对变差、一对微降，反馈效果门未通过。六候选对共享原策略的稳定偏航MAE均值增加0.0567%–0.4016%，未达到原下降5%且各reset改善的要求，六个控制改善门全部FAILED。

这是一轮有限H32、零终值、无critic的SuperMimic思想先导实验，已完成模型→策略→物理反馈→模型→策略的循环，但没有证明完整方法的控制收益。原完整模型资格仍为NOT_REQUALIFIED，不替换运行策略。

wheel本轮只更新关联结果入口，新增训练、模型调用和物理步均为0。48不重跑，v27不重绑，冻结协议和演示策略保持不变。

- [D1完整结果](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/feedback79_results_20261103.md)
- [D1机器摘要](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/reports/feedback79_results_20261103.json)
- [批准的配对方案](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/feedback79_proposal_20261103.md)

实际正式CPU82.739秒、峰值内存约0.770GiB；CPU单线程、单环境及原资源监督器约束均保留。D1原始实现提交为`03adaa5`，计划SHA为`107cebce4a95190cfe4725debb8efe2692c81517d188e672eca9b940c7cfa4b5`。没有用单seed结果选策略或修改门槛。
