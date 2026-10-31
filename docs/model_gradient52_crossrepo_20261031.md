# 52：在D1检验模型梯度，数据阶段提前停止

本次重新审视SuperMimic路线：51仅检验单步训练后的离线预测，不能代表多步训练、闭环反传及在线反馈。D1新增独立的8步模型拟合和32步可微闭环接口，并预注册一个差动轮速反馈参数；不修改wheel冻结源和v27绑定。

实际结果是**数据准入未通过，模型还没有训练**。D1接口准入完成128策略步，64个actor fixtures最大差4.77e-7，同快照32步分支完全一致。随后18个8秒采集试验只有13个完成、5个提前终止，正式执行以INCOMPLETE收尾。3个模型seed全部NOT_RUN，模型更新、策略网络更新和θ提案均为0。

五次终止均在原始偏航幅度0.3rad/s的课程中，6段该幅度有5段终止；0.1/0.2的12段全部完成。无动作噪声的开发和探针也有失败。数据课程超出既有资格任务的工作域，是本次设计需要修正的地方；幅度、平移和时序没有逐项隔离，不能下单变量因果结论。

含准入共6533策略步 / 65330 native / 32665 PD；正式峰值内存0.623GiB、最低有效内存6.104GiB，无OOM。首次准入因误填Torch版本在构造环境前拒绝，0步；独立附录只修正版本身份，原失败保留。18份完整/提前终止数据全部保留，不续跑，不用剩余13段替代原预注册门。

wheel本次新增训练和物理均为0，原演示与成绩不变。D1的[完整结果、失败清单和读回图](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/model_gradient52_results_20261031.md)、[机器摘要](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/reports/model_gradient52_results_20261031.json)以及[启动前计划](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/model_gradient52_plan_20261031.md)可核验。尚不能回答“模型梯度是否改善控制”，也不能由本次停止推断SuperMimic不适用。
