# 81：D1目标函数对照完成，完整改善门未通过

D1已执行第二版单变量协议：三份冻结模型，各用相同六个稳态起点比较原六项成本C6与偏航目标Q6，固定一步改动。共六次实际策略头更新、零模型更新；方向生成D、独立短筛V及完整验收H互不重叠。

V短筛3/3通过，372条分支全部安全、15组原策略重复逐位相同，因此按预注册执行全部63个H回合。H全部满足原任务、停车和安全阈值，但三个Q6候选完整偏航相对原策略只改善 **0.231453% /0.219626% /0.019220%**，均不足5%；平均J分别增加 **0.141820% /0.109497% /0.036487%**。Q6完整偏航相对C6反而恶化 **0.223162% /0.126203% /0.238713%**，完整候选拒绝。

四尺度强性质亦未通过：Q6有33/36个组合满足严格双侧顺序，失败的主尺度没有被更小步幅替换。模型预测主候选9/9配对短Q下降，物理只有6/9下降；模型响应误差与短窗口收益推广到完整任务仍未解决，不能将内部AD/FD一致当作真实控制收益。

实际总计53528 policy/535280 native/267640 PD，正式CPU143.948秒，峰值0.854GiB，最低有效可用2.869GiB。45项合成测试及独立物理、策略审计通过；全部1823个原始文件126172535 B本机保留，没有异常或partial运行。

- [D1完整结果、失败项及限制](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/objective81_results_20261103.md)
- [D1机器摘要与原始文件SHA](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/reports/objective81_results_20261103.json)
- [D1批准方案](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/objective81_proposal_20261103.md)
- [SuperMimic等原始论文的采用边界](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/objective81_literature_20261103.md)

完整模型资格仍为NOT_REQUALIFIED，默认策略未替换。wheel仅同步结果入口，新增训练、模型调用、物理步均为0；48不重跑，v27不重绑，冻结来源保持不变。
