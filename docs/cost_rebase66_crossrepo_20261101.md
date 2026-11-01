# D1 65–66：偏航方向改善后，总代价响应仍需修复

此页同步D1的离线诊断，不新增wheel控制性能结论。wheel新增训练、仿真为0，48归档不重跑，冻结源和v27保持。

64三个配对模型的注册首拍yaw响应方向从13/18提高到18/18，但65检查全部六模型×九个新状态后发现，三个active的单拍总代价方向均只有8/9。偏航响应正确不等于总代价方向正确。

66固定自由模型、actor及状态/动作敏感度，仅将代价求值基点重定位到真实零分支轨迹。5个既有active H1/H32错误只纠正2个：两个32拍错误仍保留且误差扩大；64003 active还在H4新增一个已解析反号。全部54点state/action/previous的JVP差为0，共3456个总前缀、20736个分项AD/FD自检通过。因此，这组剩余方向错误不能仅靠修正代价求值基点解释；后续应检查真实轨迹上的一步联合响应。

66用时28.118秒CPU，峰值0.4800GiB，0物理步、0模型/策略/控制更新。完整执行，无partial或失败执行；所有旧失败和未知参考仍保留。真实未来数据只用于离线定位，不能把这个oracle检验当作部署修复，64完整资格仍为REFERENCE_UNRESOLVED。

证据固定到D1提交`f403053e1e1b36fb82183974ce7f385023ad4f69`：

- [65全部九点梯度图及分项结果](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/f403053e1e1b36fb82183974ce7f385023ad4f69/docs/cost_prefix65_results_20261101.md)
- [66结果、失败项与资源回执](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/f403053e1e1b36fb82183974ce7f385023ad4f69/docs/cost_rebase66_results_20261101.md)
- [66执行前计划](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/f403053e1e1b36fb82183974ce7f385023ad4f69/docs/cost_rebase66_plan_20261101.md)
- [66完整机器摘要和原始SHA](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/f403053e1e1b36fb82183974ce7f385023ad4f69/reports/cost_rebase66_results_20261101.json)
- 运行源码`4ffa8d8`的[专用测试](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/41294131965)及[静态CI](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/41294132015)均成功。

wheel v27 SHA256为`d3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09`，未重绑。
