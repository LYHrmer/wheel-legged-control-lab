# D1 64：三对首拍方向改善，完整多步修复仍未通过

此页同步独立 D1 仓库的模型实验，不把结果并入 wheel 的控制性能。wheel 本轮新增训练、仿真均为零；48归档和冻结源保持，v27未重绑。

只改变辅助导数监督后48行的来源，固定前48行、原7200条状态转移、H8、损失权重及1000次更新。三个配对模型的注册首拍方向均从13/18到18/18：18点由9个旧56回归点和9个新643xx测试点组成，均未用于拟合，且每点四档扰动必须全部通过。旧593xx首拍回归另外均从6/9到9/9。

|模型seed|首拍方向 control → active|35个已解析总代价参考 control → active|
|---|---:|---:|
|64001|13/18 → 18/18|25/35 → 30/35|
|64002|13/18 → 18/18|27/35 → 31/35|
|64003|13/18 → 18/18|34/35 → 34/35|

总代价正式门仍为36/36；第36点64301/110因单侧差0.341845超过0.25而未解析，没有剔除或重判。三个active仍有5、4、1个明确方向错误，64002还新增64303/110反号。完整修复和整体配对收益均为 `REFERENCE_UNRESOLVED`，没有控制性能提升结论。首拍幅值比约0.207–1.908，方向正确不等于幅值或完整Jacobian准确。

D1正式任务完成6000次模型更新、4680policy/46800native/23400PD；CPU94.031秒，监督器墙钟106.647秒，峰值0.7899GiB，最小有效可用4.2384GiB，策略及控制参数更新为零。无partial、失败执行、重试或续跑。所有正负训练标签保留，62原覆盖假设REJECTED不变。

证据固定到D1提交 `685c5d37de1ab6136cc17788b28ee75cbe3b5f2d`：

- [完整结果与失败项](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/685c5d37de1ab6136cc17788b28ee75cbe3b5f2d/docs/mixed_sign64_results_20261101.md)
- [执行前计划](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/685c5d37de1ab6136cc17788b28ee75cbe3b5f2d/docs/mixed_sign64_plan_20261101.md)
- [机器报告与原始文件SHA](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/685c5d37de1ab6136cc17788b28ee75cbe3b5f2d/reports/mixed_sign64_results_20261101.json)
- [D1源码提交的54项实验测试](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/41291425442)及[静态检查](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/41291425472)均成功；运行绑定的源码提交为 `b70704670f1c5f723e2a50c86febc468d2292600`。

wheel v27 SHA256仍为 `d3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09`。两仓的失败和原始记录均保留，未将训练拟合或方向诊断当作部署资格。
