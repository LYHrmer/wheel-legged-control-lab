# D1 68–69：速度响应拟合改善，总代价方向仍未修复

本页同步D1的三种子配对结果，wheel本轮新增训练、仿真均为0。两仓成绩分别报告；既有演示策略、冻结源、48归档及v27保持。

67已定位到首拍前向/侧向响应失配。68只读旧训练分支，全部24状态×2通道通过稳定性准入。69保持原状态和yaw目标，只增加固定权重的body-vx/vy导数损失；三个control最终参数SHA精确复现64 active，再完成所有配对评测。

|模型seed|测试body响应RMSE降幅|正式body方向 /54，control → active|H32代价方向 /45|H1代价方向 /27|
|---|---:|---:|---:|---:|
|64001|64.86%|45 → 50|39 → 40|25 → 23|
|64002|84.73%|9 → 53|39 → 40|25 → 22|
|64003|64.01%|45 → 53|41 → 41|26 → 26|

全部54个body通道的四尺度原始符号在三个active上均正确，正式缺口分别为4、1、1个vx通道预测幅值低于1e−5。3078/3078项AD/FD自检通过，不能将幅值失格解释为求导不稳定。27个状态包括旧56、旧64及全新693xx各9个；仅新693xx的RMSE也分别降低46.28%、77.86%、49.77%，未用于训练或调权重。

完整修复未通过：yaw三对均26/27，六模型都在69301/260反号；代价新增5个幅值合格反号及2个幅值失格点。旧64303/110五个已知失败没有一例达到修复门。H32正式45点中保留旧64301/110的未知参考，不把44个已解析点改作验收分母。原始RMSE改善不等于通过无退化门，也不是控制性能收益；策略与控制参数未更新。

![D1三个配对模型的完整方向计数与响应误差](https://raw.githubusercontent.com/LYHrmer/ddt-d1-rl-lab/be49b0d90ff57b2a3fc88e04a21d31042ba7e727/docs/assets/body_jvp69/body_jvp69_paired_summary.png)

D1正式执行完整：6000模型更新，4680policy/46800native/23400PD；CPU111.863秒，峰值0.7914GiB，最低有效可用4.0744GiB。738项输入与1033份封存产物复核通过。没有partial、重试、OOM、新增训练物理或GPU并行；Nice10、单CPU单线程、MemoryHigh2GiB/MemoryMax3GiB/Swap0保持。

证据固定到D1结果提交`be49b0d90ff57b2a3fc88e04a21d31042ba7e727`：

- [68训练标签准入](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/be49b0d90ff57b2a3fc88e04a21d31042ba7e727/docs/body_response68_results_20261101.md)
- [69完整结果与七例新增失败](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/be49b0d90ff57b2a3fc88e04a21d31042ba7e727/docs/body_jvp69_results_20261101.md)
- [69执行前计划](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/be49b0d90ff57b2a3fc88e04a21d31042ba7e727/docs/body_jvp69_plan_20261101.md)
- [完整机器报告、资源与原始SHA](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/be49b0d90ff57b2a3fc88e04a21d31042ba7e727/reports/body_jvp69_results_20261101.json)
- 运行源码`ae1d485`的[专用72项测试](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/41298532861)及[静态CI](https://github.com/LYHrmer/ddt-d1-rl-lab/actions/runs/41298532868)均成功。

下一步应依据保存的完整分项轨迹区分响应幅值、代价基点和多步传播问题；本轮证据不足以把残余误差归为单一机制。wheel v27 SHA256仍为`d3c38429362e82745179f2c526b2ee52d805526df390d52346c805e5a51fff09`，未重绑。
