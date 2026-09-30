# C17 首个 yaw 单因素干预决策

2026-09-30，Astra。依据 root 已执行的纯保存数据诊断 `yaw17/saved_yaw_diagnosis_17.json`，选择一个对称的内部 yaw 请求上限干预，不追加 yaw 训练。本文件冻结科学因素与预测；实际场次、root预算及精确源集由B合同补齐后才能运行。

## 已看到的证据

grouped 的完整 hold RMS为0.13036747，原门为0.12。有效 nominal yaw 在51.75%的hold行被限在±0.6；所有hold行均无wheel target clip、wheel torque clip或wheel protection。

最主要误差来自负yaw的平台：101行servo=-0.3，effective nominal yaw全为−0.6，实际yaw均值−0.10198249，误差RMS=0.19915295，占完整hold SSE的58.9246%。正yaw首段平台51行也全部顶到+0.6，RMS=0.15818044。因而不能把主要缺口描述为仅发生在servo换向过程；平台上仍持续存在反馈请求受限和欠跟踪。

grouped 有偏负的wheel residual differential，完整hold均值−0.28828402 rad/s。但在上述负平台，它与负nominal同向（均值−0.31484274），不是反向抵消。这不能单独解释负平台上最大的新增误差；腿姿、轮载荷和状态差异仍可能影响实际响应。训练中对应高速大yaw邻域实际暴露为0，只是覆盖缺口证据，不优先于这个直接可干预的控制饱和点。

## 唯一因素

保持当前yaw反馈增益4：

`yaw_unlimited = servo_yaw + 4.0 * (servo_yaw - pre_body_yaw)`

`effective_yaw = clip(yaw_unlimited, -L, +L)`

baseline `L=0.6 rad/s`；唯一候选 `L=1.2 rad/s`。nominal wheel公式仍为 `(servo_vx-effective_yaw*lateral_foot_offset)/0.087`，后续全部RL leg/wheel动作、wheel PI、common P、支撑/阻尼与最终物理保护保持原样。

1.2为本批唯一预选对称上界。已存hold最大post误差约0.22362，对应补偿请求量级 `0.3+4×0.22362≈1.1945`；这只解释为何选择该量级，不当成pre-state的精确重算，更不预测修改后轨迹。实际实现和reader须用真实pre yaw独立计算unlimited request。保留明确有限上界优于无界移除clip，不开展多增益/多上限网格。

改变的是基础控制器内部允许的差动补偿请求，不是任务命令或验收容差。raw |yaw|≤0.3、servo slew±0.006/tick、wheel target±30 rad/s、wheel torque±12 Nm、原roll/pitch/接触/地图/停止门、hold RMS≤0.12全部不变。所有16个RL通道照常执行，不做符号/工况屏蔽。

## 最小实现与检验

只改yaw参数时，新adapter可以继承原preview/reset/compute，初始化后在自身 `_nominal` 对象上设置 `yaw_request_limit_rps=1.2`。新compute薄wrapper仅补充新adapter schema、L/gain、pre_body_yaw、unlimited/effective请求身份，底层纯扭矩算术不变。对新reader而言，新增独立nominal重算与真实pre-state绑定即可；将来若ramp确需改PI，再复制compute/math与独立recompute，不为yaw参数提前重写整个控制器。

root必要pure验证应覆盖：L=.6的新接入与旧公式在相同保存输入上一致；未饱和区L=1.2不改变请求；正负超限区映射正确且仍有上界；gate、RL向量和物理保护没有被更改。该验证不是新物理成功证据。

随后使用新development seed、同一grouped checkpoint、同一脚本与配对初态比较L=.6和1.2。主要预测是旧饱和平台SSE下降且完整hold RMS改善；有效请求应在旧±.6之外实际工作，而不由下游严重饱和接管。任一方向的安全/终止/速度退化必须保留；只有通过原完整任务门才进入最终组合资格。失败后不能直接加增益或再扩大L，须先根据该次完整轨迹重新定位。

本次Astra只读JSON/源码并写决策，新增测试、模型与物理调用为0。
