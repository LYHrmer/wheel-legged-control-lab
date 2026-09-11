# 延迟、短历史与随机化

这项练习接在[命令条件 PPO](locomotion_lab.md)之后，检查测量晚到时历史观测能补充哪些信息，
以及低层反馈怎样影响结果。

## 先分清两个时钟

| 通道 | 更新周期 | 本次训练范围 | 发生在哪里 |
|---|---:|---:|---|
| 整包测量延迟 | 10 ms | 0–2 拍，即 0–20 ms | 传感器包到融合估计器之间 |
| 执行器纯延迟 | 2 ms | 0–3 拍，即 0–6 ms | 请求力矩到实际关节力矩之间 |
| 执行器一阶响应 | 2 ms | 时间常数 0–6 ms | 与纯延迟不同的动态环节 |

`--measurement-delay 3` 是 30 ms，不是 6 ms。测量延迟也不是把策略动作推迟三拍。
后两种执行器效应怎样辨识和补偿，见[执行器移植练习](actuator_transfer.md)。

一个最简单的延迟模型是 `y_k = x_(k-d)`。控制器拿着旧状态计算当前力矩，误差反馈可能
加重已经发生的运动。提高控制频率、增加历史或降低增益都不是稳定性证明，需要闭环对照。
当前 fusion 还依赖已知初始放置姿态和理想接触开关，不等于真实机器人完整状态估计。

## 四帧观测是什么

基础观测仍是 82 维；`history=4` 按从旧到新拼成 328 维：

```text
[o_(k-3), o_(k-2), o_(k-1), o_k] → MLP → 8 维残差
```

相邻帧相差 10 ms，最旧到最新跨度是 **30 ms**，不是 40 ms。reset 时没有过去三帧，
因此重复初始观测填满；不能拿上一回合末帧补齐。每帧还包含命令、基线与控制记忆，
并非 82 个独立传感器读数。拼接没有额外读取真值，也没有引入 LSTM 或显式延迟估计器。

动手检查：给历史包装器连续输入标记为 1、2、3 的观测，手写每一步输出顺序；
然后检查 [history 测试](../tests/test_d1_observation_history.py)。第二个问题是：
若传感器一直迟到 20 ms，四帧里最新测量是否就变成“现在”？答案是否定的。
历史提供的是变化线索，策略仍需学习如何利用它。

## 固定预算对照

三组各用训练 seed 27000、28000、29000，每个模型 32768 个样本，固定取最后的 checkpoint。

| 组 | 观测 | 训练扰动 |
|---|---|---|
| h1 | 单帧 82 维 | 无测量延迟，理想执行器 |
| h4 | 四帧 328 维 | 无测量延迟，理想执行器 |
| h4_dr | 四帧 328 维 | 上表的测量与执行器随机化，执行器增益 0.95–1.05 |

所有组使用同一低层配置：轮速 PI `0.55 / 1.5`、偏航反馈 `4`、腿部 PD 比例 `1`、
姿态 PD 比例 `0.25`。这些是诊断后选定的研究参数，不是仓库默认参数。
姿态 Kp 和 Kd 一起缩放，不保证阻尼比不变；不能只给 PPO 组换控制器再归因于学习。

在两张开发道路、两个回合 seed（17/29）上分别测试 0、20、30 ms 测量延迟，
评测执行器均为理想通道。真正的测量噪声 seed 由回合 seed 派生，保存在
`episode.json` 的 `measurement_seed`，并非直接使用 17/29。
每组在每个延迟有 12 个策略案例，但只有 **3 个独立训练 seed**；公共零残差只有 4 个案例，不能复制
三遍当成 12 次独立试验。30 ms 超出这次训练的测量延迟范围，道路本身并非全新留出。

下面只复跑其中一个学习模型，输出目录须未存在。去掉 `--delay-randomization` 得到 h4；
再将 `--history` 改为 1 得到 h1。三组分别重训，不能把单帧 checkpoint 硬改成四帧。

```bash
python scripts/run_d1_locomotion_experiment.py train \
  --source imu_encoder_fusion --history 4 --delay-randomization \
  --wheel-kp 0.55 --wheel-ki 1.5 --yaw-feedback-gain 4 \
  --leg-feedback-scale 1 --attitude-feedback-scale 0.25 \
  --seed 27000 --steps 32768 --workers 4 --output results/my_delay_h4_dr

python scripts/run_d1_locomotion_experiment.py evaluate \
  --source imu_encoder_fusion --history 4 --measurement-delay 3 \
  --wheel-kp 0.55 --wheel-ki 1.5 --yaw-feedback-gain 4 \
  --leg-feedback-scale 1 --attitude-feedback-scale 0.25 \
  --policy results/my_delay_h4_dr/checkpoint.zip \
  --metadata results/my_delay_h4_dr/checkpoint.json \
  --output results/my_delay_h4_dr_30ms
```

## 怎样读结论

[九模型报告](../results/d1_v3_delay_ablation/README.md)保留了完整结果：20 ms 下 h1、h4、
h4_dr 分别完成 2/12、0/12、1/12，30 ms 下均为 0/12。这次没有得到跨训练 seed 的稳定改善。

先检查 60 s 完成数、终止原因、失败时长和实际地形曝光，再看完整回合的跟踪指标。
配对时固定训练 seed、道路和回合 seed，并核对派生的 `measurement_seed` 相同；
任一案例提前失败，不计算两者的完整时长 RMSE 差。
“失败前平均误差更小”不能替代成功完成任务。

还要区分两个问题：h1 对 h4 同时改变输入长度与第一层网络参数量，不能据此证明某种
独立的记忆机制；h4 对 h4_dr 同时随机化测量和执行器通道，也不能单独归因于测量延迟。
若要回答后一个问题，下一项实验应拆开两种随机化，而不是继续堆更多扰动。

下载原始数据后可以用[独立分析器](../scripts/analyze_d1_delay_ablation.py)复算完整矩阵。
它会拒绝缺模型、少案例或配置不一致的输入；检查当前 checkpoint 与 sidecar 的哈希，
但旧评测协议只保存了外部模型路径，没有逐评测模型快照，不能独立证明当时加载的模型字节。

```bash
python scripts/analyze_d1_delay_ablation.py \
  restored-artifacts/results/d1_v3_delay_ablation \
  --output results/my_delay_analysis.json
```

另有一个[组合压力测试](../results/d1_v3_unseen_stress/README.md)：30 ms 测量延迟、
两倍每样本噪声标准差、0.6 倍滑动摩擦，零残差与三份 h4_dr 策略共 16 例全部失败。
没有进入非平坦区域，因此不能称为复杂地形上的外推验证通过。具体时长、字段和重做命令留在结果页。

这次练习的验收问题：如果增加历史仍然摔倒，你能从记录判断是观测不足、动作限幅、
低层反馈失稳，还是训练预算不足吗？若证据不能区分，应写下缺少的对照，不能把猜测写成原因。

## 分开测量与执行器延迟

[24 例通道对照](../results/d1_delay_channel_study/README.md)不训练 PPO，固定同一轮腿低层。
在一条开发道路上，分别将测量或执行器延迟设成 0、10、20、30 ms，每组用回合 seed 17/29。
另外用无噪声融合器、延迟真值和即时真值作对照。

两通道的 10 ms 案例均完成 60 s；20 ms 案例在约 20–22 s 失败。
seed17 的 20 ms 案例此前通过了前 3 s 检查，那段命令只要求站立。
30 ms 的案例还没接触非平坦区域就已失败。比较时先看失败时刻对应什么命令、是否到了起伏路段。

测量时刻为 `t_m`，发布时刻为 `t`。同一估计值可分别与 `truth(t_m)` 和 `truth(t)` 比较：
前者检查采样时刻的估计误差，后者还包含数据过时的影响。延迟真值在采样时刻的误差可以为零，
却依然无法让当前闭环稳定。不能只拿一条总误差曲线判断融合器是否写错。

执行器 30 ms 对应 15 个物理子步。试从 NPZ 的 `actuator_trace` 检查：
`delayed_nm[n] = limited_nm[n-15]`，前 15 步由初始化零填充。
本实验一阶时间常数为零、增益为 1，所以 `applied_nm` 应与 `delayed_nm` 相等。
这些检查通过只说明声明的延迟模型被执行了，不能证明它符合真机。

再检查两种限幅统计：`torque_input_clipped_fraction=0` 只表示进入执行器通道的力矩没有
再被裁剪。控制器可能已经提前限幅，要同时查看 `controller_torque_clipped_fraction`。
后者以 16 个轴×已执行控制拍数为分母，不是“出现过限幅的帧比例”。

本轮无噪声和真值对照也失败，融合器与合成噪声不是这些失败的必要条件。
尚未隔离启动零填充队列的贡献，也没定位到某个具体 PD 分支；这项研究没有修复延迟稳定性。
脚本与复跑命令都在结果页，重新运行时使用新目录。
