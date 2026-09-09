# 原奖励单位对照组

本目录是 reward scale=1 的 mixed 训练对照。seed 7000、8000 各训练 32768 步，其他配置与[0.01 缩放组](../d1_terrain_reward_ab_scaled/README.md)一致。

| seed | 全程速度 RMSE [m/s] | 最后 1 s RMSE [m/s] | final critic EV | 达标 case |
|---|---:|---:|---:|---:|
| 7000 | 0.098441 | 0.075431 | −0.00110 | 13/24 |
| 8000 | 0.098681 | 0.074188 | −0.00095 | 13/24 |

每个策略都存活到 4 s。11 个起伏 case 的末段速度误差超过预先设定的门槛；不能把存活率当成任务通过率。

本组与缩放组的训练地形记录逐项一致，零残差评测 CSV 也一致。完整比较和图见[奖励单位 A/B 说明](../d1_terrain_reward_ab_scaled/README.md)，包含缩放后控制误差略增的结果，缩放组 seed 8000 的负 EV 也一并保留。

复算采用[第二版源码快照](../d1_terrain_tracking_v2_source/README.md)，训练参数见 protocol.json。学习入口为[第二版地形记录](../../docs/terrain_tracking_v2.md)。92 个运行产物的 manifest 保持原字节，本说明是运行后新增文件。
