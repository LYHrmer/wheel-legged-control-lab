# 连续任务开发记录

四次运行使用相同的静态道路与 45 s 命令表。实际碰撞地形是 5 mm / 0.8 m 起伏和平滑 ±4° 坡，详见 [任务说明](../../docs/continuous_task.md)。没有中途 reset。

| 目录 | 状态/地面来源 | 偏航控制 | 时长 | 完成 | 质量 |
|---|---|---|---:|---|---|
| [zero_legacy_p5](zero_legacy_p5/summary.json) | oracle | P=5 | 37.61 s | 否，越界 | 未通过 |
| [zero_p2_no_integral](zero_p2_no_integral/summary.json) | oracle | P=2，I=0 | 45 s | 是 | 转向未通过 |
| [zero_pi_development](zero_pi_development/summary.json) | oracle | P=2，I=3 | 45 s | 是 | 通过开发门槛 |
| [zero_sensor_development](zero_sensor_development/summary.json) | IMU/编码器估计及测量支撑平面 | P=2，I=3 | 45 s | 是 | 左转高度 RMSE 26.23 mm，超过 25 mm |

这些都是零残差基线，没有加载 PPO。PI 参数与质量门槛是在这条开发路线上确定的，不用于宣称留出地形的泛化性能。两个完整视频分别对应 [oracle 基线](videos/zero_pi_45s.mp4) 和 [传感器基线](videos/zero_sensor_45s.mp4)，每个 45.05 s，含初始帧和最后一帧。画面由当次实际 qpos/qvel 生成，未重新运行控制器。

每个目录的 `manifest.json` 核验 CSV、状态数组、协议和汇总。`compiled_model` 指向共享的 `models/<内容 SHA>.mjb.gz`，同时记录压缩文件 SHA 与解压后原始模型 SHA。四次运行的物理模型内容相同，控制与估计方式在协议中单独记录。

`source_sha256` 是运行时的历史源码快照，不随之后添加 45 维偏航命令观测或传感器延迟支持而更新。原始轨迹与汇总保持当次生成内容。模型包装从每次一份 78 MB 的 MJB 改为共享约 34 MB 的无损 gzip，未改变任何物理状态；旧 MJB 与原清单在本机 `/tmp/d1-continuous-model-archive-mHW4mF` 暂存，可恢复。重新从压缩模型生成的 oracle 视频与压缩前视频 SHA 完全一致。
