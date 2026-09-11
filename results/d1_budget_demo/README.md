# 82 维主线演示证据

[同条件对照片](comparison.mp4)左侧为零残差，右侧为预先指定的八维 PPO
（训练 seed 31000、最终 262144 样本 checkpoint）。两侧完整保留 60 s，
20 fps、1201 帧，包括初始与终止状态；没有按效果选择片段。

这是一张固定开发道路上的展示，不能代替正式多种子留出评测。

| 本案例指标 | 零残差 | PPO |
|---|---:|---:|
| 速度 RMSE，m/s | 0.03292 | 0.04239 |
| 偏航角速度 RMSE，rad/s | 0.03397 | 0.05118 |
| 净空 RMSE，mm | 11.759 | 6.651 |
| 回合完成 | 60 s | 60 s |

该模型在本案例降低净空误差，同时增加速度和偏航误差；不能写成全面改善。
原始轨迹分别在 `zero/` 和 `ppo/`，单片为 [zero.mp4](zero.mp4) 和
[ppo.mp4](ppo.mp4)。两份记录使用同一编译模型、初始 qpos/qvel、地形、
融合测量配置、随机种子、控制器参数和逐拍命令。

[pair_audit.json](pair_audit.json)检查正式开发第一案例的十项配置来源，
以及两份演示全部 6000 拍命令和时间戳。`comparison.json`检查所有渲染帧
对应的保存状态序号一致。`audit.json`另对三段成片完整解码并记录 SHA-256。
详细命令和来源见 [commands.md](commands.md)、[task.json](task.json)。

## 实际 GUI 集成验证

[窗口录像](gui_retry/window.mp4)通过 X11 向此次启动的 MuJoCo 窗口注入
按键；录像只抓取该窗口的客户区。它是自动 GUI 集成验证，不是人工手动验收。
窗口菜单保留原生外观，主线对照片使用无菜单的状态录像。

[gui_audit.json](gui_audit.json)从实际执行的 CSV 验证了前进增量、后退、
左右转向、净空升降、Space 停车、无输入后的运动命令失效、升降键不会恢复
已失效运动命令，以及 Escape 退出。[事件记录](gui_retry/events.json)
包含 15 次注入的实际时间、窗口 ID 和所属 PID。

录像完整解码为 132 帧、20 fps、6.6 s。Escape 使仿真在 5.05 s（505 拍）
退出，`stop_reason=keyboard_escape`；墙钟与仿真时间不同。因此 CLI 返回 1、
`completed=false`符合主动结束的记录方式，不能当作跌倒。

首次录屏因信号中断导致 MP4 缺少 moov，失败视频、日志和轨迹保留在 `gui/`，
当时的驱动为 `gui_check_first_attempt.py`。修正为 FFmpeg 固定时长正常收尾
后，在新的 `gui_retry/` 成功重录，未覆盖失败证据。

当前 82 维入口没有跳跃或横移，Space 表示停车。旧 42 维跳跃演示不属于
本次主线，以上全部为 MuJoCo 仿真证据。

## 验证脚本

- `audit_demo.py`：核对配对条件、GUI 命令响应、完整解码；`--gui-only`仅核对 GUI。
- `gui_check.py`：按 PID 确定自有窗口后注入按键、抓取客户区；运行需要桌面权限。
- `compose_pair.py`：检查帧索引后拼接左右画面，不重新做物理积分。
- `probe_renderer.py`：只渲染五个保存状态来测量软件／GPU 渲染耗时。

全部执行仅写入本目录。仓库源文件和正在运行的正式实验保持不变。
