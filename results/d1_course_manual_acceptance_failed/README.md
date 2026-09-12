# 课程入口人工试驾：未通过

本组保存用户对旧课程键盘方案的实际试驾失败。用户反馈除 W 外其他按键效果不明显，要求 A/D 侧移、Q/E 转向、空格跳跃、R 倒地自救。此反馈针对旧方案；新的交互修改和复验应使用独立目录。

原始记录共 60000 个控制拍、600 s 累计仿真、679.83 s 墙钟，只有一个轨迹段，以 duration 结束，源码前后哈希一致。CSV 第 15472 拍（累计仿真 154.72 s）首次进入 recovery，合计 normal 15341 拍、degraded 130 拍、recovery 44529 拍。出现 recovery 状态不能证明已成功自救；人工结论仍是未通过。

记录中 ready/crouch/thrust/flight/landing 分别为 57480/525/315/735/945 拍。状态机经过跳跃阶段不等于跳跃或越障任务已成功，持续运行到设定时长也不等于驾驶验收通过。本课程使用项目原有 LQR/MPC 与 oracle 状态，不属于正式 PPO 预算实验的 200 个评测案例。

`rollout/` 完整复制自 `results/my_course_acceptance_20260912_01`，保留源码压缩、编译模型、逐拍 CSV、NPZ 状态及键盘事件。`human_feedback.json` 与 `process.log` 原样保留；`independent_counts.json` 是对已保存 CSV 的独立计数，不重新运行仿真。复制前后源文件哈希与目标逐一核对，结果见 `delivery_manifest.json`；清单覆盖除自身外的所有文件。

Git 保存本说明、反馈、清单、协议和日志；大模型、原始 CSV/JSONL/NPZ 与源码压缩通过带校验和的 Release 附件提供。历史自动窗口通过记录在 `results/d1_course_keyboard_revision`，不能替代本组人工失败结论。
