# 2026-09-12 课程键盘入口自动复验

本目录完整保存三次真实 GLFW 窗口检查。`course_gui_check_03` 的自动检查通过：965 个控制拍、9.65 秒累计仿真、15.68 秒墙钟，6 段区域依次为 rough、ramp、stairs、bumps、jump、start；前后方向请求达到 +0.3/-0.2 m/s，J 请求后记录到 crouch、thrust、flight、landing 四个阶段，最后 Esc 退出，运行前后源码哈希一致。`report.json`、逐拍 CSV、事件和截图可交叉核对。

这是项目已有 D1LQRVMCController、oracle 状态的独立课程演示入口，policy 为 none。数字键按区域重置，6 段轨迹不连续；观察到跳跃状态机四阶段不能证明成功越障。上述通过状态属于自动窗口与输入检查，也不构成 PPO 性能结论。用户随后人工试驾仍未通过，反馈除 W 外其他按键效果不明显，并要求 A/D 平移、Q/E 转向、Space 跳跃、R 倒地自救。新的交互修订另行开展，本组冻结的是此前按键方案与自动检查证据。

两次失败原样保留：

- `course_gui_check_01`：XSendEvent 的瞬时按下/松开触发了重复区域重置，记录为 11 段，前向请求最高约 0.156 m/s，未生成通过报告。事件、截图和原始输出均保留。
- `course_gui_check_02`：数字键和 J 的离散按键间隔已调整，但窗口管理器尚未激活目标窗口，持续运动请求全程为零，未通过驾驶断言。
- `course_gui_check_03`：先请求窗口管理器激活并设置输入焦点，区域数字键与 J 保持 80 ms 后松开；这次生成了真实通过报告。`verify_held_gui.py` 是此次通过后的当前测试脚本快照；此前两次测试脚本版本没有单独保存，本目录不会用当前快照冒充旧版本。

`attempts_summary.json` 从保存的 CSV、summary 和 segments 提取计数与范围，并将执行者的故障诊断与直接观测分开。每次目录中的 `source.tar.gz`、`model.mjb`、状态、日志和失败材料逐字节复制，没有重新运行或重写。原始源码压缩包含其当时捕获的一个 `__pycache__` 文件，保留原始哈希。`delivery_manifest.json` 覆盖本目录除清单自身外的每个文件，记录 bytes/SHA-256，并核对复制前后源文件。

`code_contributions.json` 公开记录实际 Claude Opus 的模型、会话和地形显示模块贡献范围。地形网格只加入渲染 scene，贴合真实高度场；不是碰撞体。独立 `terrain_display/real_heightfield_audit.json` 对真实开发高度场做了 2340 次 MuJoCo 射线检查，最大高度误差约 7.9e-16 m，模型与物理数据未变化。课程障碍本身来自原有课程几何体，该高度场审查不代表对课程障碍的完整验证。

Git 保存说明、JSON、脚本、日志和 PNG；CSV、JSONL、NPZ、MJB、源码压缩包随校验和 Release 附件提供。可复验脚本依赖项目路径及 X11 桌面，运行时应指定全新输出目录；地形射线脚本写入其所在目录，复验前请复制到临时工作目录。较早的 `results/d1_keyboard_revision` 已冻结，未被此组修改。
