# A/D 真实 GLFW 侧移自动验证

GUI07 的五段自动键盘试验和独立审计均通过。测试使用私有 Xvfb 显示、llvmpipe 软件渲染和显式 `--render-quality low`，共 4230 个控制拍、42.30 秒仿真时间、282.49 秒墙钟时间。该次测试没有达到实时速度；这些结果不代表人工试驾或硬件验收。

| 按键 | 侧移完成耗时（仿真秒） | 实际侧向位移（cm） | 前后漂移（mm） | yaw 变化（rad） |
| --- | ---: | ---: | ---: | ---: |
| A | 10.57 | +3.714 | -6.058 | 0.005089 |
| D | 10.57 | -3.721 | -6.063 | -0.004875 |

独立审计从记录的 qpos/qvel 重建选定时刻的接触几何，确认 A/D 两个方向均通过四条腿依次离地、落地实现位移；结束时四轮接地，机身速度低于 0.04 m/s。对应轮腿最低点离地高度及姿态、速度详见 [audit_report.json](audit_report.json)。本控制器读取仿真真值，目前只验证平地；没有坡道、碎石或台阶侧移通过的结论。

五段依次为 A 完成、D 完成、抬腿时松键并阻止跳跃、抬腿时真实 X11 失焦、正常跳跃及 Q 航向调整。两次取消均持续由侧移控制器落脚，再切回轮式驾驶；取消段没有执行跳跃。最后一段完成跳跃并产生航向响应。R 是显式重置到当前区域起点，五段初态逐一核验；它不是物理翻身自救。

低画质实现由真实 Claude Opus 会话 `21e03aa8-6ac3-4a69-8853-39ea981442dd` 产出，原始补丁和应用补丁保留在 [provenance](provenance)。本地修正了补丁文件头和一行说明文字；验证由本地实际运行完成。低画质关闭窗口 MSAA 与场景阴影、反射，保持几何、窗口尺寸及物理/控制时间步。A/D 快步控制器的 Opus 实现与后续动力学校正见 [此前开发记录](../d1_side_step_development/README.md)。

同一二进制模型、同一 viewer 源码的独立五帧对照（包括首帧）中，normal 平均 0.3077 秒/帧，low 平均 0.0927 秒/帧。两组均未推进物理（0 步），qpos 和仿真时间不变。该小样本测量不等同整程实时性能，见 [render_benchmark.json](render_benchmark.json)。相关 21 项既有回归测试与目标文件 Ruff 检查通过，见 [targeted_qa.json](targeted_qa.json)。

生产源码的逐文件哈希记录在 [protocol.json](protocol.json)，试验内源码保持不变；独立审计核验了源代码归档和原始输出 manifest。测试 harness 单独保存于 [harness](harness)，本目录全部文件的哈希见 `files.json`。Harness 保留试验时路径，用于追溯实际执行内容；原始 work 中的布局和本目录的收集布局不同。

此前 GUI01..06 的失败记录没有删除或覆盖，现有 summary/failure 文件与哈希见 [prior_attempts.json](prior_attempts.json)。GUI01 缺少完整结束记录；GUI02/04 有焦点取消；GUI03 是直接脚本导入错误；GUI05 是新 X server 尚无窗口管理器 atom 引起的 BadAtom；GUI06 因软件渲染缓慢在 7.14 仿真秒、232.67 墙钟秒时超时。这些尝试不计为通过。

本目录只发布小型证据。完整逐拍 telemetry、states.npz、model.mjb 和 source.tar.gz 保留在本地 work，大小和 SHA256 见 [large_artifacts.json](large_artifacts.json)；它们没有随本目录上传。v0.8 的 25 个冻结结果根及分片未修改。截图均来自隔离自动试验，不能替代用户实际操作验收。
