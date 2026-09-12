# 课程 heading 键盘修订证据

`heading_gui_check_01` 的真实 GLFW 自动检查通过：2158 个控制拍、21.58 s 累计仿真、23.74 s 墙钟、2 个 start 区域轨迹段，最后 Esc 退出，源码前后一致。完整目录原样保留事件、CSV/NPZ、编译模型、源码压缩、日志和五张窗口截图。源码压缩的 46 个成员逐一匹配协议 SHA，其中包含实际 `CourseKeyboardCommands` 类；当前 `verify_held_gui.py` 支持 `--heading`，作为检查脚本快照保存。

本版 W/S 请求前后运动，Q/E 修改并保留目标 heading，Space 请求已有受保护跳跃，R 明确执行仿真器重置到本区域出生点，X 取消请求，T/G 调整高度。A/D 侧步尚未实现；R 不是真实物理自救。检查记录经过跳跃状态机四阶段，不能据此称越障成功。自动检查也不能替代人工验收；此前人工失败记录继续保留在 `d1_course_manual_acceptance_failed`。

`exploratory/heading_adapter_probe_03` 包含平地与课程 rough 地面各 Q/E 四例，每例 1600 拍。保存的末态 heading 误差分别为 0.05132564、0.05132417、0.06563769、0.06548506 rad，均小于原探针断言的 0.08 rad，记录的 recovery 拍数均为零。原 summary 的 `fallen_steps` 实际统计的是 `safety == recovery`；本组按 CSV 独立复算了误差、末速度、位移和该计数，见 `evidence_summary.json`。

所有探索候选原样保留：`yaw_response_probe`、`yaw_response_probe_high`、`velocity_hold_probe`、`heading_adapter_probe`、`heading_adapter_probe_02`、`heading_adapter_probe_03`、`heading_damping_probe` 和 `heading_wheel_feedback_probe`。第一版 adapter 因 yaw 请求超过接口上限失败，第二版四例误差约 0.246/0.264/0.249/0.067 rad，未满足全部案例的误差要求。没有删除这些失败以只保留第三版。

`probe_scripts/` 保存八个原探针脚本。它们执行时导入的完整控制实现未逐次冻结，因此这些探针属于探索记录，不能宣称可精确重放每个历史物理结果。原脚本还含本机固定输出路径；复验须另建目录并明确记录所用源码。完整源码证据以最终 GUI 的 `rollout/source.tar.gz` 与协议为准，不能反向当作所有早期候选的源码。

新的 heading 控制器由 root 本地实现。此次新的 Opus 请求因 HTTP402 返回且输出为零，没有代码贡献；较早的地形显示和课程事件函数贡献按原证据组另行归属。后续侧步实现不属于本冻结组。本组使用原有课程控制器与 oracle 状态，policy 为 none，不增加正式 PPO 预算实验的案例数。

Git 提交本说明、小型 JSON、PNG 和原探针/检查脚本；CSV、JSONL、NPZ、MJB、源码压缩和原始日志完整保存在 Release 包范围。`delivery_manifest.json` 覆盖除自身外的全部文件，复制前后源与目标逐字节核验，未重写原始记录。
