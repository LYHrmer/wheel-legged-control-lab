# GUI stability baseline — 2026-09-14

全部使用实际 `scripts.run_d1_course_drive.run` 和 MuJoCo，输入为逐控制拍的长按键集合，没有创建 GLFW 窗口。检查时没有运行中的课程 GUI。LQR + VMC、oracle、legacy contact allocation、legacy_mixed 测量时序、wheel velocity gain=.55；没有 PPO。

已保存 21 场、34,600 个真实控制 transition；初始每场 settle 2 s，physics dt=2 ms，control dt=10 ms。`validation.json` 在实现变更前检查每场 49 个源文件一致，三项重跑的全状态/扭矩逐数组完全相同。所有涉及 qpos 的位移均来自积分后真值；表中速度是 world-x 差分，不冒充 body-COM 速度。

## 长按 W 与松键

平地 drive 8 s、课程 drive 12 s；随后零键 4 s。诊断阈值在 protocol 中先行固定：无恢复接管，roll/pitch<.3 rad、yaw漂移<.15 rad、y漂移<.10 m、末2s速度与目标偏差≤25%，最后1s平均|vx|<.05 m/s。课程是否完整越障需要额外轮地几何判据；本表不认证通过障碍。

| 场景 | 档位 | 目标vx | drive末2s vx | drive x进展 | 松键4s Δx | 最后1s abs(vx) | 诊断未通过项 |
|---|---:|---:|---:|---:|---:|---:|---|
| flat/start | 1 | 0.30 | 0.310764 | 2.241932 | +0.148469 | 0.001999 | 无 |
| flat/start | 2 | 0.40 | 0.412514 | 2.945524 | +0.185677 | 0.001013 | 无 |
| flat/start | 3 | 0.50 | 0.498512 | 3.537917 | +0.274233 | 0.010862 | 无 |
| course/rough | 1 | 0.30 | 0.000166 | 0.231883 | -0.018693 | 0.000500 | steady_vx_within_25_percent |
| course/rough | 2 | 0.40 | 0.235927 | 2.938621 | +0.239715 | 0.037051 | steady_vx_within_25_percent |
| course/rough | 3 | 0.50 | 0.281338 | 3.463758 | +0.117574 | 0.005883 | steady_vx_within_25_percent |
| course/ramp | 1 | 0.30 | 0.606109 | 3.891640 | +0.173508 | 0.001184 | steady_vx_within_25_percent |
| course/ramp | 2 | 0.40 | 0.498306 | 4.333175 | -0.140819 | 0.058507 | last_second_abs_vx_below_0p05 |
| course/ramp | 3 | 0.50 | 0.482772 | 4.601123 | +0.211057 | 0.002991 | 无 |
| course/stairs | 1 | 0.30 | 0.000021 | 0.549373 | -0.020317 | 0.002974 | steady_vx_within_25_percent |
| course/stairs | 2 | 0.40 | 0.009195 | 0.921516 | -0.078382 | 0.053875 | last_second_abs_vx_below_0p05, steady_vx_within_25_percent |
| course/stairs | 3 | 0.50 | 0.470750 | 3.442573 | +0.497145 | 0.132945 | last_second_abs_vx_below_0p05 |

平地三档稳定；所有课程 yaw/y/姿态都在本轮诊断阈值内，主要失败是纵向卡阻、坡道超速及松键后的残余移动。rough一档 boost 1050拍仍停在第一批石块前；stairs一二档分别boost 950/828拍，末2s几乎不前进。不能再把“没有前进”归因于W长按未进入控制器。

## A/D 的短按与完整侧步

短按：t=2..3 s；完整：持续保持按键直到实际 side_active 首次退出，再释放，不预设完成秒数。释放后的模拟仍继续，保留后续轮式控制行为。

| 输入 | 交接时间 | 结果 | 交接时定向侧移 | 交接4s后保留 | 保留率 |
|---|---:|---|---:|---:|---:|
| A/short | 3.48 | failed: cancelled | -0.028647 | -0.000131 | 0.0046 |
| A/one_cycle | 12.56 | done: success | +0.037032 | +0.036403 | 0.9830 |
| D/short | 3.48 | failed: cancelled | -0.028574 | -0.000128 | 0.0045 |
| D/one_cycle | 12.56 | done: success | +0.037042 | +0.036410 | 0.9829 |

短按1s仍在支撑重心移动阶段，A起初实际向右，D镜像；cancel/abort后回到原位。完整10.56s侧步成功后，交接4s保留约98.3%的3.7cm位移。因此“已完成的侧步被legacy控制拉回”在本条件下被反证，不能据此盲目加入世界Y锚定。体验改善应先定义短按/松键是停止重复、完成当前落脚还是取消整步；不要默默把短按改成释放后还自动运动10s。

## 松键参考的一变量因果检查

`probe_release_reference.py` 只在第一拍零速度请求时将 controller._distance_reference_m=controller._distance_m；保持实际距离积分、其他memory、增益、物理参数、每拍步数不变。该干预在work独立探针内完成。

| 场景 | 变体 | 松键前ref-distance | 松键4s Δx | 最后1s abs(vx) |
|---|---|---:|---:|---:|
| flat_start_gear3 | unchanged | +0.226327 | +0.274233 | 0.010862 |
| flat_start_gear3 | anchor_once_at_release | +0.226327 | +0.148027 | 0.013388 |
| course_ramp_gear2 | unchanged | -0.151833 | -0.140819 | 0.058507 |
| course_ramp_gear2 | anchor_once_at_release | -0.151833 | -0.056776 | 0.034806 |

两对直到松键前（含松键前最后状态）的 qpos/qvel/time/torque 逐数组一致，基线重复也与 forward_02 全轨迹一致。正的参考误差导致继续前冲；坡道负误差意味着机器人已经超前，旧控制器会追身后的参考，导致回退。重锚降低残余位移但不保证立即停止。

机制位置：`hierarchical.py:_advance_position_reference` 将参考限制在实际积分距离±.55m；零速度只停止推进参考，未清误差。`CourseKeyboardCommands` 松键目标确实立即清零。`CourseSideStepDrive` 只拥有每拍唯一torque与唯一plant.step，是课程专用修复的合适接入点。

## 记录与复现

`forward_01` 第一条物理已完成，但work helper输出np.bool_导致JSON序列化错误；原partial文件保留，`forward_01_failure.json`如实记录，`forward_01_first_case_recovered_analysis.json`从已完成raw复算。旧helper原文为`probe_gui_baseline_run01.py`；修正版只做bool转换和明确绑定局部对象，`forward_02`首条物理轨迹与原始逐数组完全相同。

源码后来修复制动后重新运行主helper会测新行为；旧基线依据每场source.tar.gz/protocol源SHA/model.mjb保留，不可把同一路径名称视为同一版本。每场raw目录含states.npz、telemetry.csv、键盘事件文件、protocol、source archive、compiled model和manifest；输入计划在外层*.input.json。

```bash
PYTHONPATH=.local-deps:src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B /absolute/path/probe_gui_baseline.py --repo /path/to/repo --output NEW_DIR --subset forward
```
