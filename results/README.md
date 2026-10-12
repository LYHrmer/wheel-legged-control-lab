# 实验索引

121 个结果目录，66 个自带 README。下面按**模型**分两层索引，因为两层的成绩不可互换：平面 6 状态教学模型用来把公式对到代码上，全机 D1 才是能力结论的来源。表格区分已完成的对照与仍未通过的能力，**不把历史任务的成绩拼到当前控制器上**。

当前主线入口是 [82 维命令条件实验](../docs/locomotion_lab.md)；逐轮研究记录在[研究日志](../docs/research_log.md)。

## 平面 6 状态教学模型

3 关节（`base_x` / `body_pitch` / `leg_extension`）、2 执行器（`wheel_force` / `leg_force`）。

| 记录 | 可以检查什么 |
|---|---|
| **[三方外环对照](benchmark/)** | **README 首屏那张表的出处**：LQR、约束 MPC、2 维有界残差 PPO 在 nominal / push / mismatch_delay 三场景的 6 指标（[metrics.md](benchmark/metrics.md)），以及 20 个配对种子的均值±标准差（[randomized_audit.md](benchmark/randomized_audit.md)）。读数前先看[复现说明](benchmark/REPRODUCTION.md)的 MuJoCo 版本容差 |

## 全机 D1（23 nq / 22 nv / 16 执行器）

### 主线课程对照（C14–C22）

这几个是文件数最多的目录，逐轮结论见[研究日志](../docs/research_log.md)。**课程收益门与 RL 贡献门分开判**：C22 的课程门通过，而 `RL_contribution_passed=false`。

| 记录 | 可以检查什么 |
|---|---|
| [C21–22 课程对照与保留评估](d1_main_route_execution_20260930_22/README.md) | B/zero pooled SSE 1.038768、成本比 1.040694、六项里 1 项达标；B 相对 A 误差 −13.50%、成本 −7.67% |
| [C17–18 七场景验收](d1_main_route_execution_20260930_17_18/README.md) | 七个固定场景全过，同时 RL 收益未达预注册门 |
| [C20 课程训练](d1_main_route_execution_20260930_20/README.md) | A 完整读回、B 几何前提失败及离线修复候选 |
| [C15–16 裁剪比较](d1_main_route_execution_20260930_15_16/README.md) | grouped RL final 的裁剪对照 |
| [B22 GUI 与真实 15 mm 台阶](d1_b22_gui_true15mm_20260930/) | 有界验收、计步与失败记录。"真实台阶"指真实碰撞几何，**不是实机** |

### A/D 横移线

**至今没有任何合格的实测横移速度**；C35 的静力推断已被 C36–C38 否证。

| 记录 | 可以检查什么 |
|---|---|
| [侧移 RL 结果](d1_ad_lateral_rl_20261001/README.md) | 64 周期训练与十场评估；物理资格通过，额外收益仅 0.10%–0.41% |
| [速度补偿与参考](d1_ad_velocity_reference_20261003/README.md) | C34 唯一案例 9.37 → 8.83 s，目标误差与保持率门失败 |
| [固定时序与价值诊断](d1_ad_lateral_diagnostic_20261003/README.md) | C32/C33：离线价值拟合不等于 RL 提速；30 mm 仍需 9.37 s |
| [固定参考前置验证](d1_ad_lateral_skill_20261001/README.md) | C30 六场配对、约 8% 固定参考提速与安全取消记录 |

### 其余对照与诊断

| 记录 | 可以检查什么 |
|---|---|
| [驾驶稳定性开发](d1_driving_stability_development/README.md) | 偏航消融、航向外环、制动回归和真实跳跃净空；独立85维学习任务与Opus实现出处，尚未完成驾驶验收 |
| [六模型开发/留出报告](d1_v3_locomotion_report/README.md) | 零残差与 PPO；三训练 seed；速度收益与高度、机械活动量代价 |
| [动作探针](d1_v3_action_probes/README.md) | 单腿与单轮动作含义、零残差启停和转向 |
| [延迟诊断](d1_v3_delay_diagnosis/README.md) | 同 seed 地面表示对照、反馈缩放、完整道路仍失败 |
| [九模型延迟对照](d1_v3_delay_ablation/README.md) | 单帧、四帧、随机化；固定预算，完整失败矩阵与 576 次更新复算 |
| [组合压力测试](d1_v3_unseen_stress/README.md) | 30 ms、两倍噪声标准差与 0.6 倍滑动摩擦；16 例均失败 |
| [执行器辨识移植](d1_v3_actuator_transfer/README.md) | 合成增益/时间常数/延迟拟合、独立激励验证、整机补偿得失 |
| [位置编码器辨识](encoder_identification_position_only/README.md) | 独立单轮练习：仅位置与 q+v 拟合、模型失配、未见激励及四臂 PI 前馈对照 |
| [编码器速度反馈](encoder_feedback_position_only/README.md) | 162 例；位置估速的 tracking/reversal 跟踪误差高于独立速度参考，stress 误差下降但限幅增多 |
| [摩擦与延迟辨识](friction_delay_cross_study/README.md) | 54 次拟合中 24 次延迟误判；low_speed 在额外摩擦场景的留出预测与控制误差均增大 |
| [同底座动作对照](d1_shared_action_study/README.md) | 六模型共用轮腿低层；共享两维与独立八维的逐种子结果，384 次 PPO 数学记录复算 |
| [预算存档接口检查](d1_budget_smoke_final/README.md) | 检查点参数是否对应更新记录，并验证迁移后的日志复算；正式实验已完成训练，评测中断后待续 |
| [PPO 训练诊断](../docs/budget_learning_lab.md#ppo更新时发生了什么) | 六条轨迹、3072 次更新；区分日志与固定子集，并检查一次 value loss 尖峰 |
| [测量与执行器延迟](d1_delay_channel_study/README.md) | 24 个零残差开发案例；20 ms 短检查通过，完整道路仍失败 |
| [同条件主线演示](../docs/demo_showcase.md) | 零残差与预定 PPO 的 60 s 对照片，真实窗口自动按键验证，人工试驾待填 |
| [同步采样检查](d1_v3_sampling_final_audit/README.md) | 采样相位、COM 与机身原点速度的区别 |
| [历史连续任务](d1_continuous_policy/README.md) | 45 维路线的固定预算与配对分析；不等于新主线 |

### 预算研究

| 记录 | 可以检查什么 |
|---|---|
| [预算曲线正式结果](d1_budget_study/) | 200 例、198 完整、192 达标；**增加预算未带来一致改善** |
| [动作消融分析](d1_action_ablation_analysis/README.md) | 共享两维与独立八维动作的留出对照 |

## 哪些留在 Git、哪些在 Release

协议、结果 JSON 和小体积图像留在 Git。整机逐步 CSV/NPZ/MJB 与源码归档由 Release 附件提供；
上表的位置编码器单轮实验较小，完整记录直接留在 Git。
见[下载与复算说明](../docs/reproducibility.md)。目录里的失败记录也是实验结果，
清理草稿时不删除它们。
