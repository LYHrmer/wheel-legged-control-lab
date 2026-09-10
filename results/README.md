# 实验索引

当前入口是 [82 维命令条件实验](../docs/locomotion_lab.md)。
下表区分已完成的对照与仍未通过的能力，不把历史任务的成绩拼到当前控制器上。

| 记录 | 可以检查什么 |
|---|---|
| [六模型开发/留出报告](d1_v3_locomotion_report/README.md) | 零残差与 PPO；三训练 seed；速度收益与高度、机械活动量代价 |
| [动作探针](d1_v3_action_probes/README.md) | 单腿与单轮动作含义、零残差启停和转向 |
| [延迟诊断](d1_v3_delay_diagnosis/README.md) | 同 seed 地面表示对照、反馈缩放、完整道路仍失败 |
| [九模型延迟对照](d1_v3_delay_ablation/README.md) | 单帧、四帧、随机化；固定预算，完整失败矩阵与 576 次更新复算 |
| [组合压力测试](d1_v3_unseen_stress/README.md) | 30 ms、两倍噪声标准差与 0.6 倍滑动摩擦；16 例均失败 |
| [执行器辨识移植](d1_v3_actuator_transfer/README.md) | 合成增益/时间常数/延迟拟合、独立激励验证、整机补偿得失 |
| [位置编码器辨识](encoder_identification_position_only/README.md) | 独立单轮练习：仅位置与 q+v 拟合、模型失配、未见激励及四臂 PI 前馈对照 |
| [同步采样检查](d1_v3_sampling_final_audit/README.md) | 采样相位、COM 与机身原点速度的区别 |
| [历史连续任务](d1_continuous_policy/README.md) | 45 维路线的固定预算与配对分析；不等于新主线 |

协议、结果 JSON 和小体积图像留在 Git。整机逐步 CSV/NPZ/MJB 与源码归档由 Release 附件提供；
上表的位置编码器单轮实验较小，完整记录直接留在 Git。
见[下载与复算说明](../docs/reproducibility.md)。目录里的失败记录也是实验结果，
清理草稿时不删除它们。
