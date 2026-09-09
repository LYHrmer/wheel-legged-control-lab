# D1 采样模式：冻结源码后的耗时复跑

本页的 CSV 和源码归档随 Release 提供，见[下载与复算](../../docs/reproducibility.md)。

同步采样消除了记录中的位置时相偏差，但没有加速这条控制路径。12 个回合全部完成，每回合 400 个控制周期；原始 CSV 共 4,800 行。下表保留三次运行的耗时，编号顺序为 0 / 1 / 2。

| 反馈来源 | 采样模式 | 周期中位数，ms | 周期 P99，ms | 三次中最大周期，ms |
|---|---|---|---|---:|
| oracle | legacy_mixed | 1.270 / 1.249 / 1.249 | 1.581 / 1.398 / 1.376 | 1.914 |
| oracle | synchronized | 1.349 / 1.355 / 1.347 | 1.542 / 1.641 / 1.552 | 1.746 |
| IMU/encoder fusion | legacy_mixed | 1.897 / 1.854 / 1.865 | 3.904 / 2.571 / 2.067 | 4.456 |
| IMU/encoder fusion | synchronized | 2.043 / 1.997 / 1.941 | 3.179 / 2.403 / 2.227 | 7.026 |

耗时统计排除了每回合第一个周期，控制误差没有排除。全部 4,800 个周期中的最大耗时也是 7.026 ms。本次观测值低于 10 ms 控制周期，但桌面系统的有限次测量不提供硬实时保证。

| 反馈来源 | 采样模式 | 高度 RMSE，mm | 尾段速度 RMSE，m/s | 最大位置时相偏差，mm |
|---|---|---:|---:|---:|
| oracle | legacy_mixed | 9.015294 | 0.029133597 | 0.464796 |
| oracle | synchronized | 9.006972 | 0.023209162 | 0 |
| IMU/encoder fusion | legacy_mixed | 8.749582 | 0.041439534 | 0.483691 |
| IMU/encoder fusion | synchronized | 8.796144 | 0.039042204 | 0 |

每组的三次控制误差完全相同，因为它们重复同一条确定性指令。速度误差在同步采样下较小，fusion 的高度误差略有增加；不能据此宣称所有控制指标都改善。

## 测量范围

本次使用 i7-14650HX，24 个逻辑 CPU，Python 3.10.12、MuJoCo 3.12.0。BLAS/OMP/MKL 线程数均为 1。其他项目任务暂停了重计算，桌面程序仍在运行。启动前与退出确认后的负载记录见 [execution_context.json](execution_context.json)。

每回合仿真 4 s，物理步长 2 ms，控制步长 10 ms；前 0.5 s 的速度指令为零，随后为 0.2 m/s。三次重复用于观察计时波动，不是三个独立地形或随机种子。速度尾段严格使用 CSV 中 `time_s > 1.0` 的行，共 3,612 行；浮点时钟在边界附近的取值也保留，不能另外按行号截出 300 行替代。

计时从 `controller.compute` 开始，到所选反馈源的 `read` 结束。模型编译与 LQR 辨识不计入，CSV 写入和额外的审计真值读取也不计入。这是直接 LQR 控制器的采样模式对照，未经过完整的 v3 `D1ControlLoop`，也没有启用执行器通道或 PPO 推理。

高度误差来自积分后的 `qpos`。速度使用各采样模式发布的 COM/body 速度，legacy 的发布时相本身就是这项实验的变量。位置时相偏差是发布的机身原点与积分后 `qpos[:3]` 的最大差值；它为零不代表传感器估计误差为零。

## 数据与复算

[原始结果](../d1_v3_sampling_final/)包含 12 份 CSV，另有 [protocol.json](../d1_v3_sampling_final/protocol.json) 和源码快照。[source_consistency.json](../d1_v3_sampling_final/source_consistency.json)确认运行期间源码未变。

独立审计未导入仿真或 benchmark 实现，用标准库从 CSV 重算分位数，检查每一项汇总误差。52 个源码文件的归档 SHA256 全部匹配；审计时还核对了当前源码树的精确文件集合与内容。结果见 [audit.json](audit.json)，逐回合复算值见 recomputed.csv（`recomputed.csv`）。执行过的审计源码保存在 [audit_source.py](audit_source.py)。这些检查证明记录之间一致，不能单靠哈希证明实验数据的真实性。

旧 [d1_v3_sampling](../d1_v3_sampling/) 的一致性文件记录了运行期间 `wheel_leg_controller.py` 变化，因此没有并入本次结果，也没有把那份记录改成“通过”。

从仓库根目录复算，输出需选择尚不存在的新目录：

```bash
rtk env PYTHONPATH=src:.local-deps python3 scripts/audit_d1_runtime_benchmark.py \
  --input results/d1_v3_sampling_final --source-root . \
  --output /tmp/d1-sampling-audit-check
```

以后修改源码后，`--source-root .` 会按设计报不一致。此时省略它可以检查历史 CSV 与归档，但不能再声称历史结果对应当前源码。
