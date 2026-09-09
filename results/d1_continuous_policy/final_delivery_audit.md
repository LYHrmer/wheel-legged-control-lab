# 连续任务独立复核（2026-09-09）

三个 131072 步策略的实际训练记录、42 次评价和两份 45 秒视频可以追溯到冻结源码。质量门槛没有通过：18 次最终策略留出评价均完成，质量通过 0/18；同条件零残差为 0/6。PPO 的高度 RMSE 比基线增加约 3.34 mm，暂不替换基线。

本次重新执行原分析器，266 个实验产物 SHA、36 个归档文件 SHA 和全部 CSV 的 RMSE 重算通过。[补充审计](final_delivery_audit.json) 还逐行检查了 195000 个控制步中的误差字段与残差力映射，差值为零。每个训练种子包含 28 个完成 episode，共 126000 步；固定预算末尾另有 5072 步尚未组成完整 episode。Monitor 的回报与原始采样奖励按 worker 分段求和吻合，最大差 4.83×10⁻⁷ 来自六位小数舍入。

[真值污染测试](../../tests/test_d1_continuous_delivery.py) 保持传感器采集不变，只改评测用位置、速度和地面高度。两次各 8 步真实仿真返回的 45 维观测及关节力矩逐位一致，奖励和真值日志按预期变化。具体改动为机体 z 加 0.03 m，机体系纵向速度加 0.3 m/s，同时更新对应的世界系速度；评测地面高度设为 0.02 m。代码检查也确认，在线 LQR/VMC 和策略编码只接收估计状态；已知初始放置姿态、名义模型及理想接触开关仍是先验。此测试不证明估计准确或足以直接上真机。

两份视频重新全解码成功，各 901 帧、960×540、20 fps、45.05 s。实际查看了 PPO 32 秒转向帧和两份 45 秒末帧：策略 SHA 与 sensor-command45 标签可读，末帧保留 `quality_pass=False`。PPO 的 Fz 在这些帧中为 +20 N。视频使用记录的积分后状态，真值位置取样仍早一个 2 ms 子步；原审计对此已有精确检查。

补充审计没有重新训练，也没有再次运行完整 45 秒复演；它重新核对已有复演的原始状态数组和 CSV，两个记录仍逐字节一致。连续任务的旧训练日志没有保存 Gaussian 均值、标准差和 log_prob，所以不能事后从该文件复原 PPO 概率比。实际 raw sample、执行 clip 和奖励缩放可以检查。现有留出仅覆盖新的测量噪声序列与 45/50 秒调度，同一张道路没有构成新地形留出。

本次串行运行 17 项测试全部通过（交付测试 1 项、训练入口 9 项、分析器 7 项）。未修改传感器、控制器或训练源文件。以下命令均在仓库根目录运行；审计输出要求新路径，避免覆盖已保存的记录。

```bash
rtk env PYTHONPATH=src:.local-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests/test_d1_continuous_delivery.py tests/test_d1_continuous_training.py tests/test_d1_continuous_analysis.py -q

rtk env PYTHONPATH=src:.local-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/audit_d1_continuous_delivery.py --output /tmp/my_continuous_delivery_audit.json
```

原分析器本次输出在 `/tmp/d1-continuous-delivery-recheck-20260909`，与仓库中的 [independent_audit.json](analysis_sensor45_v1/independent_audit.json) 数值一致。补充审计源码见 [audit_d1_continuous_delivery.py](../../scripts/audit_d1_continuous_delivery.py)。
