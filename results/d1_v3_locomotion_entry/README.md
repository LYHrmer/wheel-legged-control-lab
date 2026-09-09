# 同一套环境里的程序运行和键盘指令

入口是 [run_d1_locomotion.py](../../scripts/run_d1_locomotion.py)。默认无窗口，传感来源是 IMU/编码器融合，策略输出全零。全零策略下，固定轮腿控制器仍负责站立和跟踪。旧的 sensor45、两维动作策略不能直接装到这个入口。

先在仓库根目录跑一段平地程序指令：

```bash
rtk env PYTHONPATH=src:.local-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 scripts/run_d1_locomotion.py --seconds 8 --output /tmp/d1-program
```

`--source oracle` 可换成仿真真值，记录会明确标出来源。默认传感噪声沿用轮腿探针的合成配置：陀螺仪 0.002 rad/s、加速度计 0.03 m/s²；编码器位置 0.0005 rad、速度 0.005 rad/s。这些数值没有经过 D1 真机辨识。测量延迟的单位是 10 ms，执行器延迟的单位是 2 ms，两个参数不能混用。

## 键盘模式

显式加 `--keyboard` 才打开本地窗口。需要可用的图形会话；这次没有实际启动窗口验收。

```bash
rtk env PYTHONPATH=src:.local-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 scripts/run_d1_locomotion.py --keyboard --seconds 60 --output /tmp/d1-keyboard
```

| 按键 | 指令变化与上限 |
|---|---|
| W / S | 每次 ±0.05 m/s，范围 -0.20 到 +0.30 m/s |
| A / D | 每次 ±0.05 rad/s，范围 ±0.25 rad/s |
| R / F | 每次 ±5 mm，净空范围 0.43 到 0.48 m |
| 空格 | 请求速度和转向归零，保持当前高度 |
| Esc | 结束本次记录 |

没有收到新的运动键事件超过 0.8 秒，速度和转向请求归零。高度键不会刷新这个计时，也不会恢复已过期的运动。按住键依赖窗口的重复按键事件。停车距离取决于实际控制响应，空格不等于硬件急停。当前不支持横移或跳跃。

新按键在后续 prepare 调用时被采样。已经交给策略的本拍指令不能被替换。窗口使用独立的模型和状态副本；拖动可视化中的物体不会给控制仿真施加外力。

## 记录和复跑

`telemetry.csv` 每行对应一次实际完成的控制间隔。`decision_time_s` 是动作决策时间，`measurement_time_s` 是这次决策消耗的测量时间，`time_s` 是动作执行后的物理时间。指令取自已执行的 transition，不取 step 末尾准备的下一拍。

`states.npz` 的 qpos/qvel 包含初始状态，随后一行对应一次控制间隔。观测也按相同顺序保存。动作有“策略返回”和“运行时限幅后”两份，不能把前者叫作未截断的高斯采样。每拍 5 个物理子步的执行器实际力矩另存为 `actuator_applied_nm`，单位 N m。

`protocol.json` 保存源配置和实际命令来源。Python 依赖另有源码归档，模型资产只记哈希。`summary.json` 的 completed 只表示走到时间上限，不能代替跟踪质量或地形通过率。

重放已执行的指令：

```bash
rtk env PYTHONPATH=src:.local-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 scripts/run_d1_locomotion.py --command-csv /tmp/d1-program/telemetry.csv \
  --output /tmp/d1-replayed
```

CSV 只提供指令，不会自动替你恢复原来的环境参数。复跑时，原 protocol 中的环境配置必须相同，seed 也不变。有策略时还要使用原模型和历史长度。保存的 CSV 不包含终止后那拍未执行的指令，重放只在最后的终止观测里保持末个目标。输出目录必须是新的，脚本不会覆盖已有实验。

## 策略加载的范围

`--policy model.zip --metadata metadata.json --history-length 2` 只加载可信的本地 PPO 文件。SB3 的模型序列化能执行 Python，不能拿这个入口试不明来源的文件。

训练端复用 `locomotion_checkpoint.write_checkpoint_metadata`，推理端调用 `load_locomotion_policy`。需要先 reset 环境。校验包含模型哈希，也检查 baseline、控制器版本和观测语义。历史长度相同、数组长度相同，都不足以证明来源兼容。地形或域随机化参数可以不同，成功加载不代表鲁棒性已经通过。

这次 25 项测试覆盖了模型保存/加载和错误元数据拒绝。一次三拍的实际仿真里，保存的指令复跑得到了逐值相同的 qpos/qvel。可视化隔离测试把假窗口的重力与 qpos 改坏，控制仿真轨迹仍与无窗口运行一致。这里没有新训练结果，也没有真机测试。

低层的转向结果和延迟失稳记录见[轮腿动作探针](../d1_v3_action_probes/README.md)。20 ms 测量延迟下的转弯仍会摔倒，增加这个入口没有消除该问题。键盘命令类的初稿由 Claude Opus 辅助编写，项目侧修正净空初值并检查超时、限幅和停车行为；调用日志保留在本地。
