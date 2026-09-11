# 主线演示与手动试驾

[60 秒对照片](../results/d1_budget_demo/comparison.mp4)把零残差放在左侧，预定的
八维 PPO 放在右侧。模型取训练 seed 31000 的最终 262144 样本存档。
两侧使用同一开发道路，初始状态和测量噪声配置一致，全部 6000 拍命令也已
逐拍核对。原始记录与审计在[演示记录](../results/d1_budget_demo/README.md)。

这个案例里，PPO 的净空 RMSE 从 11.759 降到 6.651 mm；速度 RMSE 从
0.03292 增到 0.04239 m/s，偏航角速度 RMSE 从 0.03397 增到 0.05118 rad/s。
两侧都完成 60 s。这只是一张 development 道路上的一个案例，不能代替正式
多种子留出评测，也不能据此宣称 PPO 全面改善。

## 打开键盘窗口

在仓库根目录执行，输出目录必须不存在；已经用过 `_01` 时换一个名字。
命令沿用演示中的五项低层增益，`sensor` 对应 IMU/编码器融合。这里不加载
PPO，窗口跑的是零残差控制。

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/src:$PWD/.local-deps" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/run_d1_locomotion.py --keyboard --source sensor \
  --baseline wheel_leg --action-mode independent8 --seed 1017 --seconds 60 \
  --wheel-kp 0.55 --wheel-ki 1.5 --yaw-feedback-gain 4 \
  --leg-feedback-scale 1 --attitude-feedback-scale 0.25 \
  --output results/my_keyboard_acceptance_01
```

| 按键 | 命令变化 |
|---|---|
| W / S | 前进速度目标每次加 / 减 0.05 m/s，范围 −0.20 到 0.30 m/s。S 是减量：当前 +0.10 时按一次 S 得到 +0.05，不会直接变成倒退。 |
| A / D | 偏航角速度目标每次加 / 减 0.05 rad/s，范围 ±0.25 rad/s。 |
| R / F | 净空目标每次加 / 减 0.005 m，范围 0.43 到 0.48 m。 |
| Space | 前进速度与偏航命令归零，净空目标保持不变。 |
| Esc | 请求结束回合。记录为 `keyboard_escape`，当前 CLI 返回退出码 1。 |

运动键停止输入超过 0.8 秒墙钟时间后，前进与偏航命令归零。R/F 不延长
这段时间，也不会恢复已失效的运动命令。命令归零后还要靠控制器和物理系统
减速，这个超时机制不保证紧急制动。当前 82 维入口没有跳跃和横移。

## 现有 GUI 证据与待填记录

[窗口录像](../results/d1_budget_demo/gui_retry/window.mp4)来自真实 MuJoCo
窗口，按键由 X11 自动注入，只捕获该窗口客户区。程序已核对实际 CSV 中
的命令变化；它属于自动 GUI 集成验证。人工试驾尚未验收。

下面的表留给实际操作者。填入本次窗口观察和 `telemetry.csv` 对应记录后，
再写“通过”或具体问题。

日期：________　操作人：________　输出目录：________

| 项目 | 操作 | 实际观察 / CSV 证据 | 结果 |
|---|---|---|---|
| 速度增减 | 连续按 W，再按 S，检查目标逐次变化。 | 未填写 | 未验收 |
| 转向 | 分别按 A、D，检查偏航命令的变化。 | 未填写 | 未验收 |
| 净空 | 按 R、F，检查净空目标；留意是否误续运动命令。 | 未填写 | 未验收 |
| 停车 | 有运动命令时按 Space，检查速度与偏航归零、净空不变。 | 未填写 | 未验收 |
| 超时 | 停止运动键输入超过 0.8 秒，再按 R/F，检查运动命令保持为零。 | 未填写 | 未验收 |
| 退出 | 按 Esc，核对 `summary.json` 中的 `keyboard_escape`。 | 未填写 | 未验收 |

Esc 主动退出时，`completed=false` 与退出码 1 本身不表示跌倒；应同时查看
`stop_reason`。录像中的 GUI 回合也是按 Esc 结束的。
