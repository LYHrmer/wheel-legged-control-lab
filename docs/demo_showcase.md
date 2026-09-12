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
PPO，窗口跑的是零残差控制。开发道路包含缓坡、毫米级起伏及小台阶；出生区平坦，
沿正 x 方向前进约 1.4 m 才开始离开平坦区。窗口显示地形参数，不将小起伏画成大障碍。

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/src:$PWD/.local-deps" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/run_d1_locomotion.py --keyboard --source sensor \
  --baseline wheel_leg --action-mode independent8 --seed 1017 --seconds 600 \
  --terrain development --terrain-index 0 \
  --wheel-kp 0.55 --wheel-ki 1.5 --yaw-feedback-gain 4 \
  --leg-feedback-scale 1 --attitude-feedback-scale 0.25 \
  --output results/my_keyboard_acceptance_01
```

| 按键 | 命令变化 |
|---|---|
| W / S | 按住持续前进 / 倒退，目标渐变至 +0.30 / −0.20 m/s；松开后该轴命令归零。 |
| A / D | 按住持续左转 / 右转，最大 ±0.25 rad/s；可与 W/S 同时使用。 |
| R / F | 按住连续调整净空，范围 0.43 到 0.48 m；松开后保持高度目标。 |
| Space | 前进速度与偏航命令归零，净空目标保持不变。 |
| Esc | 请求结束回合。记录为 `keyboard_escape`，当前 CLI 返回退出码 1。 |
| 鼠标左键 / 滚轮 / C | 拖动旋转视角，滚轮缩放，C 恢复跟随相机的默认角度与距离。 |

点击窗口取得焦点后再操作。长按直接读取按键状态，不依赖操作系统的连发频率。
松开方向键、窗口失去焦点或按 Space 都会清掉对应运动请求；相反方向的键相互抵消。
命令归零后还要靠控制器和物理系统减速。当前 82 维入口没有跳跃和横移。

## 现有 GUI 证据与待填记录

[窗口录像](../results/d1_budget_demo/gui_retry/window.mp4)来自真实 MuJoCo
窗口，按键由 X11 自动注入，只捕获该窗口客户区。程序已核对实际 CSV 中
的命令变化；它属于旧版脉冲输入的自动 GUI 集成验证。

2026-09-12 用户实际操作了 191.06 s 仿真回合，以关闭窗口结束。用户报告长按 WASD
不能持续运动、W 有时让地面难以看清，以及默认平地缺少地形。该次人工验收未通过。
MuJoCo 原生查看器的 W 同时切换线框显示，修订入口使用独立 GLFW 窗口处理驾驶按键，
并将键盘默认场景改为开发道路。新版还需人工复验，不能沿用旧自动化结果写成通过。

下面的表留给实际操作者。填入本次窗口观察和 `telemetry.csv` 对应记录后，
再写“通过”或具体问题。

日期：________　操作人：________　输出目录：________

| 项目 | 操作 | 实际观察 / CSV 证据 | 结果 |
|---|---|---|---|
| 持续行驶 | 按住 W 两秒再松开，按住 S 检查倒退；检查地面显示没有变化。 | 未填写 | 待复验 |
| 转向 | 分别长按 A、D，再同时按 W+A 检查边走边转。 | 未填写 | 待复验 |
| 净空 | 按住 R、F，检查高度连续调整且不会启动行驶。 | 未填写 | 待复验 |
| 停车 | 有运动命令时按 Space，检查速度与偏航归零、净空不变。 | 未填写 | 未验收 |
| 失焦 | 行驶时切到其他窗口，检查运动请求归零；回到窗口松开并重新按键。 | 未填写 | 待复验 |
| 退出 | 按 Esc，核对 `summary.json` 中的 `keyboard_escape`。 | 未填写 | 未验收 |

Esc 主动退出时，`completed=false` 与退出码 1 本身不表示跌倒；应同时查看
`stop_reason`。录像中的 GUI 回合也是按 Esc 结束的。
