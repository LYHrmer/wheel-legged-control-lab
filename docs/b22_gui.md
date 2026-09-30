# B22 策略窗口

这个入口使用 C18 基控、99 维观测和 16 维 RL 残差，固定加载 B22：`7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691`。它与旧 `d1-rl`、LQR/MPC 课程窗口是不同入口。B22 GUI 的同轨数值检查与有界键盘验收均已通过，完整证据和失败记录见 [交付报告](b22_gui_true15mm_20261001.md)。

在本机终端直接复制整行。先检查运行文件和权重；检查不加载模型、不推进仿真：

```bash
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22.py --check
```

默认打开 1.6 m/s 固定直行脚本。机器人自动完成准备、加速、保持和制动，任务结束后保存并退出：

```bash
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22.py
```

其他固定脚本和 zero 基控对照：

```bash
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22.py --profile yaw_1p2
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22.py --profile yaw_1p2 --mirror
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22.py --profile ramp_0p45_complete
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22.py --profile bumps_0p4
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22.py --profile rough_0p35
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22.py --actor zero
```

`flat_0p6` 也可选。每次启动使用一个固定地形和命令序列，不是自由穿越整个课程地图。原六项、镜像转向和 B22 模型的控制能力依据 C22 已封存结果；这次 GUI 验收不会重跑全部旧物理批次。

限定键盘窗口：

```bash
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22.py --mode keyboard --seconds 60
```

| 操作 | 含义 |
|---|---|
| 按住 W | 前进，固定目标 1.2 m/s |
| W＋Q / W＋E | 左／右偏航，目标 ±0.3 rad/s |
| 松开 W、Space、X、Stop 按钮 | 请求减速停止 |
| 失去窗口焦点、输入超过 0.25 s 未更新 | 请求停止；恢复焦点后先松开 W 才能重新前进 |
| R / Reset 按钮 | 本次窗口最多一次仿真复位；总运行预算继续累计 |
| Esc / 关闭窗口 | 结束这次仿真并保存记录 |

Space 当前是停止键。A/D 横移、跳跃和物理自救未开放；R 不代表机器人能自行翻正。固定 1.6 m/s 直行证据不授予任意高速转向资格。OS 自动输入测试也不等于人工试驾或用户 GPU 性能测试。

结果自动写入仓库 `results/b22_<时间>_<唯一后缀>/`。通常无需指定 `--output`；显式指定时目录必须不存在。界面显示框架/模型标识、请求/应用/实际速度与输入状态；运行记录包含模型、控制及原生子步计数。zero 由启动参数和运行回执识别，窗口固定的 B22 标题本身不证明加载了 B。当前依赖已验证的本机 Linux/Python 3.10/定制 MuJoCo 环境，研究源码和模型随仓库保存。

B22 相对 A 的有限课程收益已验证，但“RL 超过 zero”的原六项贡献门仍未通过。上述是 MuJoCo oracle 仿真，不是实机部署结论。

本次真实 GLFW/XTest 键盘验收：1102 控制步，实时系数 0.980、12.09 FPS，26/26 有效回调匹配，停止、失焦、输入超时、一次复位及重复复位拒绝全部通过。归档会在窗口结束后继续写入，请等待终端命令返回；它不推进额外物理。
