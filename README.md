# Wheel-Legged Control Lab

[![tests](https://github.com/LYHrmer/wheel-legged-control-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/LYHrmer/wheel-legged-control-lab/actions/workflows/tests.yml)

D1 轮足机器人的控制与强化学习练习仓库，全部在 MuJoCo 里完成，没有实机部分。整机模型是
`23 nq / 22 nv / 16 执行器`，物理 500 Hz，决策 100 Hz，一个控制拍里做五次 2 ms 积分。
仓库同时保留一个六状态平面教学模型，用来先把线性化、LQR、MPC 和 PPO 的公式对到代码上。

当前主线是命令条件轮足任务：训练、评测和键盘共用同一个控制循环、同一份 82 维观测。
固定预算的六模型开发/留出对照已经跑完，结果在下面的表里。早期的 42/44/45 维实验路线和
任务空间逆动力学 QP 原型都还在仓库中，入口不同、checkpoint 互不兼容，位置见
[文档导航](docs/index.md)。README 只讲当前这一条主线。

原始轨迹与源码快照的下载方式见[复现说明](docs/reproducibility.md)。

项目与本末科技的官方代码无关，只报告仿真结果，不主张 sim-to-real 或数字孪生。

## 主线控制流

命令有三个量：前进速度、偏航角速度、机身离地净空。每拍先发布状态、预览命令与基线，编码
82 维观测，策略给出动作，控制器输出 16 轴力矩，再做五次物理积分。
[控制循环](src/wheel_legged_control/d1/control_loop.py)要求先 `prepare(command)` 再
`step(action)`，逐拍时序和坐标系细节在[命令条件轮足实验](docs/locomotion_lab.md)。两种控制
结构共用这套循环：

| 结构 | 动作 | 归一化 ±1 对应的尺度 |
|---|---|---|
| `wheel_leg` | 8 维：四腿轮心伸缩修正、四轮速度目标修正 | ±4 cm、±4 rad/s |
| `lqr` | 2 维：纵向、竖直机身力残差 | ±11.25 N、±20 N |

八维方案同时换掉了低层控制结构，它和两维方案的差别不能全部算到动作维数上。每种结构各留一份
自己的零残差对照。运行入口的状态来源两选一：`oracle` 读真值和碰撞几何高度；`sensor` 用 IMU/编码器融合
加理想接触开关拟合支撑面，需要已知的初始放置姿态，没有定位、在线 bias 估计或滑移模型。

## 装好并跑第一次

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[rl,dev]"
```

上面先安装 CPU 版 PyTorch。其他平台的安装方式见
[PyTorch 官方说明](https://pytorch.org/get-started/locally/)。
第一次跑零残差，12 s 平地，不开窗口：

```bash
python scripts/run_d1_locomotion.py --source oracle --seconds 12 \
  --output results/first_flat_zero
```

不给 `--policy` 就是零残差，不给 `--keyboard` 就不打开 viewer，无桌面环境也能跑完。输出
目录必须是新的，脚本不覆盖旧实验。跑完先看 `summary.json` 的终止原因，再看 `telemetry.csv`
里已执行的命令和误差、`states.npz` 的真实状态与逐子步力矩，`manifest.json` 记录每个产物的
SHA-256。

换 60 s 的开发道路：

```bash
python scripts/run_d1_locomotion.py --source oracle --seconds 60 \
  --terrain development --terrain-index 0 --output results/first_dev_road
```

地图固定 12×6 m，出生点在 `(-3.8, 0)` 的平地。回合跑满 60 s 不等于走过起伏路段，先在轨迹
里确认实际遇到了什么地形。

加载 checkpoint 要同时给 `--policy` 和它的 `--metadata` sidecar，脚本会校验 baseline、
各项 schema、history 长度、观测/动作空间、控制器增益和模型 SHA-256，任一项不一致直接拒绝；
训练与回放命令在[命令条件轮足实验](docs/locomotion_lab.md)。只加载自己生成或信任的 SB3
文件，它的序列化格式能执行 Python。

## 键盘接口

```bash
python scripts/run_d1_locomotion.py --keyboard --source sensor \
  --output results/my_keyboard_run
```

| 按键 | 作用 | 步长与范围 |
|---|---|---|
| `W` / `S` | 前进速度目标增/减 | 0.05 m/s，`-0.20…0.30 m/s` |
| `A` / `D` | 左/右偏航角速度 | 0.05 rad/s，`±0.25 rad/s` |
| `R` / `F` | 升/降机身净空 | 0.005 m，`0.43…0.48 m`，初值 0.455 m |
| `Space` | 前进与偏航命令归零 | 净空保持不变 |
| `Esc` | 结束回合 | 终止原因记为 `keyboard_escape` |

运动键停下超过 0.8 秒（真实时间）后，前进和偏航命令自动归零。净空键不刷新这个计时，也不会
唤醒已经过期的运动命令。超时只清掉命令，刹车仍然由控制器和物理决定，这些是软件命令限幅，
不是急停。新入口没有跳跃，也没有横移。D1 四轮没有转向机构，`A/D` 靠左右轮差速改航向。旧入口
`wheel-legged-d1-play` 的 `Space` 是跳跃，属于历史 42 维路径，两套按键不要混用。

## 留出道路结果

六个模型 = 两种结构 × 训练种子 24000/25000/26000，每个固定 32768 个样本，oracle 状态源、
单帧 82 维观测，训练用四张固定道路，开发和留出各两张，checkpoint 一律取预算末端，没有按
留出成绩挑过。下表对两张留出道路等权平均，PPO 再对三个训练种子等权平均。

| 控制方式 | 速度 RMSE m/s | 高度 RMSE mm | 机械活动量 W | 质量达标 |
|---|---:|---:|---:|---:|
| LQR 零残差 | 0.07969 | 15.30 | 47.44 | 1/2 道路 |
| LQR + PPO | 0.08482 | 15.99 | 44.63 | 2/6 种子×道路 |
| 轮腿零残差 | 0.04016 | 9.84 | 16.03 | 2/2 道路 |
| 轮腿 + PPO | 0.03328 | 11.76 | 20.50 | 6/6 种子×道路 |

轮腿 PPO 把留出速度 RMSE 从 0.04016 降到 0.03328 m/s，同时高度 RMSE 增加约 1.9 mm、机械活动
量增加约 4.5 W。LQR 结构在开发道路上达标 5/6，迁到留出只剩 2/6，这项退化保留在结果里。这批
证据支持继续研究八维动作接口，不支持“RL 已经全面超过传统控制”，也不能当成动作维数的消融。

固定 oracle 时，同一路面的两个评测 seed 产生了逐字节相同的 CSV 和 NPZ，它们只是重复标签，
不是独立复现，报告因此不给置信区间。完整指标、逐道路结果和两段 60 s 状态录像见
[实验报告](results/d1_v3_locomotion_report/README.md)，两段录像走的道路不同，不能当逐帧公平
对照。

## 现在还不行的部分

新增[九模型延迟对照](results/d1_v3_delay_ablation/README.md)：20 ms 下单帧、四帧、
四帧加随机化分别完成 2/12、0/12、1/12；30 ms 下均为 0/12。
增加历史与这次随机化训练没有带来稳定改善，全部模型及失败记录保留。

- 整包测量延迟加到 20 ms 后，融合状态下的轮腿零残差在开发道路四个案例全部失稳，分别在
  2.05、1.20、0.91、4.41 s。把轮速增益改成 `0.55 / 1.5 / 4` 后，8 s 平地转向探针的累计
  转角达到指令积分的 70.8% 和 70.7%，但完整开发环境的四个案例仍在 3.1 s 内失稳；默认增益
  保持 `2.2 / 3.0 / 4.0`。记录在 [延迟诊断目录](results/d1_v3_delay_diagnosis/)。
- 曝光指标只检查机身/轮心投影处的地形高度和坡度，是几何曝光，不证明每个轮子保持接触，
  也不证明越过了整个障碍。
- 机械活动量是各子步 `sum(abs(力矩 × 关节速度))` 的均值，单位 W，不是电池功率。
- 没有 ROS2 链路、实机参数辨识和 sim-to-real。融合估计依赖已知初始位姿与理想接触开关。
- 轮腿控制按轮心 Jacobian 映射支撑力，没有滑移反馈。两维力残差的接触分配路径仍有静力近似。

## 文档

[文档导航](docs/index.md)是唯一的入口地图，分成学习顺序、当前 82 维主线、历史
42/44/45 维路线与 QP 原型、证据位置四节。常用页面：

- [命令条件轮足实验](docs/locomotion_lab.md)：主线的观测、奖励、八维动作含义，以及从真实
  PPO 更新记录复算 GAE 和概率比的步骤。
- [学习指南](docs/learning_guide.md)：LQR、MPC、GAE 与残差 RL 的基础练习和验收问题。
- [评测协议](docs/evaluation_protocol.md)：场景、成功定义和统计口径。
- [延迟与短历史](docs/delay_learning_lab.md)：四帧观测、两种延迟、随机化与失败配对。
- [执行器辨识移植](docs/actuator_transfer.md)：合成台架参数进入整机，补偿收益与代价。
- [旧键盘课程与受保护跳跃演示](docs/interactive_course.md)：LEGACY，42 维路径，只用于复现
  旧结果，不属于新键盘接口。

## 模型来源与许可

D1 的 URDF 与 STL 来自 Apache-2.0 授权的
[`Rangens/WMP-D1-loco`](https://github.com/Rangens/WMP-D1-loco)，固定到提交
`540e98d0a0c2212bc74908b98088b870a79e2f53`，原许可证随资产保存在
`src/wheel_legged_control/d1/assets/LICENSE-WMP-D1-loco.txt`。结构和量级参考本末科技
[`D1 开发手册`](https://d1-development-manual-cn.readthedocs.io/zh-cn/latest/)与公开
[`DDTRobot`](https://github.com/DDTRobot) 仓库，没有复制许可证不明确的代码。本地
`d1h_wcd_description` 的检查记录在 [d1_model_card.md](docs/d1_model_card.md)。仓库自身代码
按 MIT 发布。

## 测试

```bash
pytest
ruff check src tests examples scripts
```

pytest 守的是接口和验收条件，例如同步采样的有限差分检查、观测与奖励的手算值、键盘命令逻辑。
键盘部分只有自动化测试，`KeyboardCommands` 的时钟可以注入，`key_callback` 按 keycode 直接
调用，没有人在图形窗口里手动验证过它。原始记录另有一份独立复算：

```bash
python scripts/audit_d1_locomotion.py --matrix-root results \
  --output results/my_phase4_arithmetic.json
```

它只用 NumPy 和标准库，不导入仿真器或训练框架，只写新文件；哈希一致只说明字节一致。
