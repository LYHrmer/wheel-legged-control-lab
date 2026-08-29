# D1 模型卡与本地 URDF 审计

这份文档回答两个问题：仓库里的 D1 模型从哪里来，以及为什么没有直接上传另一份
`d1h_wcd_description`。

## 最终采用的公开模型

仓库使用
[`Rangens/WMP-D1-loco`](https://github.com/Rangens/WMP-D1-loco/tree/540e98d0a0c2212bc74908b98088b870a79e2f53/resources/robots/d1)
提交 `540e98d0a0c2212bc74908b98088b870a79e2f53` 中的 D1 URDF 与 STL。选择它的主要原因
不是“名字更像官方”，而是来源固定、仓库声明 Apache-2.0，适合在公开学习项目中复用。

导入后的可复现参数为：

| 项目 | 数值 |
|---|---:|
| 总质量 | `48.14686526 kg` |
| 广义位置 `nq` | `23` |
| 广义速度 `nv` | `22` |
| 执行器 `nu` | `16` |
| 浮动基座 | `7 nq / 6 nv` |
| 腿关节力矩上限 | `±80 Nm` |
| 轮关节力矩上限 | `±12 Nm` |
| 腿/轮速度上限 | `20 / 30 rad/s` |

MuJoCo 的 URDF 导入器默认得到固定基座、零执行器模型。因此
[`build_d1_model()`](../src/wheel_legged_control/d1/model.py) 统一补上浮动基座、16 个直接
力矩执行器、关节阻尼/电枢、轮地摩擦、地面和光源。URDF 只增加了
`discardvisual="false"`，STL 未改动。完整许可证与改动记录见
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。

本末科技公开的
[`DDTRobot/d1_mjlab`](https://github.com/DDTRobot/d1_mjlab) 被用来交叉核对自由度、质量、
关节顺序和控制量级，但该仓库当前没有明确 LICENSE，因此本项目没有复制其中的代码或
XML。模型来源和算法来源被刻意分开。

## `d1h_wcd_description` 的审计结果

用户提供的本地目录能通过 `check_urdf`，其中网格引用也完整；`mujoco/robot.xml` 和
`scene.xml` 能由 MuJoCo 3.12 正常编译。它不是坏文件，但暂时不适合直接公开：

| 检查项 | 本地模型结果 |
|---|---:|
| URDF links / joints / movable joints | `21 / 20 / 16` |
| URDF transmission | `0` |
| MJCF `nq / nv / nu` | `23 / 22 / 14` |
| MJCF 总质量 | `53.00052 kg` |
| MJCF sensors | `53` |
| `package.xml` license | `TODO` |

关键问题有四个：

1. 它与 WMP 公开模型、官方当前 `d1_mjlab` 模型均不相同，不能从文件内容证明可再发布。
2. URDF 的前侧 `FL/FR_foot_joint` 是 fixed，并另外挂了两个被动小轮；MJCF 却把
   `FL/FR_foot_joint` 建成可转大轮，只是注释掉执行器。两套动力学不是同一台机器人。
3. MJCF 只有 14 个执行器，而传感器仍覆盖 16 个关节；若按“D1 16-DOF 全驱”训练，动作
   语义会错位。
4. Xacro 依赖 ROS 包索引，目录没有安装进工作空间时不能独立展开；场景也没有站立
   keyframe 或控制器，零控制加载后机器人只会自由落体。

因此它可以在确认授权后用于“WCD 特殊构型”的本地研究，但不能把它叫作标准 D1 模型，
也不应在许可证未知时推到公开 GitHub。本项目保留了审计结论，没有保留或上传这些文件。

## 验证边界

已验证：

- URDF XML 与运动学树可解析；所有公开网格存在；
- 16 个关节和执行器名称一一对应；
- 500 Hz 物理、100 Hz 控制连续运行；
- VMC 静止 5 秒保持四轮接触，未发生非轮部位触地；
- 固定场景与 30 个随机域种子完成 LQR/MPC/PPO 对照；
- 单元测试覆盖质量/自由度、接触、闭环稳定性、MPC 约束与 Gymnasium API。

未验证：

- 电机电流环、减速器效率、母线电压与热衰减；
- 轮胎形变、地面材料辨识、结构柔性与装配间隙；
- 编码器/IMU 时间同步、真实状态估计器与 ROS2 通信；
- 与某一台实机的参数辨识或 sim-to-real。

所以 README 中使用“full-body D1 simulation”，不使用“数字孪生”或“已完成实机部署”。
