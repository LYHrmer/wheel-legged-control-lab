# 20 ms 测量延迟：从短探针到完整道路

这组零残差诊断没有通过 20 ms 延迟下的 60 s 开发任务。降低反馈增益后，失败从出生点附近推迟到
约 20–22 s，机器人已经驶入起伏区域；这仍是失败。这里保留诊断过程，方便逐项重做。

所有延迟指整包 IMU/编码器测量延迟：控制周期 10 ms，两个决策步为 20 ms。
执行器仍为理想直通，除非各目录的协议另有明确设置。数据由 MuJoCo 3.12.0 生成，
噪声参数是人为设置的，未经真实 D1 标定。

## 先固定要回答的问题

最初的矛盾是：相同轮速 PI，在无限平面上能完成延迟转向探针，换成完整任务却在开始
前进前失稳。先不训练策略，依次检查配置、地面表示和反馈。

| 配置 | 20 ms 平面短探针 | 60 s 开发任务，两路 × 两回合 seed |
|---|---|---|
| 原轮速 PI 2.2/3.0，偏航反馈 4 | 未通过原延迟探针 | 4/4 失败，0.91–4.41 s |
| 轮速 PI 0.55/1.5，偏航反馈 4 | 两个 8 s 转向达到指令转角的 70.83% / 70.71% | 4/4 失败，1.70–3.02 s |
| 上一行，再把腿部 PD 乘 0.25 | 通过下面的 4 s 站立检查 | 4/4 失败，19.76–22.40 s |
| 上一行的轮速 PI，仅把姿态 PD 乘 0.25 | 通过下面的 4 s 站立检查 | 4/4 失败，20.23–21.32 s |

最后两行各只缩放一组 PD，不是同时修改。轮速 PI 0.55/1.5、腿部与姿态 PD 都不变时，
无延迟的四个开发案例全部完成并达标，见
[无延迟对照](low_bandwidth_delay0_development/evaluation.json)。
短探针的 70% 是事先确定的转角门槛，不是完整道路的质量门槛。

## 地面高度相同，碰撞表示仍然不同

[2×2 对照](geometry_domain_parity/summary.json)统一了出生位置、测量 seed、
零动作和恒定站立命令。两个因素是：

- 无限平面，或所有高度均为零的 heightfield；
- 是否在 reset 前调用一次中性的 `set_domain`。

两种平面案例都完成 4 s；两种 heightfield 案例的轨迹也互相一致，但只执行到 1.02 s。
后者在 0.73 s 首次出现四轮均无接触，随后准备的姿态目标超过允许范围。
记录有 103 行，其中最后一行是被拒绝的候选命令，实际只执行了 102 拍。
不能把记录行数当成积分次数。

真实地面高度、坡度和非平坦曝光一直为零。中性 `set_domain` 没有改变任一种地面的
轨迹，因此这次对照不支持“随机化调用本身导致失败”。

D1 轮子的碰撞几何为 cylinder。MuJoCo 对平面与凸几何、heightfield 使用不同的碰撞
处理路径，见[官方计算说明](https://mujoco.readthedocs.io/en/latest/computation/)。
这是继续检查接触与反馈耦合的理由，尚不足以判定仿真器存在 bug。

## 一次只改一个因素

下面各行从同一 heightfield 站立案例开始。仅这个表里的改动生效，不累积上一行的改动。

| 单一变化 | 实际结束时刻 | 结果 |
|---|---:|---|
| 不修改 | 1.02 s | 下一拍姿态命令被拒绝 |
| 约束求解迭代 20 → 100 | 1.02 s | 轨迹没有改变 |
| CCD 容差 1e-6 → 1e-8 | 1.59 s | 摔倒或机身接触 |
| 支撑面姿态固定为名义水平，仍用测量高度 | 1.73 s | 摔倒或机身接触 |
| 腿部 PD 80/3 → 20/0.75 | 4.00 s | 站立检查完成 |
| 姿态 PD 180/24 → 45/6 | 4.00 s | 站立检查完成 |

原始指标见[数值设置对照](heightfield_single_variable/summary.json)与
[反馈缩放对照](feedback_bandwidth/summary.json)。
名义水平姿态是固定先验，没有读取真实地形修正反馈。

延迟会使反馈使用过去的状态。降低 PD 系数可能减轻这类振荡，但也会降低跟踪刚度。
这里同时缩放 Kp、Kd，并不保持二阶近似的阻尼比不变，不能称为等阻尼设计。
4 s 站立通过后，还要重新检查运动、转向和高度命令。

## 完整道路把什么问题暴露出来

腿部 PD 缩放的四个案例已有 44–50% 的非平坦几何曝光；高度 RMSE 却达到约
0.038–0.042 m。姿态 PD 缩放的高度 RMSE 较小，约 0.0105–0.0156 m，
偏航 RMSE 则达到约 0.21–0.49 rad/s。两组都在转向阶段附近结束。

这些数值覆盖的是不同长度的失败轨迹，不能拿 RMSE 较小的一组直接宣称更优。
逐案例的时长和终止原因分别在
[腿部缩放](leg_pd_quarter_delay20_development/evaluation.json)、
[姿态缩放](attitude_quarter_delay20_development/evaluation.json)中。
没有修改正式六模型实验的控制器或旧 checkpoint。

## 自己复做

先安装项目依赖，在仓库根目录运行，输出目录每次换新：

```bash
python scripts/probe_d1_delay_parity.py --output results/my_delay_parity

python scripts/diagnose_d1_heightfield_delay.py \
  --variants reference solver_iterations_100 ccd_tolerance_1e-8 fixed_attitude \
  --output results/my_heightfield_diagnosis

python scripts/diagnose_d1_heightfield_delay.py \
  --variants reference leg_pd_quarter attitude_feedback_quarter \
  --output results/my_feedback_diagnosis

python scripts/run_d1_locomotion_experiment.py evaluate \
  --source imu_encoder_fusion --measurement-delay 2 --split development \
  --wheel-kp 0.55 --wheel-ki 1.5 --attitude-feedback-scale 0.25 \
  --output results/my_delayed_development
```

最后一条使用正式五参数接口，与当时单独标记的诊断脚本 schema 不同；应比较实际控制
参数与轨迹，不能改旧元数据让它伪装成新版本记录。原始 CSV 与源码快照的获取方法见
[复现说明](../../docs/reproducibility.md)。

建议先预测“接触消失、关节超速、姿态目标被拒绝”三者的先后，再从 CSV 找首次发生时刻。
最后一项只是停止条件，最早的问题可能已在前几拍出现。
