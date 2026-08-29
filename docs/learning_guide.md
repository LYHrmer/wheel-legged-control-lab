# 从 LQR、MPC 到残差强化学习

这份指南面向第一次把经典控制和强化学习放进同一个机器人项目的同学。建议不要直接
运行训练命令后只看 GIF，而是按下面顺序阅读和实验。

## 0. 先认识问题

模型保留轮足机器人矢状面最关键的三个自由度：

\[
q=[x,\theta,l]^\top,
\quad
x_s=[x,\theta,l,\dot x,\dot\theta,\dot l]^\top.
\]

- `x`：轮轴水平位置；
- `theta`：机身 Pitch，`0` 表示直立；
- `l`：两条腿同步伸缩量；
- `F_w`：两侧轮电机折算到地面的总水平力；
- `F_l`：两腿的对称伸缩推力。

首先打开 [`wheel_legged_planar.xml`](../src/wheel_legged_control/assets/wheel_legged_planar.xml)，
找到三个 joint 和两个 actuator。然后运行：

```bash
python examples/controller_walkthrough.py
```

输出中应当看到开环存在模长大于 `1` 的离散特征值，说明它确实是需要主动平衡的倒立
系统；LQR 闭环特征值则全部位于单位圆内。

## 1. MuJoCo 动力学如何变成线性模型

控制器并没有手写一套与仿真脱节的倒立摆方程。[`model.py`](../src/wheel_legged_control/model.py)
在直立平衡点附近对 MuJoCo 状态和控制分别施加很小的正、负扰动，通过中心差分得到

\[
A_c[:,i]\approx\frac{f(x+\epsilon_i,u)-f(x-\epsilon_i,u)}{2\epsilon_i},
\]

\[
B_c[:,j]\approx\frac{f(x,u+\epsilon_j)-f(x,u-\epsilon_j)}{2\epsilon_j}.
\]

随后用零阶保持把连续模型离散为控制周期 `T_s=20 ms` 下的 `(A,B)`。对应代码是
`WheelLeggedPlant.linearize()`。

学习要点：线性模型只在平衡点附近准确；MuJoCo 闭环评估仍运行原始非线性动力学。

## 2. LQR 在做什么

LQR 最小化无限时域二次型代价：

\[
J=\sum_{k=0}^{\infty}(e_k^TQe_k+\Delta u_k^TR\Delta u_k),
\qquad \Delta u_k=-Ke_k.
\]

本项目的状态权重按顺序为：

```text
Q = diag(0.4, 140, 90, 4, 14, 5)
         x   θ    l  ẋ  θ̇  l̇
R = diag(0.025, 0.004)
          Fw     Fl
```

Pitch 权重最大，因为首先要避免摔倒；腿高次之。输入权重不能只比较数字大小，因为轮力和
腿力的物理量级不同。`solve_discrete_are` 求出 Riccati 方程，再计算反馈增益 `K`。

速度命令不能简单写成一个不断远离的绝对位置目标，因此
[`LQRController.reference()`](../src/wheel_legged_control/controllers.py) 使用有界移动位置
参考，既帮助消除速度稳态误差，又防止受到推力后位置参考无限积累。

## 3. MPC 比 LQR 多了什么

线性 MPC 使用相同的 `(A,B,Q,R)`，但每个控制周期显式预测未来 20 步：

\[
X=S_xx_k+S_uU.
\]

代入代价函数后得到带 box constraints 的凝聚 QP：

\[
\min_U \frac{1}{2}U^THU+f^TU,
\quad u_{min}\le u_i\le u_{max}.
\]

只执行最优序列的第一个输入，下一周期根据新状态重新求解。实现位于
[`LinearMPCController`](../src/wheel_legged_control/controllers.py)。项目记录求解 P95 耗时，
而不只是宣称“实时”。

可以尝试：把 `horizon` 从 `20` 改为 `5/40`，比较跟踪误差与求解时间。

## 4. 为什么使用残差 RL

纯 RL 直接输出全部执行器力时，策略必须同时学会“不摔倒”和“补偿模型失配”。本项目让
LQR 负责局部稳定，PPO 只学习一个有界修正：

\[
u=\operatorname{clip}(u_{LQR}+
\operatorname{diag}(18,35)\,a_{PPO}).
\]

因此动作 `[-1,1]^2` 分别对应 `±18 N` 轮力残差和 `±35 N` 腿力残差。即使策略输出
极值，最终控制仍经过执行器限幅和延迟模型。环境实现见
[`env.py`](../src/wheel_legged_control/env.py)。

策略观察量没有绝对世界位置，而包含速度、Pitch、Pitch 角速度、腿高误差、腿速、命令、
基线控制量和上一时刻残差。这样可以避免策略死记某个地图坐标。

## 5. 奖励函数逐项拆解

完整实现单独放在 [`rewards.py`](../src/wheel_legged_control/rewards.py)，环境返回的
`info["reward_terms"]` 会公开每个分量。正常状态下：

\[
\begin{aligned}
r_{bal}&=0.45\exp(-18\theta^2-0.35\dot\theta^2),\\
r_{vel}&=0.35\exp(-1.8(\dot x-\dot x_{cmd})^2),\\
r_h&=0.20\exp(-90(l-l_{cmd})^2-0.08\dot l^2),\\
r_a&=-0.04\lVert a_{PPO}\rVert_2^2,\\
r&=r_{bal}+r_{vel}+r_h+r_a.
\end{aligned}
\]

越过 `|theta| >= 0.70 rad` 时终止回合并额外奖励 `-10`。

权重之和在理想跟踪、零残差时恰好为 `1.0`，便于读日志。使用指数形式有两个目的：

1. 每个正向分量天然有界，不会因为某个量纲的误差把其他目标完全淹没；
2. 越接近目标，奖励梯度越能区分细小改进。

动作惩罚阻止 PPO 在名义场景无意义地覆盖 LQR。它不能太大，否则策略不愿补偿额外载荷；
也不能太小，否则策略会用高频大动作换取一点跟踪收益。

运行下面的脚本可打印一个样例状态的每项贡献，并生成奖励地形：

```bash
python examples/reward_walkthrough.py
```

![Reward landscape](../results/learning/reward_landscape.png)

## 6. PPO 训练循环

训练入口是 [`train.py`](../src/wheel_legged_control/train.py)，核心流程可以写成：

```text
repeat:
    在 8 个随机化环境中收集 256 步 rollout
    用 value network 计算 GAE advantage
    repeat 10 epochs:
        打乱样本，按 256 batch 切分
        优化 clipped policy objective + value loss
    保存策略
```

训练时随机化躯干质量、阻尼、0–40 ms 延迟、传感器噪声、初始 Pitch、速度/腿高命令和
一次水平推力。具体范围为：

| 参数 | 训练范围 |
|---|---:|
| 躯干质量比例 | `0.80–1.25` |
| 广义阻尼比例 | `0.70–1.40` |
| 控制延迟 | `0–2` 个控制周期 |
| 初始 Pitch | `-0.09–0.09 rad` |
| 速度命令 | `-1.0–1.0 m/s` |
| 腿高命令 | `-0.07–0.07 m` |
| 随机推力 | `25–55 N`，方向随机 |

提交的 checkpoint 使用种子 `7`、八环境、300,000 步。

## 7. 如何判断训练是否真的有效

不要只看训练 reward。运行：

```bash
wheel-legged-benchmark \
  --policy results/residual_ppo/model.zip \
  --audit-episodes 20 \
  --gif
```

固定场景检查可解释的时序曲线，随机域审计则用相同的 20 个种子公平比较 LQR、MPC 和
LQR+PPO。当前结果显示 PPO 改善速度和腿高，但略微牺牲 Pitch 与控制努力。这种多指标
结论比“RL 成功了”更有价值。

## 8. 建议练习

1. **奖励消融**：分别把速度、高度或动作项权重设为零，记录策略会钻什么空子。
2. **LQI 基线**：给腿高误差加入积分状态，检验恒定载荷问题是否仍需要 RL。
3. **约束实验**：缩小轮力上限，观察 LQR clip 与 MPC 预测约束的区别。
4. **纯 PPO 对照**：让策略输出完整力，但保持相同训练步数和随机化范围。
5. **历史观测**：堆叠最近 3–5 帧，研究延迟场景是否改善。
6. **三维升级**：加入左右轮、左右腿、Roll/Yaw，再引入 VMC 或 WBC。

每个练习至少报告三个指标和失败案例。尤其是第二项：如果简单积分器已经解决问题，就不应
把同一部分收益包装成“RL 独有能力”。
