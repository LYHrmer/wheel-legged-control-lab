# 从降阶 LQR/MPC 到 D1 整机残差强化学习

这不是一份“运行训练命令就结束”的说明。项目分成两个层次：先在六状态模型上看懂控制
算法，再到 16 执行器 D1 上面对接触、力矩分配、约束、延迟和强化学习。建议边运行、边改
参数、边记录失败，而不是一开始就训练 PPO。

## 0. 学习路线

| 阶段 | 模型 | 先回答的问题 | 对应入口 |
|---|---|---|---|
| A1 | 平面三自由度 | 为什么开环不稳定，线性化是什么 | `examples/controller_walkthrough.py` |
| A2 | 平面三自由度 | LQR 与 MPC 的代价、约束有何区别 | `controllers.py` |
| A3 | 平面三自由度 | 奖励各项如何影响策略 | `examples/reward_walkthrough.py` |
| B1 | D1 16-DOF | URDF 如何变成能落地的整机动力学 | `d1/model.py` |
| B2 | D1 16-DOF | 四个轮地支撑力如何分配到关节 | `d1/controllers.py` |
| B3 | D1 16-DOF | 如何在接触闭环上辨识 LQR/MPC 模型 | `d1/linear_model.py` |
| B4 | D1 16-DOF | RL 应补偿什么、不能掩盖什么 | `d1/env.py`、`d1/rewards.py` |
| B5 | D1 16-DOF | 如何证明收益不是偶然 | `d1/experiments.py` |

模型来源、许可证和另一份本地 URDF 为什么没有上传，单独记录在
[`d1_model_card.md`](d1_model_card.md)。

---

## 第一部分：先把 LQR、MPC 和残差 RL 看懂

### 1. 六状态教学模型

平面模型保留轮足机器人矢状面最关键的三个自由度：

\[
q=[x,\theta,l]^\top,
\qquad
x_s=[x,\theta,l,\dot x,\dot\theta,\dot l]^\top.
\]

- `x`：轮轴水平位置；
- `theta`：机身 Pitch；
- `l`：对称腿伸缩量；
- `F_w`：两轮折算后的总水平力；
- `F_l`：两腿的对称推力。

运行：

```bash
python examples/controller_walkthrough.py
```

开环离散矩阵应存在模长大于 `1` 的特征值；LQR 闭环特征值应全部进入单位圆。这个检查
很重要，否则一个本来就稳定的模型也能画出“控制成功”的曲线。

### 2. 从 MuJoCo 到线性模型

[`WheelLeggedPlant.linearize()`](../src/wheel_legged_control/model.py) 没有另写一套与仿真
脱节的方程，而是在平衡点附近对状态和输入做中心差分：

\[
A_c[:,i]\approx\frac{f(x+\epsilon_i,u)-f(x-\epsilon_i,u)}{2\epsilon_i},
\quad
B_c[:,j]\approx\frac{f(x,u+\epsilon_j)-f(x,u-\epsilon_j)}{2\epsilon_j}.
\]

再用零阶保持离散到 `20 ms` 控制周期。线性模型只用于控制器设计，最终评估始终运行
MuJoCo 非线性模型。

### 3. LQR 与 MPC

LQR 最小化无限时域二次型：

\[
J=\sum_{k=0}^{\infty}(e_k^TQe_k+\Delta u_k^TR\Delta u_k),
\qquad \Delta u_k=-Ke_k.
\]

教学模型的权重为：

```text
Q = diag(0.4, 140, 90, 4, 14, 5)
         x    θ   l  ẋ  θ̇  l̇
R = diag(0.025, 0.004)
          Fw     Fl
```

MPC 使用同一个线性模型，但显式预测 20 步并限制执行器：

\[
X=S_xx_k+S_uU,
\qquad
\min_U \frac12 U^THU+f^TU,
\quad u_{min}\le u_i\le u_{max}.
\]

它每次只执行第一个输入，然后根据新状态重新求解。仓库记录 P95 求解时间，而不只写
“可以实时运行”。

### 4. 为什么先做残差策略

教学层的 PPO 不直接接管控制：

\[
u=\operatorname{clip}\left(u_{LQR}+
\operatorname{diag}(18,35)a_{PPO}\right),
\quad a_{PPO}\in[-1,1]^2.
\]

LQR 负责局部稳定，策略只补偿参数失配、延迟与非线性。奖励逐项放在
[`rewards.py`](../src/wheel_legged_control/rewards.py)，并通过 `info["reward_terms"]` 返回。
运行 `python examples/reward_walkthrough.py` 可以看到每项贡献和奖励地形。

![Planar reward landscape](../results/learning/reward_landscape.png)

---

## 第二部分：D1 整机控制

### 5. 整机模型究竟多了什么

D1 层不是把平面小车换一张 STL 外观。它包含：

- 浮动基座 `7 nq / 6 nv`；
- 四条腿，每条 `hip/thigh/calf/wheel` 四个关节；
- `16` 个独立力矩执行器；
- URDF 惯量、关节范围和简化碰撞体；
- 四个半径 `0.087 m` 的显式滚动接触；
- 腿关节 `±80 Nm`、轮关节 `±12 Nm` 的饱和；
- `500 Hz` MuJoCo 物理与 `100 Hz` 控制。

MuJoCo 直接导入 URDF 时只有固定基座和零执行器。仓库通过 `MjSpec` 在一个入口里补齐
浮动关节、执行器、地面与接触参数，避免控制器各自偷偷修改模型。

### 6. 低层 VMC：先站住，再谈 LQR

仅靠关节 PD，模型会因重力下沉并产生明显 Pitch。低层控制器同时使用：

1. 腿关节位置 PD；
2. 四轮速度反馈；
3. 基于虚拟模型的高度、Roll、Pitch 支撑力分配。

目标机身 wrench 写成：

\[
w_d=\begin{bmatrix}F_z&M_x&M_y\end{bmatrix}^T.
\]

四个轮地向上支撑力为 `f=[f_FL,f_FR,f_RL,f_RR]`。由轮心相对机身的位置
`r_i=(x_i,y_i,z_i)` 得到：

\[
\underbrace{
\begin{bmatrix}
1&1&1&1\\
y_{FL}&y_{FR}&y_{RL}&y_{RR}\\
-x_{FL}&-x_{FR}&-x_{RL}&-x_{RR}
\end{bmatrix}}_{A_f}f=w_d.
\]

代码用最小二乘求 `f=A_f^\dagger w_d`，再约束每个支撑力非负。机器人需要向地面施加
相反方向的力，因此每条腿的力矩为：

\[
\tau_i=J_i^T[0,0,-f_i]^T.
\]

这一步位于 [`D1VMCController`](../src/wheel_legged_control/d1/controllers.py)。名义姿态下
静止 5 秒，四轮持续接触，最终 Pitch 约 `0.22°`。

### 7. 在“接触 + VMC”闭环上辨识模型

D1 外层没有复用小车的 `(A,B)`。流程是：

1. 让 VMC 在平地稳定 5 秒；
2. 保存完整 `qpos/qvel` 接触平衡点；
3. 对 `x、pitch、vx、pitch_rate` 分别做正负扰动；
4. 对总纵向轮地力做正负扰动；
5. 每次运行一个 `10 ms` 闭环控制步，中心差分得到离散模型。

外层状态和输入为：

\[
x_r=[x,\theta,\dot x,\dot\theta]^T,
\qquad u_r=F_x.
\]

识别结果由测试实时重算，当前闭环工作点约为 `pitch=0.00386 rad`。LQR 使用：

```text
Q = diag(0.2, 400, 20, 30)
R = 1e-8
|Fx| <= 180 N
```

这里 `R` 很小并不代表“不惩罚控制”，而是输入单位为牛顿，`B` 的量级约为
`10^-5`。只看权重数字、不看单位，会得出错误结论。

MPC 使用相同模型、权重和终端 Riccati 代价，预测 20 个 `10 ms` 步长，并在优化内显式
施加 `±180 N` 约束。它不是另换一套模型，所以 LQR/MPC 对比更公平。

### 8. D1 残差 PPO 的动作与观察

策略仍然不输出 16 维原始关节力矩，只输出两个归一化残差：

\[
\Delta u=\operatorname{diag}(45\text{ N},80\text{ N})a_{PPO}.
\]

- 第一维：总纵向轮地力修正；
- 第二维：总竖直支撑力修正。

低层 VMC 再把它们分配到 16 个执行器，并执行关节力矩饱和。42 维观察包括：

| 分组 | 维数 | 说明 |
|---|---:|---|
| 机身线速度 | 3 | 机体系，归一化 |
| 机身角速度 | 3 | 机体系，归一化 |
| 投影重力 | 3 | 代替欧拉角奇异表示 |
| 速度命令/高度误差 | 2 | 当前控制目标 |
| 12 个腿关节位置误差 | 12 | 轮角度不作为绝对位置输入 |
| 16 个关节速度 | 16 | 按公开速度上限归一化 |
| LQR 当前纵向力 | 1 | 策略知道基线已做了什么 |
| 上一残差动作 | 2 | 抑制高频动作 |

绝对世界位置没有输入策略，避免它记住某一段轨迹。

### 9. 奖励函数

理想状态的正奖励上限为 `3.9`：

\[
\begin{aligned}
r_v &= 2.0\exp[-((v_x-v_{cmd})/0.35)^2],\\
r_u &= 1.0\exp[-(\phi/0.22)^2-(\theta/0.22)^2],\\
r_h &= 0.7\exp[-((z-z_{cmd})/0.045)^2],\\
r_{alive} &= 0.2,\\
r_a &= -0.04\lVert a_t\rVert_2^2,\\
r_{\Delta a} &= -0.025\lVert a_t-a_{t-1}\rVert_2^2.
\end{aligned}
\]

另外惩罚软关节限位、非轮部位触地；摔倒额外为 `-10`。实现位于
[`d1/rewards.py`](../src/wheel_legged_control/d1/rewards.py)。

奖励里没有“向前走得越远越好”，因为那会诱导策略忽略负速度命令；也没有奖励大力输出，
因为残差应该在基线已经足够时趋近于零。

### 10. 随机化与延迟

训练时每回合改变：

| 参数 | 范围 |
|---|---:|
| 基座质量比例 | `0.90–1.12` |
| 阻尼比例 | `0.80–1.25` |
| 地面摩擦比例 | `0.65–1.30` |
| 执行器强度比例 | `0.85–1.05` |
| 残差动作延迟 | `0–3` 个 `10 ms` 周期 |
| 初始 Roll | `±0.035 rad` |
| 初始 Pitch | `±0.045 rad` |
| 速度命令 | `-0.75–0.75 m/s` |
| 目标高度 | `0.445–0.475 m` |
| 随机推力 | `90–170 N`，方向随机 |

注意：当前噪声施加在策略观察上，LQR/VMC 使用仿真真值，相当于假设已有理想状态估计器。
这比宣称“已经模拟真实 IMU”更诚实，也是后续加入 EKF/延迟状态估计的明确接口。

### 11. 训练与评估

这台电脑上使用 8 个 CPU 环境训练：

```bash
wheel-legged-train \
  --robot d1 \
  --baseline lqr \
  --steps 150000 \
  --envs 8 \
  --device cpu \
  --output results/d1_residual_ppo
```

PPO rollout 批次使实际步数向上取整为 `151552`。全身 MuJoCo 是主要瓶颈，小型 MLP 放到
GPU 通常不会更快。

固定场景与 30-seed 配对审计：

```bash
MUJOCO_GL=egl wheel-legged-d1-benchmark \
  --policy results/d1_residual_ppo/model.zip \
  --audit-episodes 30 \
  --gif
```

当前随机域结果：

| 控制器 | 失败 | 速度 RMSE | Pitch RMSE | 高度 RMSE |
|---|---:|---:|---:|---:|
| D1 LQR+VMC | `0/30` | `0.291 ± 0.111 m/s` | `1.245 ± 0.886°` | `10.822 ± 4.345 mm` |
| D1 MPC+VMC | `1/30` | `0.291 ± 0.151 m/s` | `1.898 ± 2.240°` | `11.574 ± 5.289 mm` |
| D1 LQR+VMC+PPO | `0/30` | `0.285 ± 0.105 m/s` | `1.136 ± 0.642°` | `9.902 ± 3.555 mm` |

PPO − LQR 的配对均值差及 95% 置信区间：

- 速度 RMSE：`-0.005 [-0.009, -0.002] m/s`；
- Pitch RMSE：`-0.109 [-0.206, -0.013]°`；
- 高度 RMSE：`-0.920 [-1.455, -0.385] mm`。

改善不大，但三个区间都未跨过 0。这比只挑一个成功 GIF 更能说明策略学到了稳定的小修正。
MPC 在固定速度跟踪上略好，但随机域出现一次失败；这是值得继续诊断的结果，不应删除。

## 12. 建议按顺序做的实验

1. **模型审计**：运行测试，确认质量、16 执行器和四轮接触；故意注释一个轮执行器，观察
   哪个测试首先失败。
2. **VMC 消融**：关掉重力支撑项，只留关节 PD，记录高度和 Pitch 漂移。
3. **权重扫描**：每次只改一个 `Q/R` 对角元素，记录闭环谱半径、RMSE 与饱和占比。
4. **MPC horizon**：比较 `5/20/40`，同时报告误差与求解 P95。
5. **奖励消融**：去掉动作平滑项或接触惩罚，检查策略是否抖动或用机身蹭地。
6. **残差幅值消融**：比较 `±20/45/80 N`，不要默认动作范围越大越好。
7. **状态估计**：让 LQR/VMC 也接收延迟噪声状态，再实现互补滤波或 EKF。
8. **参数辨识**：若能接触实机日志，用自由衰减、阶跃和轮速实验估计阻尼、摩擦与延迟。
9. **sim-to-sim**：把同一策略通过 ROS2 接到官方 sim2sim，先对齐关节顺序和符号。
10. **实机前安全层**：加入电流、速度、姿态、通信超时与急停状态机；未完成这些步骤前不要
    把仿真策略直接下发到机器人。

每次实验保留配置、随机种子、至少三个指标和失败视频。只更新当前文档和结果目录，不建立
`final_v2_really_final` 一类副本；历史交给 Git 管理。
