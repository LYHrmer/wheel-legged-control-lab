# 从降阶 LQR/MPC 到 D1 整机残差强化学习

这份说明把实验分成两个层次。先用六状态模型拆开线性化、LQR、MPC 和奖励，再到 16 执行器
D1 上处理接触、力矩分配、延迟与残差学习。每个阶段都留有可运行的入口；先记录基线，再改
一个参数并保存失败现象，PPO 放在经典控制验证之后。

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
| B6 | D1 16-DOF | 如何加入键盘、转向、跳跃和物理地形 | `d1/interactive.py`、`d1/terrain.py` |
| B7 | D1 16-DOF | 状态延迟进入经典控制后会发生什么 | `d1/state_estimation.py` |

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

[`WheelLeggedPlant.linearize()`](../src/wheel_legged_control/model.py) 直接在仿真平衡点附近对状态
和输入做中心差分：

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

D1 层把整机动力学中的以下量带进了实验：

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
相反方向的力。每条腿的力矩为：

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

`R` 的数值很小，因为输入单位是牛顿，`B` 的量级约为 `10^-5`。判断权重时要连同状态、
输入单位和离散模型的量级一起看。

MPC 使用相同模型、权重和终端 Riccati 代价，预测 20 个 `10 ms` 步长，并在优化内显式
施加 `±180 N` 约束。两种控制器的差别集中在有限时域预测和约束求解。

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

理想状态的正奖励上限为 `4.2`：

\[
\begin{aligned}
r_v &= 2.0\exp[-((v_x-v_{cmd})/0.35)^2],\\
r_u &= 1.0\exp[-(\phi/0.22)^2-(\theta/0.22)^2],\\
r_h &= 1.0\exp[-((z-z_{cmd})/0.030)^2],\\
r_{alive} &= 0.2,\\
r_a &= -0.04(a_{x,t}^2+2a_{z,t}^2),\\
r_{\Delta a} &= -0.025(\Delta a_{x,t}^2+2\Delta a_{z,t}^2).
\end{aligned}
\]

另外惩罚软关节限位、非轮部位触地；摔倒额外为 `-10`。实现位于
[`d1/rewards.py`](../src/wheel_legged_control/d1/rewards.py)。

奖励使用速度跟踪误差，负速度命令也能正确计分。竖直动作的代价更高，是因为它的物理缩放
为 `80 N`，而纵向残差只有 `45 N`；训练日志还显示，等权动作代价会产生持续向上的偏置。

### 10. 随机化与延迟

训练时每回合改变：

| 参数 | 范围 |
|---|---:|
| 基座质量比例 | `0.90–1.12` |
| 阻尼比例 | `0.80–1.25` |
| 地面摩擦比例 | `0.65–1.30` |
| 执行器强度比例 | `0.85–1.05` |
| 残差动作延迟 | `0–3` 个 `10 ms` 周期 |
| 状态延迟（estimated） | `0–3` 个 `10 ms` 周期 |
| 初始 Roll | `±0.035 rad` |
| 初始 Pitch | `±0.045 rad` |
| 速度命令 | `-0.75–0.75 m/s` |
| 目标高度 | `0.445–0.475 m` |
| 随机推力 | `90–170 N`，方向随机 |

`oracle` 模式保留旧实验：经典控制读取统一的真值快照，`sensor_noise` 只扰动 PPO 观察。
`estimated` 模式会把带噪延迟快照同时交给 LQR/MPC、VMC、PPO、安全逻辑和地形估计。动作
延迟与状态延迟分别采样，不再共用一个队列。

这里仍没有模拟 IMU 或编码器融合。当前 source 只是确定性的误差通道，适合做灵敏度实验；
详细字段和时间语义见 [`state_estimation.md`](state_estimation.md)。

延迟补偿可选 `constant_velocity`。它用快照中的机身速度、角速度和关节速度做最多 `50 ms`
的一阶外推，接触仍保持延迟测量。评估时要保留原始状态年龄，并和 `none` 使用完全相同的
评测种子；只看外推成功的常速片段会高估效果。

### 11. 训练与评估

这台电脑上使用 8 个 CPU 环境训练：

```bash
wheel-legged-train \
  --robot d1 \
  --baseline lqr \
  --state-mode oracle \
  --latency-compensation none \
  --steps 400000 \
  --envs 8 \
  --seed 7 \
  --runs 1 \
  --device cpu \
  --output results/d1_residual_ppo
```

PPO rollout 批次使实际步数向上取整为 `401408`。本机 8 个 CPU 环境约用 3.5 分钟；全身
MuJoCo 是主要耗时，小型 MLP 放到 GPU 通常不会更快。

固定场景与 30-seed 配对审计：

```bash
MUJOCO_GL=egl wheel-legged-d1-benchmark \
  --state-mode oracle \
  --policy results/d1_residual_ppo/model.zip \
  --audit-episodes 30 \
  --gif
```

当前随机域结果来自 `oracle` 模式：

| 控制器 | 失败 | 速度 RMSE | Pitch RMSE | 高度 RMSE |
|---|---:|---:|---:|---:|
| D1 LQR+VMC | `0/30` | `0.326 ± 0.124 m/s` | `1.490 ± 0.824°` | `9.745 ± 3.491 mm` |
| D1 MPC+VMC | `1/30` | `0.298 ± 0.152 m/s` | `2.162 ± 3.216°` | `11.429 ± 9.387 mm` |
| D1 LQR+VMC+PPO | `0/30` | `0.327 ± 0.121 m/s` | `1.511 ± 0.789°` | `9.725 ± 3.262 mm` |

PPO − LQR 的配对均值差及 95% 置信区间：

- 平均奖励：`-0.047 [-0.071, -0.022]`；
- 速度 RMSE：`+0.001 [-0.009, +0.011] m/s`；
- Pitch RMSE：`+0.021 [-0.054, +0.096]°`；
- 高度 RMSE：`-0.020 [-0.622, +0.582] mm`。

三个误差区间都跨过 0，奖励区间全部低于 0。这个 PPO checkpoint 没有优于 LQR，只用于
练习残差接口、策略门控和配对评测。MPC 在随机域出现一次失败，CSV 保留了这条记录。

### 12. 键盘、转向和地形课程

```bash
wheel-legged-d1-play
```

交互入口使用同一个 D1 plant 和控制器。`W/S` 调前进速度，`A/D` 通过左右轮差速调偏航，
空格触发带接触检查的四段跳跃。按 `1…6` 可直接回到不同地形前。课程中使用轮地接触点拟合
支撑平面，Ramp、台阶、乱石、波浪路和横杆都参与碰撞。

加载 checkpoint 后按 `L` 可开关残差策略：

```bash
wheel-legged-d1-play --policy results/d1_residual_ppo/model.zip
```

跳跃、恢复、急转和明显斜坡会触发策略门控。PPO 的训练集尚未覆盖这些运动，门控避免把平地
结果扩写成全地形能力。场景参数、跳跃时序、验收表和调试记录见
[`interactive_course.md`](interactive_course.md)。

## 13. 下一轮实验

固定延迟曲线已经完成。`10 ms` 下，一阶外推保持全部回合成功并降低了部分跟踪误差；到
`20 ms`，LQR/MPC 的成功数都增加，但配对区间跨过 0，且多数控制步因关节运动学边界而退回
原始状态。`30/50 ms` 两种方法都没有成功回合。原始 CSV 在
[`results/d1_state_delay_raw`](../results/d1_state_delay_raw/delay_sweep_episodes.csv) 与
[`results/d1_state_delay_compensated`](../results/d1_state_delay_compensated/delay_sweep_episodes.csv)。

下一轮只替换预测器：用已施加力矩和局部线性模型从测量时刻滚动到控制时刻，仍复用这 30 个
种子和五个延迟点。通过条件预先定为：`20 ms` 成功率配对差的 95% 区间下界高于 0，同时平均
机械功率不增加超过 10%。未达到这两个条件，就保留为消融结果，不换掉当前默认配置。真正的
IMU/编码器估计器、参数辨识和实机安全层要等到有传感器日志后再进入主线。

每次实验保留 `training_config.json`、原始 CSV 和对应的失败视频。当前文档与结果目录直接
覆盖，历史交给 Git 管理，避免出现 `final_v2_really_final` 一类副本。
