# D1 接触力分配：从最小二乘启发式到约束优化

这份笔记记录 `d1/contact_allocation.py` 里两套分配器（legacy 最小二乘 / constrained 约束优化）
如何把期望的机身合力、合力矩变成 16 路关节力矩，以及对应的坐标约定、验收判据和已知局限。涉及
`contact_allocation.py`、`controllers.py`、`model.py`、`state_estimation.py`、
`linear_model.py`、`hierarchical.py`、`env.py`、`contact_audit.py`、`experiments.py`
（均在 [`d1/`](../src/wheel_legged_control/d1/) 下）。

## 1. 为什么要在 legacy 分配之外再做一套

`D1LegacyContactAllocator.solve()` 做的事：对 `[Fz, Mx, My]` 的 `3×4` 线性系统做最小二乘求四个轮子
的竖直支撑力，`clip` 到 `[0, 0.65mg]`；纵向合力按四轮平均分给驱动力矩通道
（`τ = F_longitudinal · r_wheel / 4`，不经过任何 Jacobian）；最后把“腿部 PD + 支撑力矩 + 驱动
力矩”的总和裁剪到关节力矩上限。这个上限名义上是 `±80/±12 Nm`（`JOINT_TORQUE_LIMIT`），但真实
值来自 `plant.actuator_torque_limit_nm`，会随域随机化的 `actuator_strength_scale`（训练场景默认
`0.85~1.05`）成比例缩放，不是恒定常数。

三个具体问题促成升级：

1. 不检查真实接触状态。几何量来自 `state.foot_offset_world`/`state.foot_jacobian`，不读
   `state.wheel_contact`；轮子瞬间离地时仍按四轮着地假设算支撑力矩下发。
2. 驱动力矩不检查摩擦锥。纵向力平均分配到四个轮子，不管每个轮子当前法向力多大，而摩擦力上限
   随法向力线性变化。
3. 可行性只是事后诊断。`max_constraint_violation` 记录裁剪前力矩超限的归一化比例，不会反过来
   影响已经算好的 `contact_force_world_n`/`achieved_wrench_world`。

`D1ConstrainedContactAllocator` 把“竖直支撑力最小二乘 + 纵向力平均分配 + 事后裁剪”换成一次
显式带约束的优化：决策变量、目标、约束、可行性判定在同一个问题里求解。两套实现通过
`contact_allocation: "legacy" | "constrained"` 切换。legacy adapter 的同状态输出由回归测试核对。

## 2. 术语：不是 centroidal wrench，也不是 full WBC

优化变量是各轮接触点上的接触力，目标是让合力/合力矩跟踪一个期望六维 wrench
`[Fx,Fy,Fz,Mx,My,Mz]`，力矩参考点是机身 `base_link` 原点（`state.base_position`），不是整机
质心也不是任何足端投影点，所以叫 body-wrench 而不是 centroidal wrench。接触力到关节力矩的
映射 `τ = -Jᶜᵀf` 是虚拟功等价关系，不是逆动力学。

不是 full WBC：不建立浮动基座运动方程 `M(q)q̈ + h(q,q̇) = Sᵀτ + Jᶜᵀλ`，没有惯量矩阵、
Coriolis/离心项、关节加速度变量，也不规划摆动腿轨迹。各轮是否参与分配取决于本步接触标志。
腿部 PD（`leg_pd`）和轮速跟踪力矩（`wheel_velocity`）在分配器之外单独算好，作为
`committed_torque_nm = leg_pd + wheel_velocity` 传入，分配器只优化“支撑+驱动”这一份力矩再
叠加上去，不是对 16 路力矩整体联合求解。它是经典 VMC 思路的收紧版：仍是“先算虚拟合力，再用
`Jᵀ` 转关节力矩”的两段式结构，只是第一段从最小二乘换成了约束优化。

## 3. 坐标系与符号

- 所有力都是 ground-on-robot；方向检查 `(foot_position - contact_point) · normal > 0`，反了
  直接抛 `ValueError`。
- 力矩参考点是 `base_link` 原点，力臂用 `wheel_contact_point - state.base_position`。
- 接触局部系 `contact_frames[i] = [rolling, lateral, normal]`（按列排列）：`normal` 取
  `wheel_contact_normal`；`rolling` 是机身前向向量在切平面上的投影并归一化；
  `lateral = normal × rolling`。D1 的四个轮子不能独立转向，四个接触共用同一个 `forward_world`
  投影是合理简化。
- 结果里的 `desired_wrench_world`/`achieved_wrench_world` 统一按 `[Fx,Fy,Fz,Mx,My,Mz]` 排列。

## 4. QP 决策变量、目标与约束

以下对应 `active_indices.size` 个当前接触（非接触轮子既不出现在决策变量里，也不出现在约束
里）。决策变量：每个激活接触 3 个分量，按局部系表示
`local_forces = [f_roll_0, f_lateral_0, f_normal_0, ...]`，四轮全接触时是 12 维。

目标函数：

```
minimize  1/2 · ‖diag(w) · (A f - w_d)‖² + 1/2 · ε · ‖f‖²,   ε = 1e-10
w = (1, 1, 1, 1/L, 1/L, 1/L),   L = wrench_characteristic_length_m（默认 0.25 m）
```

`A`（`wrench_matrix`）把局部接触力映射到世界系六维 wrench，`w_d` 是期望 wrench。残差前三个
分量单位 N、后三个单位 N·m，直接取欧氏范数会被力项主导，所以先用 `diag(w)` 把力矩残差按特征
长度 `L` 折算成力的量级。`L=0.25 m` 是开发种子上比较四个候选后的权重选择，不是几何标定值，
过程见[开发记录](contact_allocation_development.md)。第 6 节的跟踪判据用未加权的原始
`A f - w_d` 分量，不经过这个权重。

`ε‖f‖²` 是正则项：在接触数多于 wrench 自由度时（四轮 12 变量只需满足 6 个方程）保证 Hessian
正定，同时偏好范数更小的 `f`。四轮对称
场景下最小范数解恰好是均匀分配，这是问题对称性的结果；非对称接触几何下“最小范数”和“负载
均匀”不是一回事，正则项本身不是显式的负载均衡目标，也没有专门验证过非对称场景。

约束：

1. 单边法向 + 摩擦棱锥：`f_normal ∈ [0, normal_force_limit_n]`；`friction_matrix` 的四条线性
   不等式等价于 `|f_roll| + |f_lateral| ≤ μ·f_normal`，是圆形 Coulomb 摩擦锥
   `√(f_roll²+f_lateral²) ≤ μ·f_normal` 的保守内接近似，沿坐标轴方向边界重合，沿 45° 对角
   方向收紧到圆锥的 `1/√2 ≈ 0.707` 倍，位于分配器假定的摩擦圆锥内
   （`test_allocator_friction_pyramid_is_inside_the_coulomb_cone`）。`Bounds` 里另有一个用
   法向力上限（不是当前法向力）给出的框约束，只是粗略的数值边界，恒比 `friction_matrix` 更松。
2. 关节力矩上下限：`-τ_limit - τ_committed ≤ torque_matrix·f ≤ τ_limit - τ_committed`，
   `τ_committed = leg_pd + wheel_velocity`，`τ_limit` 就是当前的 `torque_limit_nm`
   （`plant.actuator_torque_limit_nm`），会随 `actuator_strength_scale` 变化，第 1 节已说明。

求解调用 `scipy.optimize.minimize`，设置 `method="SLSQP"`、`jac=True`、
`options={"maxiter": 80, "ftol": 1e-10}`，没有引入专用 QP
求解器（OSQP/qpOASES）。问题本身是凸的（二次目标 + 线性约束），但 SLSQP 仍可能数值退出。
目前没有和专用 QP 求解器比较过求解时间，也没有最坏情况耗时保证。

## 5. 接触雅可比：legacy 用轮体 COM，constrained 用接触点

legacy（`D1LegacyContactAllocator.solve()`）用 `state.foot_jacobian` 算腿部（hip/thigh/calf）
支撑力矩，这是 `mj_jacBodyCom(...)` 算出的轮体（`{leg}_foot` body）质心平移 Jacobian，不是
轮子和地面的实际接触点，两者相差约一个轮子半径（`0.087 m`）。驱动轮关节的力矩不经过任何
Jacobian，是直接用公式 `τ = F_longitudinal · r_wheel / 4` 算出来的。平地上轮心正下方就是
接触点，误差不明显；地形有坡度或轮子局部姿态变化时，质心和接触点之间的力臂会产生额外误差。

constrained 统一用 `state.wheel_contact_jacobian`，由 `mj_jac(..., wheel_contact_point[index],
foot_body_id)` 在“当前测得的接触点”（同一轮子上多个 MuJoCo 接触的平均位置）上求平移
Jacobian，`τ = -Jᶜᵀf` 这一步（包括驱动轮通道）都用这个 Jacobian，不再单独套用轮半径公式。
两个字段来自同一段 `_capture()`，并行保留；legacy 分配器仍使用 `foot_jacobian`。

## 6. 求解器接受判据、fallback，以及两个正交的结果字段

`solution.success`（SLSQP 自报是否收敛）单独不能决定要不要用这个候选解。`solve()` 会重新计算
候选解的无量纲违反比例：

```
candidate_force_violation_ratio  = max(0, max(lower-candidate), max(candidate-upper),
                                        max(friction_matrix @ candidate)) / normal_force_limit_n
candidate_torque_violation_ratio = max_i( max(|command_i| - τ_limit_i, 0) / τ_limit_i )
candidate_violation_ratio = max(candidate_force_violation_ratio, candidate_torque_violation_ratio)
```

（`command = committed_torque_nm + torque_matrix @ candidate`。）只要候选解形状正确、所有分量
和目标函数值都有限，且 `candidate_violation_ratio ≤ 1e-7`
（`D1_CANDIDATE_CONSTRAINT_VIOLATION_TOL`），就采用这个候选，`solution.success` 不参与这一步
判断。采用之后：`solution.success` 为真则 `status = CONVERGED`；为假但候选可行则
`status = FEASIBLE_NONCONVERGED`，`status_reason = f"slsqp_status_{solution.status}"` 保留
SciPy 原始状态码（`test_feasible_nonconverged_candidate_is_used_and_reported`）。

只有候选解非有限或 `candidate_violation_ratio > 1e-7`，才会丢弃 SLSQP 结果转向确定性
fallback：丢弃切向分量，只用 `lsq_linear(..., method="bvls", bounds=(0,
normal_force_limit_n))` 对法向力子问题重新求一次有界最小二乘，状态记为 `FALLBACK`/
`"solver_rejected"`。随后整体缩放法向力，使它与 committed torque 合成后落在剩余力矩范围。
若 committed torque 本身使问题无解，最后仍裁剪命令，并在 violation 中保留超限证据。
fallback 不依赖 SLSQP 中间状态，重复求解结果一致
（`test_allocator_uses_an_explicit_deterministic_fallback_when_constraints_conflict`），代价是
放弃局部切平面方向的独立跟踪；斜面法向本身仍可能包含世界系水平分量。

对外报告的 `max_constraint_violation` 用同一种归一化方式，在最终 `local_forces`（不管是被
采用的候选还是 fallback 解）上再算一遍取 `max`。legacy 分配器的 `max_constraint_violation`
也用同一种按 `τ_limit` 归一化的 ratio（`_max_torque_excess_ratio`），两条路径量纲一致，可以
直接比较。

`status`（`CONVERGED`/`FEASIBLE_NONCONVERGED`/`FALLBACK`/`NO_CONTACT`/`LEGACY`）只描述求解器/
fallback 走到哪一步，不描述解出的力有没有精确跟踪期望 wrench；这是第二个正交字段
`wrench_tracking_status`（`TRACKED`/`LIMITED`），拿 `achieved_wrench_world` 和
`desired_wrench_world` 的差重新判断，力和力矩分开：

```
force_tolerance_n   = 1 N     + 0.5% · ‖w_d[:3]‖
moment_tolerance_nm = 0.5 N·m + 0.5% · ‖w_d[3:]‖
```

同时满足才是 `TRACKED`。两个字段正交：`CONVERGED` 可能同时是 `LIMITED`（棱锥比圆锥更紧，或
法向力上限本身不足）；`FEASIBLE_NONCONVERGED`/`FALLBACK` 的解也可能是 `TRACKED`。

warm start：`D1ConstrainedContactAllocator` 用上一次求解结果的世界系接触力作下一次 SLSQP 的
初值，仅当 `state.wheel_contact` 与上次求解完全一致（`np.array_equal`）才复用
（`warm_started=True`）；接触集合变化则退回“期望力按当前激活接触数平均分配”的猜测
（`warm_started=False`）；`reset()`（被 `D1VMCController.reset()` 逐层调用）清空这两个内部
状态。无论初值来自哪里，候选解都要过同一套 `candidate_violation_ratio` 检查，warm start 不
放宽验收标准。

## 7. 三层证据：requested → allocated → MuJoCo actual

`D1ResidualEnv._info()` 暴露三层数据：

| 层 | 字段 | 来源 |
|---|---|---|
| requested（`w_d`） | `allocation_desired_wrench_world` | `D1VMCController` 按支撑/姿态 PD 算出的期望 wrench |
| allocated（`w_a`） | `allocation_achieved_wrench_world` | `wrench_matrix @ local_forces` |
| actual（`w_m`） | `measured_contact_wrench_world` | `D1Plant.measure_wheel_contact_wrench()`，直接读 `mj_contactForce` |

`w_m` 只是评估旁路，由 `D1Plant.step()` 在物理子步里采样、写进 `info`，控制路径
（`D1VMCController`、分配器、外环控制器）从不读取这个字段，只用 `state.wheel_contact`/
`wheel_contact_point`/`wheel_contact_normal`/`wheel_contact_jacobian` 这些接触几何和标志位。

三层派生三组误差，单位分别标注、不合并：

- `allocation_force/moment_error_norm`：`w_a - w_d`，分配器本身有没有跟踪上期望 wrench。
- `contact_force/moment_model_error_norm`：`w_m - w_a`。`contact_audit.py` 里标注为
  discrepancy（"Allocated-to-physics ... discrepancy"）而不是纯粹的模型误差：它混入了
  `committed_torque_nm`（腿部 PD、轮速跟踪力矩）在物理引擎里产生的地面反作用力、机身惯性、
  以及接触瞬态（滑动、多点接触、软约束）等作用，不能唯一归因到接触力
  模型（棱锥摩擦、单点接触、Jacobian-transpose 映射）本身。
- `contact_force/moment_tracking_error_norm`：`w_m - w_d`，端到端误差。

三者满足恒等式 `w_m - w_d = (w_m - w_a) + (w_a - w_d)`，向量本身精确成立，但模长只有三角
不等式 `‖w_m - w_d‖ ≤ ‖w_m - w_a‖ + ‖w_a - w_d‖`。据此诊断：若前两项都小，第三项必然也小
（被两者之和限制）。分别查看 `allocation_*_error` 与 `contact_*_model_error` 可以定位差异
主要出现在哪一段，但不能仅凭这三个范数断定动力学原因。

## 8. 力（N）和力矩（N·m）分开统计

`experiments.py` 的 `allocation_force_error_rms_n`/`allocation_moment_error_rms_nm` 是两条
独立的 RMS 序列，没有拼成六维向量算合成范数：力在百 N 量级，力矩在十 N·m 量级，强行合并会被
力这一项主导，力矩通道跑偏也很难在合成指标里看出来。和 README/评测协议里“速度、Pitch、高度
RMSE 分开报告”是同一个原则。

## 9. 五子步接触力平均与环境耗时上报

`D1Plant` 物理步长 `0.002 s`，控制步长 `0.01 s`，`physics_steps = round(0.01/0.002) = 5`。当
`step(..., measure_contact_wrench=True)` 时，每个物理子步都调用一次
`measure_wheel_contact_wrench()`，5 个样本按 `np.mean(...)` 汇总，`physics_sample_count` 记录
实际参与平均的样本数。整个控制周期使用控制步起点的同一个机身参考点计算力矩，避免把不同
参考点的力矩混在一起平均。这不是默认行为：`D1ResidualEnv.__init__` 里 `measure_contact_wrench:
bool = False`，此时只在 `step()` 循环结束后补一次单个物理子步的瞬时读数，
`physics_sample_count` 是 1。

`allocation_solve_ms` 同理需要显式打开：`D1ResidualEnv.__init__` 默认
`profile_allocation_timing=False`，此时 `info["allocation_solve_ms"]` 恒为 `0.0`，即便分配器
内部（`D1ContactAllocationResult.solve_ms`）真实记录了求解耗时；只有传入
`profile_allocation_timing=True` 才会把这个真实耗时写进 `info`。

目前唯一同时打开这两个开关的调用点是 `experiments.py` 的 `run_d1_rollout()`
（`measure_contact_wrench=True, profile_allocation_timing=True`），也就是
benchmark 与接触分配审计这两条评测路径。键盘演示直接读取 controller 的求解耗时，不受环境
的 `profile_allocation_timing` 开关控制；它没有开启五子步 wrench 采样。直接构造环境时需
显式选择开关，并核对 `allocation_timing_measured` 和 `contact_wrench_physics_samples`。

## 10. 代码入口、CLI、测试与审计脚本

```python
from wheel_legged_control.d1.contact_allocation import make_d1_contact_allocator
from wheel_legged_control.d1.env import D1ResidualEnv

env = D1ResidualEnv(contact_allocation="constrained")  # 或 "legacy"
```

`make_d1_contact_allocator(plant, mode)` 是唯一推荐的构造入口，按
`plant.nominal_total_mass_kg` 算法向力上限（`0.65mg`）并注入分配器，不建议直接实例化
`D1ConstrainedContactAllocator`/`D1LegacyContactAllocator`。

```bash
wheel-legged-d1-play --contact-allocation constrained
wheel-legged-d1-contact-audit --seed 21 --episodes 3 --output results/contact_dev
```

`play` 默认使用 `legacy`，通过参数切换。`contact-audit` 固定成对比较两种模式，不接受
分配模式参数；默认从开发种子 21 开始，运行 3 对回合。

```bash
pytest tests/test_d1_contact_allocation.py -q
```

覆盖范围：无接触时不发明支撑力（`NO_CONTACT`）；四轮对称站立时精确跟踪竖直合力；摩擦棱锥内
纵向力精确跟踪，且棱锥是 Coulomb 圆锥的内接子集；接触几何校验主动 `raise`；wrench 不可达时
返回可行候选、`wrench_tracking_status` 为 `LIMITED`；
`FEASIBLE_NONCONVERGED` 候选被采用且保留原始状态码；`committed_torque_nm` 真实压缩可行域；
约束冲突触发确定性 `FALLBACK`，重复求解结果一致；接入 `D1VMCController`/`D1LQRVMCController`
后能站稳/按指令前进；`D1LegacyContactAllocator` 数值上完全复现升级前的原始输出
（`atol=1e-10`）。

`contact_audit.py` 比较 legacy 和 constrained 两种模式各自完整的控制路径，不是只把接触分配器
单独换掉的因果消融：`hierarchical.py` 把接触分配模式和另外两处配置一起切换。外环用的
`identify_sagittal_model(control_dt, allocation_mode)` 是按模式分别用有限差分辨识出的两个
不同线性模型，外环 LQR 的权重 `_default_outer_r(allocation_mode)` 也按模式取两个不同固定值
（`D1_OUTER_R = 1e-8` legacy，`D1_CONSTRAINED_OUTER_R = 1e-6` constrained）。切换
`contact_allocation` 一次性改变了分配器、与之匹配辨识出的线性模型、外环 `R` 权重三样东西，
不能单独归因到分配器本身。

审计按 episode 统计的比例分两组。第一组来自 `status`，互斥：`allocation_converged_ratio`、
`allocation_feasible_nonconverged_ratio`、`allocation_fallback_ratio`、
`allocation_no_contact_ratio`，以及 legacy 专用的 `allocation_legacy_ratio`。第二组来自
`wrench_tracking_status`：`allocation_wrench_limited_ratio`，是独立维度，两者不冲突。
`PROMOTION_GATE` 对第一组有两条独立检查：`fallback_rate` 要求全部 episode 的平均
`allocation_fallback_ratio` 不超过 1%；`solver_degraded_rate` 再要求
`allocation_feasible_nonconverged_ratio` 与 `allocation_fallback_ratio` 的每回合总和不超过 1%，两条
都要通过。

## 11. 已知局限

- 摩擦棱锥是保守的线性近似，对角方向比物理摩擦圆盘更紧（约 0.707 倍），会把某些真实可行的
  合成切向力判定为不可行。控制器固定假定 `μ=0.55`，物理引擎的摩擦系数受域随机化影响，
  两者并不总相同；约束可行不等于真实接触一定不打滑。
- 只用单个“平均接触点”表示每个轮子（`wheel_contact_point`/`wheel_contact_normal` 是同一轮子
  上所有 MuJoCo 接触点的平均）。粗糙地形下一个轮子可能有多个真实接触点，各自受力方向不同时
  可以构成力偶，这部分贡献在“平均成一点”的近似里会丢失。
- 没有约束接触力的变化速率，每个控制步独立求解，相邻两步的解可以有较大跳变，只要跳变后力矩
  仍落在限值之内就会被接受。
- 延迟补偿只外推运动状态，接触点、法向和 Jacobian 仍来自旧快照。接触切换期间的过时几何
  没有被补偿；本轮接触审计使用无延迟的 oracle 状态。
- fallback 清零局部切向力，只保留法向力，切换可能导致驱动力突变。
- 正则项确实偏好范数更小的 `f`，但这只是最小范数偏好，不等价于“均匀分配负载”；两者仅在四轮
  对称场景下恰好一致，非对称接触几何下的负载分配没有专门验证过。
- 关节力矩限值不是恒定的 `±80/±12 Nm`：会随域随机化的 `actuator_strength_scale` 成比例缩放
  （见第 1、4 节）。法向力上限则按名义重量计算，不随执行器强度缩放。
- SLSQP 耗时在审计中报告 P99；它没有最坏情况时限保证。`maxiter=80` 是经验值。
  环境 `info` 的 `allocation_solve_ms` 只有显式打开 `profile_allocation_timing=True` 才会
  写进 `info`，默认读到的 `0.0` 不代表分配器没有耗时（见第 9 节）。
- legacy 只显式拟合竖直合力、Roll/Pitch 力矩并等分纵向力，constrained 拟合六维 wrench。
  审计可以比较整套配置的物理跟踪误差，但两种模式请求的 wrench 也不同；参见第 10 节的归因边界。
- `w_m - w_a` 是第 7 节说明的 discrepancy，混入了 committed torque、惯性、接触瞬态，不能单独
  归因到接触力模型；actual 层只是评估旁路，不参与闭环，模型误差和端到端误差都只能事后分析。
- 仍然是逐步静力学分配，不是轨迹级优化：每个控制步独立求解，不考虑本步选择对下一步可行域的
  影响，也不参与摆动相规划。
