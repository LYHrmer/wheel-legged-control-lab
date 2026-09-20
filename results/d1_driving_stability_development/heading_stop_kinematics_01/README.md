# Release 失败后的停车运动学诊断与下一独立实验合同

这是 gpt-6-astra / ultra 的只读分析和未实测控制方案。没有新增物理控制步、训练或控制实现；77 份冻结输入及旧实验未改写。仅本新目录包含诊断 helper、report 与说明。新阻尼不是已验证修复。

## 复算方式与边界

从 `/home/lyh/wheel-legged-control-lab` 执行，输出必须是不存在的新文件：

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.local-deps:src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/stop_kinematic_diagnosis_01/helper.py --input /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/heading_release_01 --output /tmp/stop_kinematic_recheck_new.json
```

实际使用 MuJoCo 3.12.0 的 `mj_kinematics`, `mj_comPos`, `mj_comVel`, `mj_collision`, `mj_objectVelocity`, `mj_jac`, `mj_jacBody`, `mj_crb`, `mj_fullM`；不调用步进、积分或 controller.compute。`mj_crb/fullM` 仅构造给定姿态的惯量矩阵。1600 个保存状态的 qpos/qvel/time 在计算前后完全相同；重建 base inertial-COM body-x 速度与日志的最大误差为零；速度分解闭合误差最大 1.00614e-16 m/s。report 记录 helper、输入轨迹、元数据、协议和源码 SHA，并核对源协议列出的现存输入。

`mj_collision` 只重建几何碰撞候选，未求解支撑力。没有候选时，使用“轮中心减世界 z 方向半径”的底点代理，并逐轮逐帧标注。因此 `point_tangent_mps` 不能统称实测接触滑移，也不能换算摩擦功。5–6 秒每条件 400 个轮帧中约 30–32 帧没有候选。运动学分解准确，并不等于独立的因果归因。

## 确定结论

固定 .5 m/s² release 改变了骤停激励，但没有解决晚段腿/机身回摆。两方向均继续失败；不调斜率、不扫参、不采用该尾段为默认。

| 指标 | 正向 bypass → release | 反向 bypass → release |
|---|---:|---:|
| t=4.00s 四轮实际平均力矩 Nm | −5.84022 → +.43966 | +5.67379 → −.60609 |
| 前一拍 t=3.99s 平均力矩 Nm | +.80132（两者相同） | −.91136（两者相同） |
| 4–4.5s body−wheel RMS m/s | .12429 → .06302 | .13340 → .07845 |
| 4–4.5s 腿运动分量 RMS m/s | .10064 → .05250 | .09646 → .05110 |
| 5–6s 腿运动分量 RMS m/s | .03387 → .03567 | .03676 → .03690 |
| 5–6s body vx RMS m/s | .03907 → .04792 | .05338 → .05463 |
| 冻结晚段最大绝对 body vx m/s | .06087 → .10602 | .08635 → .09319 |
| raw-goal 全程 vx RMSE m/s | .05954 → .06826 | .05992 → .07023 |
| 原方向最大停车延伸 m | .05407 → .10523 | .06779 → .12376 |

首拍力矩是这次日志中五个物理子步的实际 actuator_applied 值，全部与请求逐位相同，已不依赖旧诊断的 PI 重建推测。

release 首拍正向比例项从 bypass 的 −6.47093 降至 −.27553 Nm，而积分从 +.71895 只降至 +.71519，合成力矩仍为正；反向比例项 +.14129、积分 −.74738，合成仍为负。这解释了释放开始仍沿原方向驱动，不能把较长行程误称为刹车已经完成。积分随轮速误差正常下降；此处没有依据把积分复位作为下一改变。

在 t=5.00s，release 正向 body vx=−.10602、wheel rolling=−.00158 m/s，差值 −.10444 可分成腿 −.07658、base 角运动 −.01993、轮底/候选点切向 −.00792，轮轴几何项约 −.000002。反向对应 body +.09319、wheel +.00702，差值 +.08617 中腿 +.06999、角运动 +.02544、点切向 −.00926。小轮速没有表示小车体速度。

取保存状态 tick（不是下一端点）作相位比较：正向首次 body 反向从 tick437 推迟到462；反向从439推迟到465。第一轮反向速度峰从两方向 tick466 推迟到492/493；正向峰幅 .13289→.11390、反向 .11054→.09966 m/s，幅值确实下降，但峰后尾巴进入冻结的 tick500 晚段窗口。俯仰第一同向极值大致从437→464、434→457，先于反向速度峰约 .28–.36s；后续俯仰和 body 速度仍以约1.3s周期交替。平滑主要削弱并延后激励，没有显著改变后段腿运动 RMS。不能靠移动评分窗口掩盖这一点。

## 下一独立有界控制律实验（未实施）

唯一新增机制：**释放后纵向腿速度阻尼**。返回原 raw 命令直接执行链，完全关闭 release governor；轮速 PI、积分、腿位置目标、重力支持、现有姿态 PD、heading servo、残差动作和奖励均保持原式。

对腿 i，取已经发布的 `D1StateEstimate.foot_jacobian[i, :, :3]`（轮中心、非轮三关节）以及 `base_rotation[:,0]`：

```text
Jx_i = base_rotation[:,0]^T @ foot_jacobian[i,:,:3]
u_i  = Jx_i @ joint_velocity[4*i:4*i+3]
delta_tau_i = -b * Jx_i^T * u_i
```

只向12个腿关节添加该项，四个轮关节增量严格为零。不使用 ground truth、未来状态、body 速度误差、距离锚点、接触支撑权重或额外滤波。无需接触门控，因为它是关节相对运动阻尼；瞬时关节功 `sum(delta_tau_i dot qdot_i) = -b sum(u_i²) <= 0`。这是采样时刻的代数性质，不是10ms保持输入后整个离散闭环被动性的保证。

启用规则只定义停车模式：reset 后 inactive；仅在实际执行的相邻两拍 forward 命令发生 nonzero→exact zero 时激活；持续零命令保持 active；任何新非零命令立即关闭。不得用 tick400、case 名称或未来时长触发；初始零速 settling 不激活。状态只在 compute 一次推进，preview/prepare 不推进，不二次消费命令。此零策略实验中的 servo forward 与 raw forward 相等。元数据记录 latch/增益/schema；不得将其悄悄用于 PPO。

### 固定增益来源

使用 model 的名义起始姿态（保存 qpos[0] 的16个关节值与 `NOMINAL_JOINT_POSITION` 最大差值为0），构造“四轮中心固定、轮自转固定、base仅作单位x平移”的瞬时模式 g。每腿 `g_leg=-J_xyz_leg^-1 e_x`。由名义全惯量 M、原关节 Kp=80、Kd=3 和模型被动阻尼得：

```text
M_eff = g^T M g = 40.173203899817445 kg
K_eff = 80 * sum(g_leg²) = 2163.081163077078 N/m
B_old = 3 * sum(g_leg²) + g^T D_passive g = 83.81939506923678 N*s/m
B_critical = 2 sqrt(M_eff K_eff) = 589.5689972043975 N*s/m
b = max(0, B_critical - B_old) / 4 = 126.4374005337902 N*s/m per leg
```

固定使用这个数值及来源 SHA，不为物理跑结果重新选择 b。名义模式频率1.168Hz、原阻尼比.142；真实轨迹约.75Hz，差异说明该简化模式没有包括重力/支撑几何刚度、俯仰耦合、滚动和 wheel PI，**不能声称真实闭环达到临界阻尼**。这只是一个可证伪固定候选的物理设计依据。

在现有两条 bypass 轨迹上仅代数评估新增项，最大单关节增量8.691/8.755Nm，原请求加新增项的最大额定力矩比.51671/.50099；采样时刻新增关节功均非正。自由惯量的新增阻尼 `dt*lambda_max(M^-1 D_new)` 最大1.08416（标量纯阻尼 held-input 粗筛阈值2以内）。这不证明新轨迹不会限幅或离散不稳定；完整耦合闭环只有后续有界配对实测能判断。

### 实际 Opus 实现边界

新增子类/适配器，不修改冻结源码。可以新增 D1WheelLegController 子类，compute 调用 super.compute 恰好一次，保留原 PI 更新。对 `last_result.requested_torque_nm` 加腿阻尼，再按原顺序执行同样的关节额定力矩、位置外向和速度外向安全保护，不能把新增力矩加在已限幅输出后。诊断明确保留 base leg_pd 与 damping delta；如用 `dataclasses.replace` 更新原 result 的 `leg_pd_nm` 为合成腿反馈项，另记原字段，以维持 requested=leg_feedback+support+wheel 的加和身份。

disabled 分支直接返回 super 的结果，精确 bypass。enabled 但 inactive 也直接返回，确保停止前和无停止案例逐位一致。新的唯一 control_schema 禁止旧 checkpoint 误加载；只用 zero 残差。新增薄 heading env 子类在首次 reset 前替换 `_controller` 为原 adapter 包装的新 controller，沿用原环境 reset/step/control loop；不得复制整个环境或添加额外物理步。

非物理验证先检查正负 u 的耗散符号、轮通道零增量、开关/reset、preview幂等、原限幅保护、zero策略拒绝非零 action/模型加载、日志 T/T+1 与 disabled 精确复制。若设计复算或安全检查不成立，保留失败并停止该候选，不能临时减小 b 继续。

### 一次固定物理比较

主代理审阅后再由实际 Opus 编码；本文件没有执行该比较。两条件：原 bypass 与固定 b stop damper。只做与 release 实验相同的4个原 G1 profile：正/反停车各800步、正/负yaw冲击各1200步，seed55101，最高8000个真实控制转移/40000物理子步，无训练、无补跑凑时长、无额外条件。新目录、新协议；保留当前 release 失败与所有旧记录。

继续要求 bypass 对旧G1逐位复现；两条件停车 tick400 前逐位一致、无停止冲击全程逐位一致。forward原始命令在tick400归零，评分始终 endpoint COM vx 对该 executed interval 的 raw user命令，所有旧G1门槛原样保留：8秒完成、无非轮接触/摔倒/边界、实际roll/pitch≤10°、height RMS≤.015m、heading peak≤5°且RMS≤3°、全程raw vx RMSE≤.05m/s、endpoint tick500..799最大|vx|≤.03m/s、位置tick500..700累计平面路径≤.05m。

新增观察量仅用于解释：逐轮实际力矩、raw/base/damping/safe torque分解、u_i、新增功、激活状态、限幅/外向保护、PI前后、前0.5秒和5–6秒运动学分解、首次回摆相位、最大停车延伸、总反向路程。候选只有正反停车全部原门槛通过且其余配对检查通过才称停车候选通过。若任一失败，保留结果，不调 b、不加第二机制、不训练、不改评分时间；回到证据诊断。通过也不代表转向、障碍、提速或人工试驾已完成。
