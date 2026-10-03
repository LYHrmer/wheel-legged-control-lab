# C35 连续身体／脚相调度：仅设计梳理

状态：**design only**。这份文件只基于已冻结源码和 C33 已存诊断汇总；没有 C35 控制器、合同、训练或物理尝试。待 C34 真实闭合及独立读回后，再决定是否提出新实验。

## 时间约束与真正的冻结点

[C33 已存诊断](/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01/continuation33/diagnostic_review33.json) 的 alpha=1、30 mm 左右两侧各计身体 `shift/recenter` 5.13 s，串行其余脚相 4.24 s，周期 9.37 s。后者是**该轨迹的实测串行占用**，不是足部动力学的普适下界。C33 仅缩短名义身体曲线时，身体计划缩短约 0.626 s、实得约 0.33 s；后名义等待增加约 0.296 s。仅消去已观测的 0.52 s READY 失败等待，条件算术上也只到约 8.85 s。因此若要争取 1–2 s 级收益，应使部分身体动作与脚相真正并行，而非继续调同一个 alpha。

[Fast._advance](/home/lyh/wheel-legged-control-lab/scripts/d1_fast_side_step.py:482) 只在 `shift` 或 `recenter` 更新 `pose` 与 `body_accel`；`unload/lift/swing/lower/load` 分支分别更新承载权重和脚参考，身体姿态保持上次值。[Fast._prepare_leg](/home/lyh/wheel-legged-control-lab/scripts/d1_fast_side_step.py:357) 一腿完成 `load` 后才规划下一次身体位移，再进入 `shift`。[Fast.load](/home/lyh/wheel-legged-control-lab/scripts/d1_fast_side_step.py:555) 只有 `t≥0.15 s`、四足接触和身体速度 `<0.08 m/s` 才进入下一腿规划；[shift→unload](/home/lyh/wheel-legged-control-lab/scripts/d1_fast_side_step.py:491) 又等名义身体曲线结束、动态裕度 `>0.02 m`、四足接触和速度 `<0.06 m/s`。这些串行状态机条件构成当前调度边界。[C31 lift/swing 重载](/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01/continuation31/sol31/side_skill31.py:314) 只替换空中脚参考时钟，没有推进身体参考；[C33 时间适配器](/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01/continuation33/sol33/side_timing33.py:45) 也只在原 `shift/recenter` 写入 S 曲线。C34 的参考速度力项仍只在这些身体相有效。

## 候选权限 1：已经准入的单腿空中期间，允许有限的身体参考续行

在原 `shift→unload` **先**达到当前四足接触、速度和动态裕度 READY 后，脚仍按原顺序单腿离地；状态相关控制量只决定**当前支撑三角形内**的有限身体参考位移／速度与下一身体位移的提前量。身体可在 `lift/swing/lower` 与接触恢复后的 `load` 内继续移动，下一腿仍不得在上一腿落地并满足原 `load` 准入前离地。这里允许的是原来被冻结的参考调度，不自动放宽原准入速度、动态裕度阈值或扭矩上限。由于运动可能跨越 liftoff，必须重新证明身体及脚的同步参考确实维持支撑与可达性；**保留旧代码中的数值阈值不等于继承旧工况的安全资格**。

可测的在线硬约束是每个控制周期与五个 native 子步保留单摆动腿、其余三足支撑接触；空中脚横向命令的当前整体轮形最低点和前一完整 native 区间都 `>0.012 m`、且前一区间该轮无地面接触；规划和实际 IK、关节范围及扭矩限幅保持原值。保留 [Fast 空中故障门](/home/lyh/wheel-legged-control-lab/scripts/d1_fast_side_step.py:469) 的三足接触丢失 `>0.12 s`、动态裕度 `<−0.004 m` 持续 `>0.10 s` 即中止，身体姿态／高度与关节速度限值，以及 [C31 的 native 横移核](/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01/continuation31/root31/record31.py:178)。力分配、摩擦／法向力裁剪、最终电机扭矩裁剪仍走 [Fast.compute](/home/lyh/wheel-legged-control-lab/scripts/d1_fast_side_step.py:606) 原路径。完成时四足落地、正支撑裕度以及任务的真实目标误差、滚翻／碰撞、保持和尾窗门均需独立重核。[Fast 所称 dynamic_margin](/home/lyh/wheel-legged-control-lab/scripts/d1_fast_side_step.py:460) 用的是 `COM − h/g × body_accel`，其中 `body_accel` 是**参考加速度**，不是实测 COM 加速度或真实 CoP。沿用其故障门及原 `0.015 m` 参考加速度预算不构成连续运动的动态稳定证明；仍须用 native 接触点、实际反力/法向载荷及完整状态轨迹独立验证支撑与响应。

原 `shift READY` 的“到计划时间才切相”和低速等待、`load READY` 的时间/低速等待，既有保守排程成分，也保护脚离地边界；不能把它们称为纯物理安全定理，也不能悄悄删掉。若未来为连续跨 liftoff 的身体运动而**替换**低速或提前切相条件，就不再是上述同权限比较，需把替换项单列并重新取得完整 native/接触/跌倒证据。最小 fixed 可行性路线是在**同一可行动作与投影约束**下先做确定性的相位—裕度—速度调度表，独立证明左右方向各自的全时序安全和质量；只有它仍留出状态依赖差异，才考虑 RL 选择每次身体参考的大小或时点。C33 的 4.24 s 脚相占用给出最多可重叠的观测时间，不保证能实际节约任何指定秒数。

## 候选权限 2：单脚三维参考的有证据相位重叠

[C31 lift/swing](/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01/continuation31/sol31/side_skill31.py:374) 已允许在真实轮形 clearance 与完整前一 native 区间无接触后，脚上升尾段开始横向参考；但 [其记录](/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01/continuation31/sol31/side_skill31.py:470) 明确 `early_lower_enabled=False`、原因是尚无未来参考轮形证书。可研究的第二个状态相关权限是在脚仍只有**一腿空中**时，由实测轮形、三足接触、落点 IK 和余下三维扫掠 clearance 决定横向尾段何时与受限下降开始重叠，而不是按固定 swing 全时长后才下降。横向位移和下降同时发生的每一步都要保留当前及前一区间的整轮 `>0.012 m` clearance、无误触地和原脚参考速度/加速度/关节限幅；只有横向结束、稳定接触与 touchdown dwell 后才恢复承载。碰地 probe 速度仍为 [Fast 配置](/home/lyh/wheel-legged-control-lab/scripts/d1_fast_side_step.py:109) 的 `0.030 m/s`，原落地与取消回收门不变。

这个权限的 fixed 比较器应在**相同扫掠证书、相同提前下降权和相同限幅**内，使用确定的最早安全触发规则；RL 若参与，只能用当下状态选择不超过投影器许可的下降起点或脚相进度。原脚速 `0.16 m/s`、单脚顺序和 touchdown dwell 均保留，故单靠此项未必有 1–2 s 空间；应先用已存相位占用和未来证书的可行窗口估计上界，再决定是否值得物理尝试。若需要允许两脚同时离地或提高脚速才能显著提速，那就是另一套支撑/限幅权限，不能称为本候选的成功。

两候选都不能把“保持 native 故障门代码”当成已证明物理安全。最小证据顺序是：原源码与新权限差异、同权限强 fixed、逐控制和逐 native 的接触/整轮 clearance/姿态/关节/扭矩/碰撞/支撑复核、完整任务与取消/保持复核。此处不指定新试验次数、训练预算、超参或 GO；C34 结果闭合前不作下一合同选择。
