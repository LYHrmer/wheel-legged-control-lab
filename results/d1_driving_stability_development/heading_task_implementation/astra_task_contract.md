# 下一实施：85维航向目标任务

方案：GPT-6-astra / ultra，2026-09-14。核心交实际 Claude Opus，根线程集成与测试。该版本是任务定义与可观测性修正，不包装成单因素 PPO 算法收益。所有77项冻结源不动。

## 决定与当前证据

当前14回合航向外环 probe 已完成：六模型的完成数由1/6变4/6，六模型heading误差均改善，仍有速度、姿态或倒地问题。直接清除轮差异会损失多数模型的高度表现。下一步固定已经验证有贡献的P/D外环，让**新训练**看到并优化航向参考，而不是继续给全部失败旧权重叠PI、差速限制与横向纠偏。

本版本不加PI、不改8维动作、不加地形前视、不加cross-track奖励或反馈。横偏仍是G1验收指标，但任务名仅叫heading tracking；只有航向达标而横偏仍普遍失败时，再单独定义路径版本。

## 新增核心与构造

新增 `scripts/d1_heading_tracking_env.py`，类 `D1HeadingTrackingEnv(D1LocomotionEnv)`。

- 固定baseline=`wheel_leg`、action_mode=`independent8`，动作仍原float32 Box[-1,1]^8，物理尺度/低层控制器/执行器不变。
- 保留原terrain、episode_seconds、command_mode、provider_config、actuator_config、randomization、reward_config、command_source及wheel_leg_control的明确构造参数，不改其语义。`command_source`表示**用户**命令；None时由原有schedule提供用户命令。
- 外环固定kp=2.0、kd=.4、limit_rps=1.0，调用已有 `d1_heading_reference.py`。
- 新reward参数 `heading_reward_weight=.5`、`heading_sigma_rad=math.radians(5)`。有限实数、拒bool；weight范围[0,1]，sigma范围[1e-3,pi]。第一训练协议固定默认值，不边跑边调。
- 新observation_space为float32 Box[-5,5]^85；原82前缀逐值保留，含**servo命令**的原编码。额外3维是 `[sin(psi_ref-psi_est), cos(psi_ref-psi_est), user_yaw_rate/.5]`，与既有clip和dtype一致。
- schema：task=`d1-heading-reference-task-v1`，observation=`d1-proprio-servo82-heading85-v1`，reward=`d1-heading-goal-servo-rate-v1`。action/source/低层controller/control-loop schema保持其真实未变值。reference schema引用现模块常量。

## 最小继承缝与时序

不得复制base的完整reset/step、动力学、奖励或执行链。覆写_prepare处理参考/servo；reset和step在super返回后统一扩维和补充任务信息。

### _prepare

1. t=`self._steps*self.plant.control_dt`。若同一loop身份/同一tick已有成功prepared context，直接返回该cached decision的原82编码，不再调用用户callback或推进参考。否则用户命令取 `self.command_source(t)` 或 `self.schedule.cmd_at(t)`，每拍仅消费一次；准备后到达的新输入只在下一tick生效。
2. 从 `self.loop.provider.read()` 读同拍已发布缓存，它不推进物理或滤波。
3. 若 `self.loop` 是新reset创建的对象（身份与上一已登记loop不同），以该初态的实测yaw、t=0重置HeadingReference。这样不需在base验证reset options之前修改参考；无效schedule仍按原契约被拒绝。
4. `reference.advance(user.yaw_rate_rps,t)` 左保持积分上一已执行区间，再以同拍state计算heading_feedback。servo只替换yaw_rate；vx和clearance保持用户值。
5. 调用 `self.loop.prepare(servo)`，成功后登记不可变的本拍prepared context：user、servo、reference及用于观察的state/heading。返回**原82编码**；_prepare本身不返回85。
6. prepare失败时不能把未成功准备的next context覆盖成可执行context。reference内部可能已走到下一参考，但base会终止该回合；后续显式reset建立新reference。

### reset

先调用super.reset(seed,options)，保留原验证、随机流、实际初态和元数据产生过程。它返回82；用成功准备的初态context追加3维并返回85。补充新task参数/参考初态到episode_metadata，返回info内的episode_metadata也必须是补充后的同一实际内容。

不要在super.reset之前预先推进参考，不重复reset plant/provider，不消耗额外随机数。初态航向从当前状态取，不硬编码0；本版本已有base固定spawn也不影响此原则。

### step

1. 调用super.step前锁定本拍prepared context。
2. 调用super.step(action)一次。动作原样交base，原生clipping、物理转移、失败、timeout不变；base异常继续抛出，不伪造训练样本。
3. 正常返回的82来自已准备的下一拍：用next context追加3维。
4. 特殊 `info['terminal_observation_source']=='last_executable_decision'` 表示base遇下一参考域退出，返回的是最后可执行的旧82。此时必须用**step前旧context**追加3维，仍返回85，并保留terminated=True、cached-terminal标签；不能旧82+新参考混拼。不能使该终止被bootstrap。
5. 任务奖励和指标用 `self.last_transition.truth` 的真实终点。actual_dt取receipt.end_time_s−start_time_s；`psi_ref_end=wrap(psi_ref_before+actual_dt*user_yaw_rate_before)`。不使用下一_prepare已经登记的用户命令评价本拍。

## 奖励与标签

保留base所有servo追踪奖励项，只增加一个有界非正项：

```text
e = wrap(psi_truth_end - psi_ref_end)
heading_goal = dt * w * (exp(-(e / sigma)^2) - 1)
reward_terms = original_servo_reward_terms + {heading_goal: heading_goal}
returned_reward = sum(reward_terms.values())
```

新增项范围[-dt*w,0]，误差0时精确为0；本版本没有新的终止惩罚、动作惩罚或额外成功bonus。旧velocity/height等项不变，旧tracking_yaw仍以servo yaw为目标。它不是原用户的yaw-rate得分。

保留 `servo_reward_terms` 和 `servo_reward` 以便核对；`reward_terms`为实际返回奖励的分解。info增加命名清楚的 `heading_task`，至少包含user_command_before、servo_command_before、reference_heading_before/after、heading_error_after、user_yaw_rate_error_after、heading_goal_reward，以及新增观察所对应的时刻/terminal来源。若重复字段，值必须一致。

观察用指定provider state，任务评价/训练reward用明确标记的终点真值，不能让真值偷入3维actor追加项。首次oracle训练与以后estimated评测分别记录source schema。

## 元数据与合法重载

- 新episode metadata保留原字段，增加 `heading_task_config`（固定外环参数、reward参数、reference schema、追加观察字段顺序/归一化、时序和reward公式版本）及独立的 `heading_reference_initial`（本次实际参考初态）。不得用extra覆盖旧兼容字段。
- 新类的task/obs/reward schema使旧82模型自然无法载入，不补零/截断绕过校验。
- 仍使用原 `write_checkpoint_metadata`。新增一个很薄的 `load_heading_policy` helper：反序列化模型前，先比较metadata.recorded_episode.heading_task_config与运行环境配置，再调用原 `load_locomotion_policy`。原loader不会自动检查新增外环gain或reward参数，必须补这一层。
- 配置比较不含本次初态、地形/回合seed，否则会阻止正常的新场景评测；这些作为实际episode设置单独记录。
- source SHA、模型/sidecar SHA、PPO policy kwargs照常记录，真实模型/动作一致性重载检查不得省略。

## 根线程验收

1. 实际MuJoCo reset/step返回85维finite float32；原82前缀与同decision的旧encoder相同。
2. 在相同servo观察但不同合法heading/reference场景中追加项确实不同，reward随真实heading误差改变；验证该输入/目标被训练消费。
3. 参考使用上一已执行user命令的左保持；用户命令变化边界、重复prepare、reset、反向旋转跨±pi均正确。
4. 无效reset options不得推进参考/随机流/物理；无效动作或数值异常不能增加实际转移计数。
5. 特殊base terminal fallback仍返回85维旧context，terminated/truncated和terminal来源标签正确；正常timeout的next observation可按原逻辑bootstrap。
6. 奖励有界、逐项和与返回值一致；终点真值/参考对齐，不把servo_rate误差改名为user_rate误差。
7. 小PPO更新、保存、helper重载、同obs同确定性动作并真实执行一拍；旧82模型/不匹配外环配置被拒绝。
8. formal77全部保持原SHA。核心原始Opus响应与本地修复分别存档。

## 课程与16k→65k训练

新wrapper可以继承现有课程wrapper并只改_build_env，使其建立新85环境；不要修改旧脚本的_build_env。绝对课程分母固定16384：在回合reset时按min(3,4*actual_transitions//16384)决定级别，16384后继续level3。最终预算一开始固定65536，与课程分母分离。

至少3个事先声明的训练seed，fresh85模型，原PPO设置、8动作、原探索std均保留，先不同时改PI/动作参数化/GAE。旧82+heading闭环与新环境zero作为可运行参考，但新旧任务奖励不可直接当同目标回报比较；用共同G1指标判断。

建议第一seed在16384保存checkpoint并由**独立进程**做开发检查，数学/数值错误停下修复，单纯性能不足仍按冻结协议续到65536并保留失败。可用单次learn(65536)+保存callback；或保持同一模型/env活进程执行learn(16384)，再learn(49152,reset_num_timesteps=False)，中间不得close/reset训练环境，不能在训练进程加载/评估模型扰动RNG。不要只有weights恢复却声称与连续轨迹相同。

保存callback和评测不改变课程预算。后两seed同配置，所有16k/65k点全部报告，不能以第一seed结果偷换seed或奖励。新的留出集另行冻结；已有final_01现在只能作为开发资料。

若本轮沿用forward_command、user_yaw全为0，训练身份是直行heading hold；新增user-rate输入没有非零训练样本，不能宣称已学习任意转向。后续转向命令课程与验证单独完成，未验证的GUI操作域使用已通过的经典/zero基座。

完成这一模块和一次训练仍不自动通过G1。必须实际达到astra_plan.md中直行/停车/转向/横偏门槛，并使GUI采用已经通过的控制路径；然后继续台阶与真正跳跃。
