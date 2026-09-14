# PPO 实现机制独立审计

本次没有发现能解释策略退化的接口实现错误。现有证据表明，保存和加载的是实际训练后的策略，
它确实学出了较大的非零残差；速度变差不能归因于误加载初始模型、动作重复叠加、观测错拍或超时 bootstrap 丢失。
这不等于证明整个学习算法没有问题，也不能据此认定某个超参数是退化的唯一原因。

本审计仅读取源码、模型、训练日志及已完成的四策略记录。重新执行模型推理，并用两次合成终止回放
检查已安装 SB3 的数据传递；实际物理步和优化器更新均为 **0**。71 份源码、3 个模型及其 metadata、
训练 progress/episodes 和已有最终检查文件保持不变。完整数字见 `check_01/report.json`，协议见
`check_01/protocol.json`。本机 SB3 2.9.0 的相关源码 SHA256 见 `library_provenance.json`。

| 待排查机制 | 源码位置（仓库相对路径、起始行） | 实际检查与结论 |
| --- | --- | --- |
| 课程 wrapper 改写奖励或动作 | `scripts/d1_course_curriculum.py:258` | wrapper 验证形状后原样调用 base.step，再附加课程信息，返回原 observation/reward/terminated/truncated；没有额外残差或 reward 缩放 |
| independent8 映射错误或重复残差 | `src/wheel_legged_control/d1/locomotion_env.py:235`、`:466`；`control_loop.py:278`；`wheel_leg_controller.py:357` | independent8 的索引是恒等映射；每拍一次 compute；腿 extension 和轮速分别只加一次残差。loop/controller 的重复 clip 是幂等限幅，不是重复相加。已存 12,800 步 policy/raw/applied action 全部逐值相同 |
| actor 使用错拍观测 | `src/wheel_legged_control/d1/locomotion_env.py:321`、`:479`；`control_loop.py:246` | 3 个 PPO 的 9,600 个记录动作由对应 obs[k] 重推理，最大差为 1.79e-7，符合批量与逐次浮点计算差异 |
| 上一动作或命令编码错位 | `src/wheel_legged_control/d1/locomotion_observation.py:24`、`:40` | 四例 obs[k+1,74:82] 与 action[k] 完全相同，初始上一动作全零；obs[k,38] 与本拍 vx 命令除以 0.6 后完全相同 |
| reward 使用下一拍目标 | `src/wheel_legged_control/d1/locomotion_rewards.py:51` | reward 取 transition.decision 的本拍目标、步后 truth 及实际执行器子步功率；下一观测的 prepare 不替换该 transition。普通项按 dt 积分，终止成本只计一次 |
| time_limit 错误当作终止或 bootstrap 到 reset 状态 | `.local-deps/stable_baselines3/common/vec_env/dummy_vec_env.py:56`；`common/on_policy_algorithm.py:233` | 实际 SB3 合成回放保留 terminal_observation：raw reward 0.016693 的超时样本加入 γ×V_terminal 后为 3.444346；同一数据标成真正 terminated 时仍为 0.016693。没有读取 reset 观测作为该终点价值 |
| 有限控制域退出被错误 bootstrap | `src/wheel_legged_control/d1/locomotion_env.py:511` | 控制域退出返回 terminated=True、truncated=False，缓存终点观测不参与 TimeLimit bootstrap。此次四个真实最终回合均为正常 time_limit，未实际触发该失败分支 |
| 没有训练或模型加载错误 | `scripts/run_d1_course_curriculum.py:108`；`src/wheel_legged_control/d1/locomotion_checkpoint.py:229` | 三个文件内 num_timesteps=16384、_n_updates=512；metadata/hash/空间和底座契约检查已通过。重推理匹配已保存轨迹 |
| 最后 CSV 只有 508 次更新 | `.local-deps/stable_baselines3/common/on_policy_algorithm.py:323` | SB3 每次先 dump_logs 再 train，最终四个 epoch 没有后续日志行；模型里确有 512。属于末次日志缺失，不是少训练 |

初始模型按相同 seed=47000、网络、PPO 设置和本机版本重新构造，未调用 reset、step、learn 或 train。
在相同 zero 回放观测上，初始确定性动作 RMS 为 **0.00331**，训练后的 flat/mixed/curriculum
为 **0.29207 / 0.29861 / 0.28596**。它们相对重建初始模型的参数 L2 差为 **1.80 / 1.82 / 1.77**。
初始模型不是严格零策略，其标准差为 exp(-2)=0.13534；训练后各维仍约 0.132–0.135。
这里的初始化是按协议重建，并非归档的训练前快照；没有重新执行其物理回合，不能把小动作 RMS
写成已验证的初始控制性能。

有证据支持进一步检查的学习限制如下：

- **critic 的相对拟合较差，但没有数值发散证据。** 三组 127 条有效训练日志里 explained variance
  为负的次数是 121/118/118；最后 value loss 为 0.001138/0.000814/0.001115。所有日志数值有限，
  已保存策略的回放 V 约 3.10–3.57，和每拍约 0.018、γ≈0.995 的折扣量级相符。
  explained variance 的分母是目标方差，低方差回报也会放大负值；现有日志没有保存该方差、每拍训练
  advantage 或 value target，不能仅据负值认定 critic 发散或 reward 尺度写错。
- **时间尺度值得受控验证。** `scripts/run_d1_locomotion_experiment.py:74` 设 γ=exp(-0.01/2)，
  对应 2 秒折扣衰减；GAE λ=0.95 的 γλ 衰减约 **0.178 秒**，rollout 只有 **1.28 秒**。
  这描述的是 TD 残差权重衰减，不代表策略只能考虑 0.178 秒，长时信用还依赖 critic。
- **本次训练几何曝光很有限。** 16,384 步只有 5 个完整 32 秒回合加 384 步。
  curriculum 的第 3 级只有一个完整回合，最后 384 步仍在平坦起步带、nonflat=0。
  mixed 在这个固定种子实际抽到 2/2/2/3/3/0 级，不能把它写成四级均匀覆盖。
  这是按 reset 切换和有限抽样的实际结果，不是地形在回合中意外变化。
- **当前 reward 在已测回放中确实更偏好 zero。** flat 相比 zero 的高度项改善约 +0.798，
  但速度项损失约 -4.574，最终 return 也较低。不能把这次结果直接解释为“奖励把慢行当成更好”。
  部分损失更可能来自有限数据下的策略更新，但具体因果仍待受控实验。

可证伪的后续检查是：保存初始化 checkpoint 与每轮训练 value/return/advantage 方差，检查负 explained
variance 来自低目标方差还是持续预测错误；在固定随机数、预算、任务和评测条件下单独改变时间尺度，
确认速度退化是否仍然出现；把 raw sampled action 与 SB3 clipped action 都记录下来，区分探索和确定性执行。
上述都没有在本审计中实施。动作通道消融和新参数化由其他任务负责，不与本审计重复。

可复现 CLI（输出必须为新目录；只有读取/模型推理/合成终止回放）：

```bash
PYTHONPATH=.local-deps:src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -B audit_ppo_mechanisms.py --repo /path/to/wheel-legged-control-lab \
  --development /path/to/development_01 --final-check /path/to/final_policy_check_01 \
  --output /path/to/new_audit
```
