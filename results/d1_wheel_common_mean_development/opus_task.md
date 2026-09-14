# 冻结任务：只约束四轮共同高斯均值

设计：GPT-6-astra / ultra；新增核心交 Claude Opus 实际编写，根 agent 审阅/测试并执行训练。本轮不改冻结 `src/**`、`87525c3` 的旧逻辑、旧权重或旧 formal 结果。

## 核心模块

新增 `scripts/d1_wheel_common_mean_policy.py`，公开 `D1WheelCommonMeanPolicy(ActorCriticPolicy)`。唯一算法参数为 `wheel_common_mean_limit: float | None`，候选固定 **0.05**；`None` 为精确无约束对照。只接受 8 维、浮点 `Box([-1,1])`、`DiagGaussianDistribution`、非 gSDE、无分布 squash。拒绝 bool、非有限或非正的 limit；不额外施加 `c≤1` 的参数上限。

审阅补充：公开 helper 还应按输入 Tensor 的 dtype 拒绝超出正规有限浮点数表示范围的 `c`，策略构造按默认 float32 验证，避免有限 Python 标量转换成零或无穷后产生 NaN。这是输入数值有效性检查，不改变本轮 `.05` / `None` 的数学或实验条件。原任务卡曾误加 `c≤1`，已按实际 Opus 请求纠正。

设 `raw = self.action_net(latent_pi)`，前四项是腿伸长 mean，后四项为轮速 mean。

```text
m = mean(raw[..., 4:], dim=-1, keepdim=True)
b = c * tanh(m / c)
wheel_mean = raw[..., 4:] - m + b
mean = concat(raw[..., :4], wheel_mean)
```

`c=None` 时必须直接使用 raw，不消耗额外 RNG，不新建可训练层，也不改权重初始化。`c=.05` 保留所有腿 mean 和任意两个轮 mean 的差值；只改变四轮共同 mean。`m≈0` 的局部导数为 1，避免额外缩小正常局部动作梯度。物理对应的共同轮速 mean 上限为 `4c=0.2 rad/s`，理想纯滚动线速量级 `0.087*0.2=0.0174 m/s`，是当前任务的试验先验而非机器人安全界限。

覆写 `_get_action_dist_from_latent()`，把上述 mean 和**原** `self.log_std` 送入**原** `DiagGaussianDistribution.proba_distribution()`。样本仍为完整 8 维 Normal，原 SB3 rollout buffer 存储原始未裁剪样本及其 log_prob，环境继续原生 action clipping。不得对采样动作去均值/投影，不加入 tanh Jacobian，不把高斯样本换成确定性控制，不修改 GAE/PPO loss/奖励/观测/动作物理尺度。

限制的是**裁剪前高斯均值**，不是每个随机样本或实际车速。个别 mean 超过 Box 后，原生 clipping 可能改变执行后的共同 mean。因此 runner 应同时记录 raw mean、约束后 mean、`predict()` 动作/最终执行动作、原生 clipping 比例。若这一差异实际出现，则在结果中暴露；本轮不再暗加差速限幅或另一投影。

保存构造参数，支持标准 PPO 保存/重载与独立 policy 保存/重载。模型语义放在新 protocol、policy kwargs 和 checkpoint `extra` 中；底层 82/8 观察和物理动作语义不变，不伪造新传感器能力。实际采用源码、Opus 原始响应和本地修复分别记录。

## 数学与集成验收

根 agent 编写定向测试，不由 Opus 自称已经测试：

1. 同一 raw tensor 下，腿 mean 精确相同、轮两两差在浮点误差内相同、共同 mean 等于 `c*tanh(m/c)` 且绝对值不超过 c。
2. `None` 分支与同权重原 SB3 高斯的均值、方差、样本/RNG、log_prob 精确一致。候选不新增参数或改变初始化 RNG。
3. Normal 样本的 log_prob 与独立 `torch.distributions.Normal(mean, std).log_prob(action).sum(-1)` 一致；检查没有对样本作 tanh 或补错 Jacobian。梯度有限且腿/差速梯度有效。
4. float32/float64、批输入、无效 limit/动作空间/gSDE、构造参数 roundtrip。SB3 的 forward、evaluate_actions、get_distribution、predict 全使用同一 mean 语义。
5. 真实小 PPO 更新、保存和合法 sidecar 重载后同 observation 动作一致，并真实执行至少一拍。训练 logger 正确区分 rollout train calls 与 `_n_updates` 优化 epoch。

## 单变量训练对照

固定原课程 wrapper、32 s 回合、100 Hz、82 维当前观察、8 维独立残差、oracle 源、原基座/奖励/执行器。训练条件仅 **curriculum**，不再把 flat/mixed 当本轮实验因素。命令沿用原 `forward_command`：0.5 s 站立、0.5 s 线性升至 0.25 m/s，yaw=0、clearance=0.455 m。

训练种子固定 **48001、48002、48003**。每个 seed 各训练 `None` 和 `0.05`，每模型 **16384 个实际环境转移**，总计 **98304**。每对使用相同初始化参数、环境随机流和同样 PPO 参数；记录初始参数摘要/哈希，课程按完整 16384 预算四分位调度，从第一步起不改预算。

其余 PPO 参数全部沿用 `run_d1_locomotion_experiment.PPO_SETTINGS`：n_steps=128、batch_size=128、n_epochs=4、原学习率/gamma/lambda/网络/std。每模型应为 128 次 rollout 更新、512 个优化 epoch。最终模型固定为预算终点，不按最佳回报挑 checkpoint。原始高斯随机流在条件相同时对齐，轨迹因动作不同而分化属于干预本身。

先执行 seed48001 的两组**完整**16384及开发检查，再决定是否按原固定协议执行另外两对，避免 4096 中途改变课程分位、恢复 RNG 或挑 seed。如果发现数学契约、非有限物理或错误统计，先停止并修复，旧结果保留；若仅性能不佳，不以替换 seed 或偷改 c 的方式抹掉负结果。额外调参必须开新实验身份和最终测试集。

## 预声明评测

评测 JSON 已在 `evaluation_protocol.json` 固定，SHA256：

```text
dab3336f6977dacaf5766fa2b159ca1a782dd151e1eee73515f4a68e0ef08d75
```

- 开发：`dev_flat` 6 s；`dev_straight_road` 32 s。均 seed55101，沿用已诊断命令与场景。
- 最终：新 `right_offset` 参数 + 0.22 m/s 直行；新 `diagonal` 参数 + 0.27 m/s、yaw 0.01 rad/s；均 32 s，seed46171/46183。不是只换 oracle seed，也未复用旧 formal holdout 身份。
- 所有 case 的 command 都按 JSON 的 settle/ramp 同时从零升至目标 vx/yaw，高度恒定。不得把训练命令函数直接塞入 final 而忽略各自的目标参数。
- 每个 case 先固定一个真正 zero-residual 基线；然后逐一评估所有最终策略。相同初态、时间和源，无 reset/retry，首次终止结束。开发决定固定后才一次性打开最终 case 结果；读过后继续修改则该批结果降为开发资料。

报告每个 seed/case 的 vx/yaw/height RMSE、原回报及奖励分项、进度、终止/持续时间、功率/限幅、共同 mean 和实际执行共同分量，并保留逐步 trace。用每个训练 seed 的配对差比较，不把 3200 控制步当独立样本。不能把三个训练 seed × 两个 case 混成六个独立策略样本。

成功解释预先限定：三对最终模型在两项最终任务平均 vx RMSE 上方向一致优于无约束 PPO，且不增加失败，才可称本轮稳定改善速度跟踪；原回报、yaw、height 与 zero 的差值必须一起展示。速度变好但其他项变差，报告取舍；只追近 zero 而未超过其任务表现，报告缓解退化。本模块没有训练 GUI 台阶、跳跃或横移。
