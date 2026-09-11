# 固定种子的一次 value loss 尖峰核对

固定 `shared2_seed31000`，在该轨迹的 512 次训练调用中取 `train/value_loss` 最大项；没有查看 holdout 来选模型。结果为 `audit_index=167`（第 168 次 train）、`num_timesteps=86016`，日志文件 `updates/updates.jsonl` 第 168 行，对应 `updates/sample_000167.npz`。

| logger 指标 | 数值 |
| --- | ---: |
| train/value_loss | 0.5530358580872416 |
| train/explained_variance | -0.0544811487197876 |
| train/approx_kl | 0.008054859936237335 |
| train/clip_fraction | 0.0888671875 |
| train/policy_gradient_loss | -0.0028719143010675907 |
| train/std | 0.11786183714866638 |

相邻两次 value loss 为 0.00056770347237034（index 166）和 0.005928948041400872（index 168）。这描述的是一个局部尖峰。

这次 rollout 含 **一次地图边界终止并重置；证据不支持将它称为摔倒**。完整 rollout 是 128 个向量步 × 4 workers，对应全局环境转换计数 85505–86016；CSV 中共享累计 sample 标签为 85508、85512、…、86016（每个标签对应四行）。worker 0 在 rollout 内零基时间索引 39、sample=85664 终止：x 从上一条的 5.29976756 m 变成 5.30089683 m，越过 5.3 m 安全边界；`terminated=True`、reward=-1.9803207555024858、`reward_termination=-2.0`。此时 clearance=0.45385189 m，undesired_ground_contacts=0，roll_error=-0.02434315 rad，pitch_error=-0.00839280 rad。

`gae_episode_starts[40,0]=1`；完整 starts 阵列只出现这一处新 episode，`gae_last_dones` 四项均为 False。CSV 下一步 sample=85668 的 time_s=0.01、x=-3.79996211 m，确认已重置。`worker0.monitor.csv` 第四个 episode（零基 episode=3）长度为 5632，累计 worker 步数为 21416，21416×4=85664，恰与终止 sample 相同。`worker0_episodes.jsonl` 第 4、5 行分别记录 episode 3、4 的 reset 元数据，命令种子从 1234203258 切换为 2003296620。注意该 JSONL 记录 reset 元数据，本身不记录终止原因。

终止类别是依据**记录与冻结代码重建**，不是把未保存的 terminal_reason 字段当作已读到。`locomotion_env.py:492` 的规则先检查低高度、姿态或身体接触，然后检查 |x|≥5.3 或 |y|≥2.3。此处 straight 道路的最后 step 在 x=4.2 结束，x≈5.3 的地面为平面（`locomotion_terrain.py:37`、`:187`），所以记录中的 roll/pitch error 等于实际倾角，远低于 0.85 rad 摔倒阈值。结合高度与接触数，重建为 `map_boundary`。本次使用的 env、terrain、审计器、runner 四份源码 SHA 均与 study protocol 的冻结 SHA 相符。

**完整 rollout 与诊断子集要分开。** NPZ 的 `gae_rewards/gae_values/gae_episode_starts` 是完整 (128,4) 时间优先数组；`gae_last_values/gae_last_dones` 是四个 worker 的边界条件。NPZ 中直接命名为 `returns/advantages/old_values` 的数组只有 128 个元素，恰是 env-major 展平后的 worker 0 全部 128 步，不代表全部 512 个样本。这里用 NumPy 从完整 GAE 输入反向复算完整 returns；其 worker 0 子集与保存的 returns、advantages 最大绝对误差均为 0。完整 GAE rewards 与 CSV 原始 rewards 转 float32 后最大误差为 0，本次没有额外 timeout-bootstrap 奖励差异。所有 NPZ 数组、复算 returns/advantages 均有限。

完整 returns 方差=0.5420430898666382，Var(returns-old_values)=0.5715742111206055；按 `1-Var(R-V_old)/Var(R)` 复算得到 -0.0544811487197876，与 logger 完全一致。仅 worker 0 子集的 EV=-0.029777244289238114，与 JSONL 顶层 `explained_variance_old_values` 一致；两者不能混用。

终止点旧 value=3.774763345718384，lambda-return≈-1.980320692062378，advantage=-5.755084037780762。完整 rollout 的更新前 MSE=0.599718451499939；worker 0 的终止前 40 个样本占该平方误差总和的约 99.9301%。这为“此批次的终止及大幅变化的 return 目标与尖峰同时出现”提供可核对的数值联系。它不是证明该事件独自造成整个优化过程尖峰的干预实验，也不证明所有尖峰都是同一种原因。logger 的 0.553035858 是训练调用中各 minibatch/epoch 的均值，不能与更新前的 0.599718451 MSE 直接等同。

数值、终止前后 CSV 记录、worker 汇总与输入 SHA 在本目录的 `peak_diagnostics.json`。
下载原始数据后，可用本目录的 `inspect_peak.py` 复算，输出必须是尚不存在的新文件：

```bash
python results/d1_ppo_value_loss_case/inspect_peak.py \
  --study restored-artifacts/results/d1_budget_study/shared2_seed31000 \
  --output results/my_value_loss_peak.json
```

脚本只使用 NumPy 与标准库。`inspect_peak_original.py` 保留首次检查时的原始脚本，带有原机器
路径，仅作来源记录，不是移植入口。当前入口增加了路径参数与不覆盖保护，不改变数值计算；
记录中的绝对路径仍描述原运行位置，迁移后应比较数值及输入内容哈希。
