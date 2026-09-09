# D1 残差反馈诊断

关闭旧 PPO 的竖直输出后，三个种子的高度 RMSE 都下降了约 2.7 mm。原始回报随之增加，但速度改善很小，6 个名义场景的质量通过数仍为 3。这支持继续做一维纵向残差训练，尚不足以说 RL 已经超过基线。

本次只加载 mixed 的最终 checkpoint，种子为 0、1000、2000。模型没有更新。148 回合均运行满 4 s，耗时 264.68 s；每种方法都保留同一套 LQR/VMC 反馈。

## 实验范围

[协议](protocol.json)在开始仿真前写入。名义场景取既有 tracking-v2 development 的 01、04、05、10、11、22：一个平地场景、5/10 mm 起伏各两个相位，以及 +4° 坡。扰动对照只在 04、10 上进行，分别施加初始俯仰 ±0.02 rad，或从 2.0 s 开始施加 ±20 N、持续 0.1 s 的世界 x 向推力。两个因素分开测试。

零残差和固定补偿各记录 14 回合，不按模型种子复制。每个 checkpoint 的名义动作先在同一场景回放，全部 18 组都逐步复现：qpos/qvel 字节哈希相同，检查的物理量差为零。之后再在相同扰动下比较实时 PPO 与该名义动作序列。回放时仍有 LQR/VMC 反馈；只有 RL 层失去在线状态反馈。

下表是 6 个名义场景的回合指标均值。学习策略先按种子汇总，再对三个种子等权平均；零残差和固定补偿各只有一组。

| 残差方式 | 速度 RMSE m/s | 最后 1 s RMSE m/s | 高度 RMSE mm | 原始回报 |
|---|---:|---:|---:|---:|
| 零残差 | 0.11252 | 0.09610 | 13.241 | 1537.23 |
| 固定 Fz=+20 N | 0.11190 | 0.09270 | 16.038 | 1478.14 |
| 完整 PPO | 0.11183 | 0.09435 | 16.002 | 1478.52 |
| PPO 仅保留 Fx | 0.11064 | 0.08961 | 13.294 | 1537.93 |
| PPO 仅保留 Fz | 0.11189 | 0.09394 | 16.086 | 1478.51 |

所有方式在这 6 个场景中都有 3 个达到完整质量门槛。完整 PPO 和 Fx-only 的回报差约 59.41，其中努力惩罚改善 31.60、高度奖励改善 25.95，速度奖励只增加 1.80。努力项是归一化动作的平方惩罚，不是电机能耗测量。动作掩码会改变后续观测，Fx 也可能跟着变化；这是移除 Fz 通道的总效应，不能归因于一个孤立的力脉冲。

## 速度错误与动作边界

开发场景 04 的零残差最后 1 s 误差均值为 −0.116931 m/s，标准差仅 0.013123 m/s，主要表现为欠速。场景 10 则为 +0.025710 和 0.102621 m/s，主要是波动。分型使用 `RMSE² = bias² + variance`，把均方误差中偏差占比 ≥80% 标记为偏差主导、≤20% 标记为波动主导。这个描述标签不改变原有质量门槛。

[历史分型](historical_failure_types.csv)也复算了上一轮 240 回合。旧 holdout 已经公开，只作为历史材料，不参与这次场景或模型选择。

完整 PPO 在本次名义场景中的 raw Gaussian 竖直均值分别为 1.2145、1.1594、2.7876；标准差为 0.88465、0.86664、0.90983。执行时的上边界比例为 97.17%、78.88%、100%。Gaussian 均值越过 1 后，确定性执行会把它裁到 1；这说明了执行边界如何出现，还没有解释为什么训练走到了这里。环境裁剪 Gaussian 动作本身是合法设计，不能据此判定 PPO 实现有错。

CSV 的 `raw_upper_probability_z` 是当前观测下随机 Gaussian 样本超过 1 的理论概率。本次执行的是确定性策略，没有额外抽样，所以该列不代表训练时实际发生的裁剪率。

## 反馈是否有用

在每个种子的 8 个扰动回合中，实时 PPO 相对动作回放的末段速度 RMSE 差为 +0.002361、−0.001703、−0.000090 m/s，均未达到预先设定的“下降至少 10%，且至少 0.005 m/s”门槛。质量通过数的差却为 +2、0、+1。结果没有支持稳定的末段速度收益，也不能说 PPO 完全不使用状态反馈。

这里只测试两种起伏场景和小扰动。末段指标从 3.01 s 开始统计，推力在 2.1 s 结束；短暂恢复过程应另看完整轨迹和全回合 RMSE。不要把这个有限对照推广成“强化学习不需要反馈”。

## 复跑与审计

在仓库根目录运行：

```bash
rtk env PYTHONPATH=src:.local-deps python3 scripts/diagnose_d1_residual_feedback.py --mode reproduce --seeds 0
rtk env PYTHONPATH=src:.local-deps python3 scripts/diagnose_d1_residual_feedback.py --output results/d1_residual_feedback_repeat
rtk env PYTHONPATH=src:.local-deps python3 -m pytest tests/test_d1_residual_feedback.py -q
```

第一条是原症状红灯，运行 2 s。当前旧策略输出 `vertical_upper_fraction=0.94`、`raw_mu_z_mean=1.139023`，按设计以 exit 1 结束。第二条要求新目录，不覆盖旧结果。运行需要本地 PPO/MuJoCo 依赖及 [原 checkpoint](../d1_terrain_tracking_v2/training)。

[独立审计](independent_audit.json)核对了 153 个原产物及 247 个输入 SHA，复算全部 59200 个控制步，指标最大差为 1.11e-16。[summary.json](summary.json)保留每种子配对结果，原始逐步数据位于 [episodes](episodes)。

审计发现了一个汇总列错误：当次脚本按 `reward_` 前缀找奖励，额外把姿态参考列汇成了 `return_roll_target_rad`、`return_pitch_target_rad`。这两列不是奖励，不能使用。逐步奖励总和与所有物理指标均正确，配对结论未受影响。

[metrics_corrected.csv](metrics_corrected.csv)仅删除这两个误命名列；[原 metrics.csv](metrics.csv)和 [原 manifest](manifest.json)保持原字节。当次入口保存在 [源码快照](source_snapshot/diagnose_d1_residual_feedback.py)，其 SHA 与 `source_hashes.json` 一致。当前入口已改为奖励字段白名单，并用测试防止重复统计姿态参考。更正文件和本说明均为运行后补充，单独列入 [派生产物清单](derived_manifest.json)。
