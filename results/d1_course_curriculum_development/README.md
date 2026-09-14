# ATEC 启发的 D1 基础地形课程开发

已将课程调度接入真实 Gym/PPO：三条件各完成 16,384 个环境转移，共 **49,152 步**，保存最终模型并通过合法 sidecar 重载。此前三组 smoke 共 768 步、四个零残差探针共 4,800 步单列记录，不计入这次训练预算。本结果不证明课程性能优势、跳跃或大台阶能力。

方案由 gpt-6-astra / ultra 制定；Claude Opus 编写课程环境主体；本地修正、集成与实际执行的分工见 [contribution.json](contribution.json) 和 [implementation_review.md](implementation_review.md)。课程对应与后续任务见[实施说明](../../docs/atec_rl_curriculum.md)。

## 固定设置与实际曝光

单个训练种子 47000，单环境 CPU；同一 `wheel_leg` 基座、82 维 oracle 观测、独立 8 维残差、奖励、网络与物理限制。每回合最多 32 s，站立 0.5 s，随后用 0.5 s 加速到 0.25 m/s，偏航目标为零。地形从平地到训练幅值的 1/3、2/3、1，在回合边界切换；地形布局/波长/相位来自已有训练集。

| 训练组 | 难度 0/1/2/3 的实际转移 | 几何非平地转移 | 完整回合 / 预算截断回合 |
| --- | --- | ---: | --- |
| 平地 | 16384 / 0 / 0 / 0 | 0 | 5 / 1 |
| 随机混合 | 384 / 0 / 9600 / 6400 | 11781 | 5 / 1 |
| 固定阶段课程 | 6400 / 3200 / 3200 / 3584 | 6833 | 5 / 1 |

每组最后 384 步记录为 `budget_cut`，不伪造回合完成。五个完整回合均以 `time_limit` 结束，没有将超时存活写成越障成功。三组前六回合的地形索引与环境种子配对一致；随机混合没有抽到难度 1，实际曝光也不同，因此本次不构成严格等曝光的课程顺序对照。

每组完成 128 次 rollout 训练调用、512 个优化 epoch。`summary.json` 的 `ppo_updates` 是 SB3 `_n_updates` 的优化 epoch 计数，不能当成 512 次 `train()` 调用。CSV 在更新前记录统计，其最后显示 508；模型保存的是最后更新后的 512。

训练累计用时约 198.00 s，为三组各自计时之和，包含各组训练、保存与重载检查；不包含示教、GUI 或正式留出评测。训练中的回合进度后期变小，不能把已经产生梯度更新写成性能提高。

## 固定最终策略的同条件检查

另用环境 seed 55101、三级训练地形 index 0、相同初始状态和上述相同命令，独立比较零残差与三组最终模型。全部确定性执行，每组 32 s、3200 步，首次终止即结束；没有挑 checkpoint 或重新训练。这是单一训练地形上的开发比较，不是 holdout。

| 最终策略 | 净前进 m | 速度 RMSE m/s | yaw RMSE rad/s | 高度 RMSE mm | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| 零残差 | 7.520 | 0.03972 | 0.01414 | 11.05 | 61.053 |
| 平地 PPO | 4.901 | 0.08979 | 0.02424 | 7.73 | 56.886 |
| 混合 PPO | 5.620 | 0.07202 | 0.02065 | 8.57 | 58.785 |
| 课程 PPO | 4.834 | 0.09054 | 0.02059 | 8.37 | 56.814 |

三种 PPO 的高度误差更小，速度、yaw、净前进和回报均弱于零残差。本次没有证明 RL 或课程改善了综合表现。小预算、单种子和不同实际曝光也不足以否定这些方法的一般价值，更不能据此确定唯一退化原因。

四例均正常到时结束，没有机身触地、输入力矩裁剪或 actor 动作边界饱和。独立逐拍复算确认初始状态、观测、元数据及全部 3200 条命令一致，奖励分项、RMSE、位移、曝光和时序均匹配。完整值见[最终比较](final_policy_check_01/report.json)，复算见[审核报告](final_policy_check_01.audit.json)。最初辅助脚本在创建环境前发生一次导入错误，0 物理步，另保留失败记录。

## Smoke、探针和独立核对

- Smoke 每组 256 步、8 个 0.32 s 回合。课程组实际经历四级，各 64 步。所有 smoke 回合都在加速前结束，几何非平地暴露为零。
- 零残差探针四档各 12 s，净前进 2.833–2.864 m，全部正常到时结束。三档非零地形各有 589/1200 步几何暴露，终点 x≈−0.94 m；尚未到 `straight` 起伏段 x=0.5 或台阶段 x=3.0，只是前段坡面探针。
- 六个模型均保存并通过 SHA、兼容契约校验、确定性动作重载一致性及实际物理执行。独立审核另重新加载六个模型，逐个执行有限状态的一拍。
- 36 项课程测试加 20 项键盘/课程回归共 56 项通过；全仓 `src tests examples scripts` Ruff 通过。真实 MuJoCo 中，wrapper 与同配置直接环境逐拍观测、奖励、终止和暴露相同。
- [独立审核](evidence_review.json)复算曝光/步数、四个探针速度 RMSE，检查实验来源与全部 77 个正式冻结文件。训练未保留独立逐拍状态重放文件，训练曝光审核依据回合记录；不把它称为逐拍训练轨迹复算。

Smoke/probe 使用的 [runner 快照](runner_smoke_probe.py)与开发运行入口只差机器人资产哈希覆盖、命令元数据和优化 epoch 说明，训练逻辑相同；全部对应源哈希已核对。

## 复现和文件范围

从仓库根目录执行，输出必须为新目录：

```bash
python scripts/run_d1_course_curriculum.py --mode smoke --output runs/course_smoke_02
python scripts/run_d1_course_curriculum.py --mode probe --output runs/course_probe_02
python scripts/run_d1_course_curriculum.py --mode train --steps 16384 \
  --episode-seconds 32 --seed 47000 --output runs/course_development_02
python results/d1_course_curriculum_development/check_final_policies.py \
  --repo "$PWD" --models runs/course_development_02 --output runs/course_final_check_02
```

本目录保存协议、回合记录、训练 CSV、模型 sidecar、摘要和审核。六个模型 ZIP、四个探针完整逐拍 JSON 和最终比较的逐拍状态留在本机工作目录，未上传到本目录或 v0.8；大小与哈希见 [local_artifacts.json](local_artifacts.json)，可按以上命令重新生成。该批是独立开发实验，没有改写 v0.8 正式结果，也没有把模型接入独立的课程试驾窗口。
