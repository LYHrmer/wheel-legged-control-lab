# 把训练步数放到横轴上

[上一轮动作结构对照](shared_action_learning_lab.md)只测了每模型 32768 步结束时的结果。
三个训练种子没有给出稳定优于零残差的表现。仅凭这个点，无法判断继续训练是否会改善跟踪。

这一轮先保持奖励与控制器不变，每个模型连续训练到 262144 步，在四个预定位置存档。
本文记录实验条件和检查方法；正式学习曲线尚未完成，不能预先说增加训练预算有效。

2026-09-11 的实现验收为 2900 项测试通过、Ruff 通过。它覆盖保存时机和分析器的错误拒绝
等行为，不提供正式运动性能成绩。

## 一条轨迹，四个存档

每次 `learn()` 使用四个环境，每个环境采集 128 步后调用一次 `train()`。预算按环境
转移计数，一轮采样增加 `4×128=512`，与运行了多少秒墙钟时间分开记录。

| 累计转移预算 | 已完成的 `train()` 调用数 | 对应审计记录的零基索引 |
|---:|---:|---:|
| 32768 | 64 | 63 |
| 65536 | 128 | 127 |
| 131072 | 256 | 255 |
| 262144 | 512 | 511 |

每次 `train()` 内还会按 minibatch 更新参数，`n_epochs=4`。SB3 的 `_n_updates` 记录
累计优化 epoch，不能直接当成上表的调用数，也不能拿它索引 `updates.jsonl`。

`shared2` 和 `independent8` 各使用三个训练种子 `31000/32000/33000`。合计六条学习
轨迹，共采集 1572864 次环境转移，执行 3072 次 `train()`。24 个 checkpoint 中，同一条
轨迹上的四个参数状态彼此相关；分析时仍然只有每模式三个训练重复。把四档存档当成
十二个独立样本，会夸大每模式的数据量。

代码只调用一次 `learn(total_timesteps=262144)`。如果把四档预算分别从头训练，得到的
就变成四次训练实验；如果反复调用 `learn()`，还需要额外处理计步重置和学习率进度。
这里都不这样做。存档不得影响训练环境的状态或随机数流，评测回合留到全部训练完成后运行。

## 存档必须在参数更新之后

一次 rollout 结束时，数据已经采满，但这批数据对应的 PPO 更新还没有执行。若在
`on_rollout_end` 中保存“512 步模型”，保存到的可能仍是更新前的参数。

实际顺序要求为：

```text
采集 512 次转移 → 完整 train() 与审计 → 到达预算则存档 → 下一轮采样
```

第一个正式存档必须对应 `updates/updates.jsonl` 第 64 条记录里的 `param_sha256_after`，
该行的 `audit_index` 和存档的 `after_update_index` 都应为 `63`。
ZIP 文件的 SHA-256 检查文件字节是否一致，参数哈希检查排序后的张量内容是否一致；
它们计算的对象不同，不能直接互相比数值。

以 `shared2_seed31000` 为例，32768 步的文件位于
`shared2_seed31000/checkpoints/budget32768/checkpoint.zip`，同目录有 `checkpoint.json`。
训练目录中的 `checkpoints/manifest.json` 记录各预算的哈希和索引，
`checkpoints/phases.jsonl` 记录每次更新与保存的时间。根目录的最终 `checkpoint.zip`
保留旧入口兼容性，预算曲线按各 `budget{B}` 目录取模型。

全部训练结束后再逐个加载自己生成的 ZIP，核对重新加载的参数哈希，并记录到
`checkpoint_reload.json`。加载操作放在训练结束后，是为了避免模型构造消耗随机数而
影响后续采样。每份 `checkpoint.json` 仍使用主线的兼容性校验，动作映射和控制器参数
必须与评测环境一致。

保存逻辑见 [budget_checkpoints.py](../src/wheel_legged_control/d1/budget_checkpoints.py)，
它由[实验入口](../scripts/run_d1_locomotion_experiment.py)里的 `BudgetAuditedPPO.train()`
调用。整个研究的执行顺序由[预算调度器](../scripts/run_d1_budget_study.py)控制，
[分析器](../scripts/analyze_d1_budget_study.py)另行检查磁盘记录。

[存档测试](../tests/test_d1_budget_checkpoints.py)中有一个可逐步检查的对照：同种子训练
两个 rollout，一次开启存档，一次不开启。它比较每次更新前后的参数哈希，并逐项比较
审计 NPZ 中的数组。这个小实验检查存档是否扰动训练，不用于评价机器人控制能力。

## 固定的控制条件

两种模式继续使用 82 维观测、`imu_encoder_fusion` 状态源和八通道物理执行。
共享动作的广播方法见[动作结构练习](shared_action_learning_lab.md)。低层五项参数为
`wheel_kp=0.55`、`wheel_ki=1.5`、`yaw_feedback_gain=4`、`leg_feedback_scale=1`、
`attitude_feedback_scale=0.25`。没有测量延迟，也没有域随机化。

PPO 学习率为 `3e-4`，每个 minibatch 128 样本，四个优化 epoch；折扣
`γ=exp(-0.01/2)`，GAE 的 `λ=0.95`，裁剪范围 `0.2`，熵系数为零。策略与价值网络隐层
均为 `[64,64]`，初始 `log_std=-2`。这些数值与前一轮保持一致，不因预算增长而再调奖励。

新地形套件叫 `budget_compare_v1`。训练还是原来的四条道路；开发集沿用上一轮两条道路
与 `1017/1029` 重置种子。开发结果已经被看过，不再称为未见测试集。

最终评测使用预先固定的新参数实例，重置种子为 `4617/4629`：

| 布局 | 纵坡/横坡（°） | 波纹幅值/台阶高度（m） | x/y 波长（m） | x/y 相位（rad） |
|---|---|---|---|---|
| `s_bend` | `1.20 / 0.55` | `0.0080 / 0.0080` | `0.74 / 1.04` | `1.70 / 1.90` |
| `diagonal` | `-1.20 / -0.55` | `0.0075 / 0.0075` | `0.84 / 1.14` | `2.30 / 2.50` |

布局家族和命令脚本没有新增。这只能检验有限的新参数实例，不能据此宣称任意地形泛化。
道路配置位于 [locomotion_terrain.py](../src/wheel_legged_control/d1/locomotion_terrain.py)，
旧的 `v1` 与 `action_compare_v1` 数值和种子保持不变。

## 先跑短流程

在已安装项目依赖的环境里，从仓库根目录运行：

```bash
python scripts/run_d1_budget_study.py --smoke --dry-run \
  --output results/my_budget_smoke

python scripts/run_d1_budget_study.py --smoke \
  --output results/my_budget_smoke

python scripts/analyze_d1_budget_study.py results/my_budget_smoke \
  --output results/my_budget_smoke_analysis

python scripts/plot_d1_budget_study.py \
  --analysis results/my_budget_smoke_analysis/analysis.json \
  --output results/my_budget_smoke_plots
```

`--dry-run` 只打印协议，不创建目录或加载模型。实际输出目录必须不存在，失败的目录也
不能直接覆盖。Smoke 用两条 1024 步轨迹，在 512/1024 步存档，回合上限只有 0.2 s。
十二条命令、四十个评测案例用于检查接口；机器人还没充分进入道路任务，不能拿通过数
作为运动性能成绩。

本轮[最终短流程记录](../results/d1_budget_smoke_final/README.md)已完成：两条轨迹共 2048 次
转移，四个存档均实际重载，并与各自更新后的参数哈希相符。两份独立分析 JSON 的字节一致。
40/40 回合达到 0.2 s 时间上限，质量达标数为 0，非平地曝光比例也全部为零。这只验收
存档、重载和统计流程；正式的 60 s 地形任务仍需单独报告。

绘图读取独立分析器的 `analysis.json`，不启动仿真或加载策略。输出目录须新建，且不能放在
分析目录内部；这里把分析和图像放成同级目录。图像目录含 `development.png` 与 `holdout.png`，
计数另存 `counts.csv`，`manifest.json` 记录输入分析文件和图像的 SHA-256。
实现见[绘图脚本](../scripts/plot_d1_budget_study.py)。

正式实验去掉 `--smoke`，换一个新的输出目录：

```bash
python scripts/run_d1_budget_study.py --dry-run --output results/my_budget_study
python scripts/run_d1_budget_study.py --output results/my_budget_study

python scripts/analyze_d1_budget_study.py results/my_budget_study \
  --output results/my_budget_study_analysis

python scripts/plot_d1_budget_study.py \
  --analysis results/my_budget_study_analysis/analysis.json \
  --output results/my_budget_study_plots
```

这些命令是复跑方法，不能当成实验已完成的记录。调度器先测开发集零残差，然后完成六次
训练，测完全部 24 个开发集存档后，才测最终评测集的零残差和 24 个策略存档。合计
56 条命令、200 个案例。所有预算都测，不根据开发结果挑一个“最好模型”替代其余结果。

## 怎样阅读曲线

横轴是累计环境转移数，按 2 的对数刻度排列；相邻四档代表预算依次翻倍。蓝色表示
`shared2`，橙色表示 `independent8`，线型区分训练种子。先沿同一种子的线看预算变化，
再看其余种子是否同向。每个 split 单独出图，不把开发和留出回合接成一条线。

一个点包含该模式、训练种子和预算下的四个评测回合，即两条道路各用两个评测重置种子；
测量噪声种子由记录中的规则派生。
误差纵轴取四个回合 RMSE 的算术平均，不能理解成把四条时间序列拼接后重算的 RMSE。
绘图保留每个种子的曲线，不画置信区间，也不挑最好种子或最好存档。

### 缺口保留了什么

绘图要求一个点的四个回合全部完成，才画它的六项指标。只要有一个失败，该点的所有
绘图值就置为 `NaN`，折线在此断开。不能把缺口补成零、用前后预算插值，或改画其余三个
完成回合的均值。那些回合经历的任务长度和失败回合不同，删掉失败回合容易让结果显得更好。

分析文件仍保留 `completed_only_means` 和每例终止原因，绘图没有删原始记录。
若一档只有三例完成，读者可以追查这三例的均值，但它不会成为完整四例曲线上的点。
四例都完成但全未达质量门槛时，曲线仍会显示；是否达标要看 `counts.csv`。

| `counts.csv` 字段 | 含义 |
|---|---|
| `completed / cases` | 完成率；正式每点分母为 4，完成要求终止原因为 `time_limit` |
| `quality_passes / cases` | 质量达标率；先完成回合，再同时满足四项误差门槛 |
| `failed` | 未完成数；需回到分析文件检查越界、摔倒等具体原因 |

这里有时口头把完成率叫“存活率”，但仅仅没摔倒仍不够；越过地图边界也算失败。质量门槛
为速度 RMSE ≤ 0.08 m/s、偏航角速度 RMSE ≤ 0.08 rad/s、高度 RMSE ≤ 0.025 m，
以及姿态 RMSE ≤ 0.15 rad。最后一项按 `sqrt(mean(roll_error² + pitch_error²))` 计算。
质量率的分母仍是全部四例，不能改成“已完成例数”来隐藏失败。

零残差是同一轮腿低层的共同参照，没有训练预算。它仍执行 PI/反馈控制，并非关掉电机。
灰色虚线只有在本 split 的四个零残差回合全部完成时才绘制；否则省略，并在图上标明
参照缺失。每个 split 只测一份零残差，不能沿预算复制后当成额外训练样本。

仅当策略与零残差都完成相同长度的任务时，计算配对 RMSE 差值。提前摔倒的两秒误差
不能与跑完六十秒的误差直接排名。`policy_minus_zero` 中“策略减零残差”的速度差为负，
只表示这对完整回合的速度误差较低；还需要检查同点的姿态代价。

### 功率曲线的单位

机械功率记录的是执行器实际力矩与关节速度乘积绝对值之和，再对时间取平均，单位为 W。
物理子步上对应 `sum(abs(applied_nm * joint_velocity_rps))`。它用于描述机械活动量；模型
未计入驱动器损耗、电池或再生制动效率，不能把这条曲线解释为耗电或续航。

第六张子图的 `action_rms` 来自归一化的八通道物理动作，数值没有能量单位。共享两维策略
先广播到八通道，再与独立八维策略按同一口径计算；它也不是 PPO 原始高斯动作的 RMS。
地形曝光标签用于检查是否到过起伏区域，本身不等于通过能力认证。

若不同预算的三个种子走向不一致，报告这种分歧。若奖励上升而跟踪质量下降，应回到
分项奖励和轨迹找原因，但不要修改本次冻结实验的奖励后混写成同一条曲线。下一轮
奖励对照需要新的实验编号和最终评测实例。

独立分析器从 CSV/NPZ 复算误差，并检查参数哈希链和保存时机。它不导入仿真器，也不
反序列化 PyTorch 模型；实际加载验证来自单独的运行记录。哈希一致证明字节相符，不能
证明传感器或动力学真实。这一轮仍没有真机证据。
