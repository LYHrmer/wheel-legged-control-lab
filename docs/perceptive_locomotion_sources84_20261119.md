# 轮足地形预瞄、世界模型与控制验收

2026-11-19。D1 的第一项任务定为直行入坡。依次核对接触前能否测到地形、看到之后还剩多少控制余量，以及模型给出的改进方向能否在物理仿真中复现。前两项用现有 LQR/VMC 和八束稀疏测距即可检验。只有它们通过，才值得让世界模型选择俯仰修正增益。[第 84 轮锁定协议](experiments84/terrain_preview84.json)

## 八个作者项目及可核对的代码

第 84 轮结束后，下一轮把强化学习设为主方法，在同一感知与动作接口上比较 PPO 与 SuperMimic 式更新。实际测得的遮挡和执行余量决定两项训练前检查。[结果与影片](terrain_preview84_results_20261119.md) · [新的四步强化学习方案](terrain_preview84_rl_followup_20261119.md)

表中日期为所查源码的提交日期。论文、仓库、训练权重和硬件执行程序分别核对。论文报告的实机结果均归原作者。

| 平台与固定版本 | 源码实现与发布范围 | 给 D1 的启示 |
|---|---|---|
| **LocoWM** · Go2-W 轮足 · `c2a772f` · 2026-10-03 | 有两阶段训练、评测和实机搬运录像。根据本体历史与基线拟执行动作，一次预测未来多个时刻的角速度、线加速度、重力方向。第二阶段冻结基线和模型，把预测序列交给 PPO 残差网络。仓内还有将模型特征置零的基线。[论文](https://arxiv.org/abs/2609.39179) · [模型目标](https://github.com/zhaozijie2022/locowm/blob/c2a772f83ee02b61200261af79e458ee68633f84/loco_rl/modules/world_model.py#L1-L5) · [冻结与残差](https://github.com/zhaozijie2022/locowm/blob/c2a772f83ee02b61200261af79e458ee68633f84/loco_rl/modules/adapter_actor_critic.py#L138-L168) · [零特征对照](https://github.com/zhaozijie2022/locowm/blob/c2a772f83ee02b61200261af79e458ee68633f84/scripts/run_stage2_nowm.sh) | 保留能独立运行的经典控制器，让新分支只负责入坡瞬态。预测依赖当前动作，这给“同一起点改变一个修正量”提供了实验依据。LocoWM 的预测特征在 `no_grad` 中计算，训练机制与下方 SuperMimic 有别。 |
| **FLORES** · 16 动作轮足 · `814db43` · 2026-05-22 | 作者公开 ROS 2 控制器、MuJoCo 接口、HIM 策略权重和适配 HIMLoco 的两个训练环境文件。单帧策略输入 57 项，共 12 帧。启用地形采样后，187 个高度点进入特权观测，策略收到的切片截止于第 57 项。[论文](https://arxiv.org/abs/2507.22345) · [部署和训练依赖](https://github.com/ZhichengSong6/FLORES/blob/814db438b70793f1115c491553d7cb39eaf6b74d/code/README.md#L86-L153) · [观测实际切片](https://github.com/ZhichengSong6/FLORES/blob/814db438b70793f1115c491553d7cb39eaf6b74d/code/HIM_FLORES/mdog_robot.py#L260-L292) | 查到执行动作的那一层。仿真配置里出现高度扫描，仍需验证部署策略是否读到它。 |
| **ETH SRU** · AoW-D、B2W · `6a25e72` · 2026-10-25 | 有 Isaac Lab 导航环境、深度编码器及低层策略权重。高层以 5 Hz 从含时延和噪声的深度等输入产生平面速度指令，低层以 50 Hz 转成关节动作。高层网络和学习算法在此环境仓以外。[论文](https://arxiv.org/abs/2506.05997) · [开放组件和控制频率](https://github.com/leggedrobotics/sru-navigation-sim/blob/6a25e724fafe972ff7b7442ee98f6b93be5bca9f/docs/README.md#L6-L47) · [深度、特权高程与低层观测](https://github.com/leggedrobotics/sru-navigation-sim/blob/6a25e724fafe972ff7b7442ee98f6b93be5bca9f/isaaclab_nav_task/navigation/navigation_env_cfg.py#L173-L231) | 可以异步使用地形测量。D1 的 100 Hz 内环按原频率运行，每包前视数据都携带原采集时刻。高层导航成功与低层入坡误差须分别记分。 |
| **JEPLO** · Go2 四足 · `460e227` · 2026-09-17 | 真实平台由 LiDAR 点云生成距离图。代码融合本体和距离图历史，再以动作历史预测下一步的两个潜在表示。作者公开训练、预训练策略、MuJoCo/硬件执行、实验数据集。部署依赖与传感布局有详细说明。[论文](https://arxiv.org/abs/2609.15770) · [动作与下一观测对齐](https://github.com/ASIG-X/JEPLO/blob/460e2272907af053d6e66b57fec71fdbe6ca1774/training/rsl_rl/rsl_rl/modules/jepa_estimator_sensor_ss.py#L315-L372) · [部署和数据入口](https://github.com/ASIG-X/JEPLO/blob/460e2272907af053d6e66b57fec71fdbe6ca1774/README.md) | 保留采集、数据到达、动作三个时间戳。第 84 轮只有每侧四束，先记录每个可见测点和有效掩码。潜在表示的预测误差与车身俯仰 RMSE 是两种不同验收量。 |
| **PROOF** · Go2 四足 · `ae910b8` · 2026-11-18 | 作者的后续实现提供教师训练和深度学生蒸馏。可导出的学生策略用本体与视觉联合估计不确定性，通过两个阈值维持门的开关状态。项目页分支 `afd6ada` 含 `actor.onnx` 和 `depth_encoder.onnx`，主仓清单把实机部署代码列为待发布。[论文](https://arxiv.org/abs/2611.18305) · [训练发布状态](https://github.com/syw-robotics/proof-loco/blob/ae910b821d635f04226489df80bdd77d1d7b0b82/README.md#L27-L39) · [门的输入与滞回](https://github.com/syw-robotics/proof-loco/blob/ae910b821d635f04226489df80bdd77d1d7b0b82/source/z_rl_proof/proof_rl/models/proof_gate_student_export.py#L98-L115) · [网页权重](https://github.com/syw-robotics/proof-loco/tree/afd6ada475cdf40c6ce784f7675bdebe02c1eae2/demo/src/policy) | 先用遮挡、过期和未来时间戳做故障工况，协议也将左右轮迹最大允许坡度差定为 3°。几何可信度由已观察到的区段决定。学习不确定性门需要另外的数据和对照。 |
| **SuperMimic** · G1 人形 · `698ea39` · 2026-11-19 | 公开采样、动力学训练、策略回传及导出代码。默认模型窗口 8 帧，监督 7 次转移。策略更新冻结动力学参数，在 32 拍想象中重新计算每一步动作，然后沿后续回报更新策略。两项现成任务从 G1 平地跟踪构建。[论文](https://arxiv.org/abs/2610.28480) · [模型窗口](https://github.com/HybridRobotics/supermimic/blob/698ea39bc901d8fe733b3c8621df3199d241f7e4/src/supermimic/learners/wm_learner.py#L145-L167) · [策略梯度](https://github.com/HybridRobotics/supermimic/blob/698ea39bc901d8fe733b3c8621df3199d241f7e4/src/supermimic/learners/policy_learner.py#L83-L125) | 复用短时闭环求导。把窗口长度换算成秒，使用 wheel 实际送入 LQR/VMC 的受限修正和自身的 10 ms 周期。 |
| **Terrain-Grounded Imagination（TGI）** · G1 人形 · `a64be1f` · 2026-11-11 | 作者仓包含方法说明、训练、导出、MuJoCo 评测和梯度检查。它在预测位置查询通过仿真预先获得的全局高度图，梯度可以穿过双线性空间查询。仓库引用的 *Watch Your Step* 提供落脚点感知方法与评测场景。作者将 TGI 本身列为独立的软件工作。[作者说明及引用](https://github.com/leggedrobotics/terrain-grounded-imagination/blob/a64be1f185110d21c680ca1baf66e436772801bc/README.md#L186-L228) · [空间查询](https://github.com/leggedrobotics/terrain-grounded-imagination/blob/a64be1f185110d21c680ca1baf66e436772801bc/tgi/src/tgi/mdp/heightmap.py#L11-L47) · [有限差分检查](https://github.com/leggedrobotics/terrain-grounded-imagination/blob/a64be1f185110d21c680ca1baf66e436772801bc/tgi/scripts/grad_check.py#L1-L14) | 把仿真全图限制在造传感器和评分一侧。策略想象时只查询起点已经测到的左右轮迹，查询位置随预测位姿移动。TGI 的差分比较使用模型内目标，D1 还要另做 MuJoCo 物理差分。 |
| **Open-DiffLoco** · Go2 四足 · `35256b6` · 2026-08-05 | 作者公开使用 MJX 可微仿真的 SHAC/JAVE 训练、策略导出及 C++ 部署入口，论文展示仅依赖本体观测的实机速度跟踪。这个梯度来自可微模拟器。[论文](https://arxiv.org/abs/2608.02069) · [训练和部署入口](https://github.com/MartinOpat/open-diffloco/blob/35256b6c8e243d658d6379ff8d17cde8e01d8211/README.md#L14-L89) | 仅用本体测量的控制仍是有实机证据的基线。SuperMimic 回答的是动作跟踪的学习效率。D1 是否需要前视要按具体任务检验，随机出现且尚未触及的坡脚正好能检验这项增量。 |

## 怎样把这些方法接到一条控制链上

前轮前方左右各设一条测距带。控制器根据同一个数据包，在预测的前轴位置两侧寻找实测端点，并核对两条轮迹斜率。通过后才把预计坡度与本体估计的差送入俯仰参考。单包更新率 6 Hz，模拟传输延迟 40 ms。控制器等待新包时继续推进里程计，同时累加原包龄。超过 250 ms 或查询位置越出有效区段，附加参考目标回零。采样有效并不保证有修正余量，必须记录限幅前后俯仰参考、基线作用力及实际关节力矩。[传感和余量约定](experiments84/terrain_preview84.json) · [力矩与纵向力控制器](../src/wheel_legged_control/d1/hierarchical.py)

学习阶段拟预测相同基线控制器及车体在受限修正后的响应。在每个想象步，从预测位姿计算前轴在**起点实测地形**中的位置，只在有效相邻点内插值。使用同一批估计器历史、限幅器状态、指令与动作时序。未来实际测到的帧用于监督和复算。真值地图的访问权限到传感器生成器和离线评分为止。闭环轨迹里的每一帧都不能当成一个独立重复。[上游逐拍重建观测](https://github.com/HybridRobotics/supermimic/blob/698ea39bc901d8fe733b3c8621df3199d241f7e4/src/supermimic/envs/world_model_env.py#L84-L130) · [本仓传感器接口](sensor_estimation.md)

**只切断地形查询的梯度。** 同一冻结模型取两条计算路径。一条用预测位置查询已测地形，另一条对查询坐标做 `detach()` 后用完全相同的点、权重和有效掩码查询。先验证前向逐步的地形值、动作、状态及代价逐项相同，再比较两条路径给出的俯仰增益梯度。后者仍可通过动作与车身动力学回传。用相同模型、相同随机量的中心差分检验完整求导实现，并另存真实 MuJoCo 的正负微扰结果。平地或等坡区中，同段两点算出的斜率是常数。只有已观测几何确实随位置变化时，这个消融才有辨识力。两种反向结果接近时，应报告未分辨出空间梯度的贡献。[TGI 空间查询与梯度核查](https://github.com/leggedrobotics/terrain-grounded-imagination/blob/a64be1f185110d21c680ca1baf66e436772801bc/tgi/scripts/grad_check.py#L1-L14)

## 四步验收

第 1、2 步按已锁定的机器协议运行。第 3、4 步是有条件的后续方案，需另行批准。预计时长含实现与核验。

| 假设 | 唯一变量 | 固定预算与预计时长 | 判定规则 |
|---|---|---|---|
| **1. 接触前能分辨坡向。** | 在同一本体姿态下改变可见前方地形 | 平地、±4° × 三个姿态，共 9 个静态工况，另加遮挡、包过期、未来时间戳 3 个故障和视野外不变性检查。零训练，0.5–1.5 小时 | 9 个误差各 ≤1°，所有坡向正确，3 类故障均能关闭预瞄，改变未见几何时输出相同。 |
| **2. 接触前的修正能改善入坡。** | 预瞄俯仰修正乘 0、+1 或 −1 | 3 种坡度 × 3 个噪声种子 × 3 种输入，共 27 段，每段最多 6 秒。16,200 控制步，含一次冷辨识共最多 83,555 原生物理步。零训练，2–4 小时 | 六个有效预瞄坡道都需在触坡前连续 ≥3 拍同向有效修正，通过坡脚后 0.4 m，区间俯仰误差均下降且配对降幅中位数 ≥10%。上、下坡各自优于反号组的中位误差。每对含平地的速度 RMSE、关节力矩 RMS 恶化均 ≤5%，无新增跌倒或力矩越限。影片固定种子 8417。 |
| **3. 已测地形能提高对干预效果的预测。** | 两个同结构模型分别接收有效地形或屏蔽地形 | 3 份配对初始化/组，每份 ≤100 次模型更新，总数 ≤600。除第 2 步开发数据外，开发和盲测各 6 个起点，按整段布局、坡脚位置及噪声种子隔离。每点固定负、零、正三个 64 拍分支，共 2,304 分支步，起点准备 ≤4,800 步。3–5 小时 | 在碰触盲测前封存预测。每个盲点核对真实限幅后正负输入可分、物理俯仰 RMSE 差绝对值 ≥0.01°。3 个有预瞄模型均须在 6 点判对符号，各自的差值预测绝对误差中位数优于无预瞄配对模型。另做上文前向相同的梯度检查，其指标单列。 |
| **4. 学到的一个增益能改善完整控制。** | 解析预瞄律中的一个标量增益 | 以同一解析控制器为起点，每份合格模型各做一次 64 拍回传。3 个候选和解析基线 × 3 个新噪声种子 × 45/50 秒，共 24 回合、最多 114,000 控制步。3–5 小时 | 三个候选各自均须 6/6 完整运行且通过原有逐阶段质量与安全门槛，同条件入坡误差配对改善。报告每个种子的绝对误差、差值与极差。独立短筛、模型内梯度通过和完整控制通过分别给出判定。 |

84 的 CPU 单环境、单线程、内存门槛和不续跑约束在机器协议中固定。本体估计、测距、控制参考或执行链任何一处失败，都可先按已存时间戳复核“首次看见→首次有效修正→前轮触坡”，再决定是否登记下一次单变量试验。D1 第 81 轮的模型梯度候选就曾出现独立短筛 3/3 通过、完整控制 0/3 通过，足以说明完整路段验收必须独立保留。[资源、计数与评分](experiments84/terrain_preview84.json) · [第 81 轮完整结果](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/8bf315f1a04b284c6c97855fa32f830894706a1b/docs/objective81_results_20261103.md)

## 检索范围与局限

本次用 AI 辅助检索并逐项查阅作者论文、项目页及 GitHub 固定提交。检索词覆盖 *wheeled quadruped terrain perception*、*Go2-W world model residual*、*LiDAR quadruped predictive learning*、*perceptive fallback*、*terrain-grounded imagination* 和 *blind differentiable locomotion*，截止日为 2026-11-19。表中收录可定位到训练、部署、观测构造或梯度实现的作者版本。举例说，Go2-W 长距离自主导航论文在代码可用性一节提供定位模块，因此没有将其整套运控计为公开实现。[作者的代码可用性说明](https://arxiv.org/html/2606.21387v1)

各项目的机器人、传感器、速度、失败定义、训练资源和部署开放程度不同。此次核对源码与发布件，没有在本机重新训练外部项目。第 84 轮的首返距离采用理想仿真值，本体测量带噪声和延迟。实车测距误差、反光、积尘及真实端到端时延仍需单独采证，跨论文的得分也不能代替这台车的配对试验。
