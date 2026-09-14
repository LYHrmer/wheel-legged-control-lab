# 用 ATEC 课程改进 D1 强化学习任务

ATEC 最值得迁移的三部分是分地形残差训练、局部残差模仿学习，以及按实际反馈结束动作阶段。当前先把基础地形课程接入 D1 的 82 维观测、8 维动作训练链；跳跃和横移需要另建任务接口。

方案由 `gpt-6-astra`、`ultra` 推理档制定，Claude Opus 编写课程环境候选，Codex 负责修正、训练入口和实际验证。所读 ATEC 版本固定为 [`4f79d7f`](https://github.com/LYHrmer/atec-robotics-projects/tree/4f79d7f05507f45674e67abde06eddfe7c038a17)。

## 哪些课程能用

| 课程或实现 | 对 D1 的具体用途 | 适配边界 |
| --- | --- | --- |
| [Task A 学习路线](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/task_a/docs/LEARNING_GUIDE.md)与[分地形训练](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/task_a/tools/d1g2_taska_train_env.py) | 保留基础控制器，让 PPO 学有限修正；从较缓地形开始，分别记录坡道、起伏和台阶上的表现 | 上游使用 Isaac Lab、D1+G2 和不同动作接口，其权重不能直接加载 |
| [Task E 模仿学习](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/docs/TASK_E_IMITATION.md) | 从本机成功的短技能采集状态与动作，先行为克隆，再比较 PPO 微调；保留同底座零残差对照 | 机械臂的 45→6 权重不能控制轮足；本机卡住或跳不起来的演示不能当成功教师 |
| [运动学与动作语义](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/docs/ALGORITHM.md) | 先验证腿长、足端目标、关节位置和轮速之间的真实映射，再设计动作空间 | 命令到位不等于轮子离地、后轮通过障碍或落地稳定 |
| [反馈与省时复盘](https://github.com/LYHrmer/atec-robotics-projects/blob/4f79d7f05507f45674e67abde06eddfe7c038a17/docs/OPTIMIZATION.md) | 用连续稳定窗口决定阶段完成，减少已到位后的等待；统计侧步释放后的净位移和完整周期时间 | 单阶段变快可能把回退或等待转移到下一阶段，要看全过程 |

这些内容支持方法选择，不是当前 D1 的性能改善证据。详细的控制与权重差异见[先前参考核对](atec_reference.md)。

## 当前任务缺什么

当前 [v3 观测](../src/wheel_legged_control/d1/locomotion_observation.py)含本体状态、前进/偏航/高度命令、基础控制建议和记忆，没有前方地形、逐轮接触、跳跃目标或横移命令。[8 维动作](../src/wheel_legged_control/d1/wheel_leg_controller.py)是四腿伸长与四轮速度修正，不能直接指定足端横向摆动。

现有[奖励](../src/wheel_legged_control/d1/locomotion_rewards.py)面向速度、偏航和高度跟踪。稳定高度与低动作变化的目标不等同于下蹲、腾空和落地任务。现有[训练地形](../src/wheel_legged_control/d1/locomotion_terrain.py)也只含名义纵坡不超过 3°、横坡不超过 2°、起伏与带斜边台阶不超过 1 cm；GUI 的较大台阶、坡道和障碍条不属于这个训练范围。

`run_d1_course_drive.py` 当前使用独立的经典控制适配器，没有加载 PPO。改进这个训练入口不会自动改变课程窗口里的控制器；模型接入必须通过观测、动作、时序与增益校验。

## 已接入的基础地形课程

新增[课程环境](../scripts/d1_course_curriculum.py)和[运行入口](../scripts/run_d1_course_curriculum.py)，保持现有 `wheel_leg` 基座、82 维输入、`independent8` 动作、奖励、传感器和物理限制一致：

- `flat`：每回合平地。
- `mixed`：每回合随机选择四档难度。
- `curriculum`：按实际训练转移预算的四分位选择难度，只在下一次回合重置时生效。

难度为平地，以及原训练地形坡度、起伏、台阶幅值的 1/3、2/3、1。四种原训练实例的布局、波长和相位保持不变。每个回合内部碰撞地形固定。三组使用相同命令：站立 0.5 s，随后用 0.5 s 加速到 0.25 m/s，偏航速度为零，高度目标 0.455 m。

本轮沿用 v3 默认的 oracle 状态源，仅为电脑仿真实验。环境记录每回合的实际步数、种子、地形参数、终止原因、最终有符号前进位移，以及已有几何地形暴露。提前结束或训练预算截断单独标记。混合与课程的实际暴露可能不同，不能仅凭相同总步数把成绩差异归因于课程顺序。课程等级也不是通过率。

先运行短训练检查：

```bash
python scripts/run_d1_course_curriculum.py --mode smoke --output runs/course_smoke_01
python scripts/run_d1_course_curriculum.py --mode probe --output runs/course_probe_01
```

`smoke` 每条件 256 次真实环境转移，回合 0.32 s，检查 PPO 更新、保存、兼容性校验、重载动作一致性和一次重载后的物理执行。此时回合在加速前就结束，不能用它证明学会地形。`probe` 用零残差在四档地形各前进 12 s，记录实际曝光与失败，也不把局部行驶称为完整越障。

独立开发训练入口为：

```bash
python scripts/run_d1_course_curriculum.py --mode train --steps 16384 \
  --episode-seconds 32 --seed 47000 --output runs/course_development_01
```

所有输出目录必须是新目录。较长训练还需要新协议、至少三个训练种子及预定留出评测；上述入口本身不证明收益。正式 v0.8 的 77 个源文件、旧权重和实验结果保持原身份。

本轮已完成三条件共 49,152 步开发训练，六个 smoke/开发模型的保存重载与实际执行检查通过。课程组四级实际曝光为 6400/3200/3200/3584 步，混合组未抽到一级；这些结果说明课程被真实训练消费，不能据此判定课程比混合更好。具体协议、结果与限制见[开发记录](../results/d1_course_curriculum_development/README.md)。

随后固定最终模型做了一次同条件 32 s 开发比较：零残差净前进 7.520 m，平地/混合/课程 PPO 分别为 4.901/5.620/4.834 m。三个 PPO 高度误差更小，但速度、yaw 和回报更差。因此，本轮交付是可运行的课程训练与完整负结果，尚未实现综合性能改善。

## 后续技能怎么练

先分别建立前进提速、滚过小台阶、摆腿越障、原地跳跃和横移任务，验证所需动作在当前机构和力矩限制内可执行。

跳跃要输入目标净空、阶段和接触信息。主指标使用同一时刻四个轮子最小净空的峰值，另检查同时腾空时间和落地后的持续稳定。机身升高或控制相位显示 `flight` 都不能代替这个指标。起跳和落地事件只记一次奖励，避免通过反复抖腿或悬轮刷分。

横移需要能表示足端横向运动的动作接口，并测量松键、落脚和控制交接后的净位移。示教要包含完整交接，不能只截取移动最多的片段。通过这些检查的本机演示才能用于 BC；随后比较零残差、教师、BC、BC+PPO 的实际完整回合。

训练和留出按整回合、种子拆分。部署策略不能读取教师当步答案或伪装成传感器的仿真真值。若增加前视地形，必须注明来源；若只让 critic 使用训练真值，还需专门实现非对称 actor/critic 输入，现有 SB3 `MlpPolicy` 不会自动提供这个能力。
