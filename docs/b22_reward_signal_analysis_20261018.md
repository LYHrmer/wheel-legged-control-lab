# B22：已有轨迹中的奖励敏感性（2026-10-18）

已有数据支持一个有限判断：classical zero在两种平地任务中的vx/yaw tracking奖励已经接近上限，原评分要求的相对SSE改善只对应很小的绝对奖励差。它还不能解释训练为何失败。实际gain0.5 B在五组任务的drive平均奖励均下降，五组全回合合计奖励也下降；rough单例全回合奖励略升，需单独保留，不能概括为所有任务均下降或已经证实奖励优化成功。

本记录只读旧35十个已完成回合的16000行控制记录、B22历史32条rollout汇总和保留的首1024行训练数组。没有训练、推理、模型构造或物理积分；35/37原null计数和资格保持原样。完整数字和SHA见[机器报告](../reports/b22_reward_signal_analysis_20261018.json)。

## 原公式与评分为何不能互换

原奖励由`full_drive_env_08.py::_reward`给出，`world_upright_course_11.py::_reward`仅将姿态项换成world roll/pitch。令`qx=((vx-consumed_vx)/0.25)^2`、`qy=((yaw_rate-consumed_yaw)/0.4)^2`，vx/yaw两项为`2 exp(-qx)+exp(-qy)`，上限3。另有上限1的lateral tracking，及height、attitude、实际torque、动作变化、腿速和termination项。所有正向tracking项总上限4。

原贡献评分在固定drive窗口累加**无权重**的`qx+qy`，并另设task/cost门。奖励对vx加权2、使用指数，覆盖全回合且含其他项。因此相对15%的SSE改善不能直接读成15%的总奖励提升。

下面把已有zero每拍的**平方归一化误差**分别代数替换为`0.85*qx`、`0.85*qy`，其余轨迹、指令和奖励项固定。它不是速度误差幅值减少15%，更不是新策略的实测结果；pooled SSE下降15%也不保证每拍每项均匀下降。这一反事实仅衡量既有误差附近的公式敏感性。

| 任务 | zero drive vx+yaw均值（上限3） | 反事实平均奖励增量 | 实际zero每拍总奖励总体标准差 | half B−zero drive奖励 | half B−zero全回合奖励 |
|---|---:|---:|---:|---:|---:|
| flat_0p6 | 2.995677 | 0.000645 | 0.009014 | -0.000193 | -0.000177 |
| flat_1p6 | 2.995683 | 0.000645 | 0.007598 | -0.000102 | -0.000067 |
| yaw_1p2 | 2.967627 | 0.004723 | 0.035834 | -0.001106 | -0.000804 |
| bumps_0p4 | 2.980945 | 0.002806 | 0.035119 | -0.000699 | -0.000423 |
| rough_0p35 | 2.934757 | 0.009254 | 0.117237 | -0.004782 | +0.000750 |

平地两项的反事实增量约0.000645/拍，只占总奖励上限4的0.016%；zero两项tracking同时位于各自上限1%以内的drive拍占91.9%与94.2%。yaw/bumps/rough的误差更大，对应增量也更大，不能把所有地形都称为同样饱和。表中标准差只是旧轨迹逐拍的描述量，含时序相关和瞬态，不是policy-gradient噪声、显著性检验或学习信噪比。

实际half B对height的drive奖励贡献均有约+0.00031～+0.00046改善，但tracking、lateral或attitude损失将它抵消，五组drive净奖励均为负。例如flat1.6的height为+0.000459，vx tracking为−0.000252、attitude为−0.000257，最终drive总奖励为−0.000102。

五组等长1600拍合计的平均奖励差约−0.000144/拍，但rough是例外：全回合+0.000750，drive−0.004782。其全回合vx、height、attitude分别+0.000685、+0.000734、+0.000164，lateral为−0.000937；drive外1130拍合计贡献更高，抵消了470拍drive损失。这个单例确实提示“全回合多项奖励”与“固定drive SSE”的时域/权重取舍，但不能证明训练优化了这一取舍或它导致RL贡献失败，更不能凭此直接宣称某个新reward会更好。

## 历史训练保存了什么

B22实际receipt为complete：本阶段32768 transitions、32×1024 rollouts、104个实际epochs、355次optimizer steps，22个train calls触发KL early stop。32条rollout的每拍平均回报总体均值为3.957183、跨rollout总体标准差0.024657，范围3.882453～3.982661。不同rollout对应不同地形、指令和更新后策略，这个变化不能当成相同任务32次独立重复的随机方差，更不能推断训练回报已单调改善。

首1024行数值归档可重读，奖励项之和与reward最大差8.89e-16，vx/yaw公式重算差0。其中首个flat0.4回合的825个实际servo正向拍，总奖励均值3.977804、逐拍标准差0.015918；同样的代数反事实增量为0.001613。该段height、attitude、action-change奖励的逐拍标准差分别约0.000278、0.000557、0.000147。它表明绝对tracking信号需与其他变化共同考虑，不能证明其他项已主导PPO更新。

本地manifest列32个numeric blocks、32个Gaussian blocks和164个full-control blocks，现各仅保留第一个：1024行数值、1024行Gaussian、200行完整控制。因此不能给全部32768行的逐项协方差、实际有效课程/接触覆盖或末期策略的信噪比。32个closed-episode标量事件和33份schedule仍在；配置与历史事件确有1.6m/s和bumps/rough/ramp，不能猜测“从未训练高速/地形”。训练episode的vx-only原始SSE也不是heldout的归一化vx+yaw SSE。

## spec22与父代来源已找到

实际归档原件为`results/d1_main_route_execution_20260930_22/spec22.json`，SHA **f061b64320e8fae35dd8ca0d186242579b4640bd5fecdd3e6da63d2f6983f8f4**。其bytes/SHA同时匹配最终metadata的`source_files`、`contract_documents`和训练session中原绝对source键。缺少的是重定位runtime目录旁的spec副本，不能说spec22原件丢失。spec自身仍是科学计划模板，其status不是实际执行回执；执行事实来自训练receipt和归档。

当前checkpoint SHA为`7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691`，与训练归档final checkpoint完全相同。实际parent model SHA `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`、parent metadata SHA `cce486dc3f0ae9f0db61babc196cb48b692644659523cab5a7aecbc9fcadeb6a`也与现存原路径文件一致；父代final policy/Adam hashes精确连接本轮initial hashes，actor与optimizer均未重初始化。

spec明确声明祖先训练81920 controls；父metadata记录65536的前代与其自身16384阶段，和这一声明一致。再加本轮32768，档案声明路径为114688。**本次未重新审计全部祖先原始计数，114688不作为新闭账的累计actual。** train22实际采用warm-start并`reset_num_timesteps=True`，故保存的32768绝非全部历史。当前combined controller也与既有评测实现相符，没有controller mismatch证据。

## 可证伪的下一步

先把“tracking的绝对信号较弱”保留为候选机制。后续若安排新的有界对照训练，应从同一个完整checkpoint+Adam起步，在相同课程、seed、预算和final-only规则下，将原reward与候选reward作等预算对照，完整留存逐项reward和更新统计；原六任务、task/cost、贡献门均照旧。若增加tracking敏感性后仍未改善真实配对SSE或损害成本，就否定该候选，而不是以训练reward变高代替成功。

命令时域也只能作为独立假设：历史驱动至episode reset的课程，与heldout的drive/制动/final分段可能提供不同反馈；但已有五组劣化包含drive窗口，不能仅用停车或课程差异解释。更换reward与更换课程应分开比较，不能混合后宣称因果。本分析未启动新的训练。
