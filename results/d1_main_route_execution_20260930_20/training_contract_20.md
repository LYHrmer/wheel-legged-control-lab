# C20 训练合同：共同 KL 护栏下的课程 A/B

本合同与 `spec20.json` 配套，只有 root 的来源 GO 后才能进入实际执行。Astra 编写期间没有模型、物理或测试调用。不得改写旧源、父模型或历史记录。

## 1. 起点、公共优化与真实计数

首段父 ZIP 为 C15 grouped `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`。policy/Adam/actor/log-std 摘要见 spec。父 ZIP 保存的 `num_timesteps=16384`、`_n_updates=64`，实际 Adam step=1280；祖先累计 controls=81920。不得把阶段计数误认成祖先总量，也不得清零 Adam、重置 actor head。

两臂 C18 combined、99D/16D、100 Hz、每拍 5 原生子步；相同 reward、128×128 Tanh 网络、lr=.0003、gamma=.99、GAE=.95、clip=.2、entropy=.001、vf=.5。actor（含 log-std）与 critic 分别 L2 clip=.5，参数组不交叠，仍为一次联合 backward 和 Adam step。无新奖励、网络、观测、动作权限或基控因素。

每臂每段 `32768 / n_steps1024 / batch256 / max_epochs4`，32 个完整 rollout/32 次 train，至多128个进入 epoch、512个 evaluate_actions 批和512个 optimizer。stage1 PPO seed201001，stage2 seed201011；两臂 command seed201002、measurement seed201003。首段每臂 first1024 位于相同 warmup 和首次更新前，须独立核初态、保存 Gaussian/数值数组逐位一致；课程分歧后不能要求后续轨迹相同。

公共 `target_kl=.03`。本地冻结 SB3 的实际语义是 minibatch `mean(exp(log_ratio)-1-log_ratio)>.045` 时，已经发生 evaluate_actions，但该批不 backward/clip/optimizer，退出本次 train 余下批次/epoch；`_n_updates` 仍增加已进入的 epoch。保留该行为，不回滚先前更新；每批保存实际 rollout/epoch/minibatch、KL、拒绝原因和 optimizer 是否发生。不能用 global_step//16 推导身份，也不能要求每次恰16步或补步。

任一实际 batch KL 非有限或>.30、loss/参数/梯度/Adam delta 非有限，停止阶段；任一完整 train 没有至少1个 optimizer，也停止并保留失败 checkpoint。正常>.045但≤.30为本次 train 的正常早停，不自动中止 learn。`.03/.30` 是本次固定的工程护栏，无参数扫描，也不声称由旧平均 KL 推导出唯一合理值。每个 train 保留第一个和最后一个实际 optimizer 的完整 pre/post 梯度（同一步只存一次），其他步骤保留范数、hash和真实 Adam delta；不为诊断追加模型计算。

## 2. 课程唯一变化

episode仍1000 ticks、raw前进从175开始，直到episode末；原出生点、净空.455、vy=0、jump=false。全体 first8 warmup 和 postwarmup八槽地形顺序按C15不变：flat、flat、flat-yaw、bumps、rough、ramp、flat1.6、ramp。

A 只把 C15 command selector seed换为201002，其余分布、首yaw符号抽样和450/550/750/850窗口不变。B 在非目标槽使用和A相同的按 source episode index 独立 PCG64 stream；仅在槽2(yaw)与槽3(bumps)采用下表。`cycle=(source_episode_index-8)//8`，取cycle%3；表中时间均左闭右开，未列窗口yaw为0。

| cycle%3 | B yaw速度 | yaw raw时序（rad/s） | B bumps速度 |
|---|---:|---|---:|
| 0 | 1.20 | [400,500):+.30；[500,700):−.30；[700,800):+.30 | .400 |
| 1 | 1.00 | [425,575):−.25；[575,725):+.25 | .325 |
| 2 | 1.20 | [450,550):−.30；[550,750):+.30；[750,850):−.30 | .250 |

这是联合端点、有限时序及 bumps速度覆盖的预定课程包。不能将B/A差异单独归因于某一格点。课程按index在reset前确定，不因上一episode结果改抽样，不在此实验改地形权重或取消warmup。

## 3. 有效覆盖与续段

reader对全部新训练tick复算覆盖，训练完成和覆盖充分分别记录。继续门要求两臂 postwarmup每个槽至少3个完整1000-control episode；每个计入的episode至少200个 action_gate开启且servo_vx非零的tick。RL动作恰好为0仍可计作有效权限窗口，不要求通过制造非零残差取得资格。

B还要求cycle0/2的yaw与cycle0/1/2的bumps指定episode完整；cycle0/2的endpoint合并后，在 `abs(servo_vx-1.2)≤.005`、`abs(servo_yaw)≥.28` 且action_gate开启时，正负yaw各≥100 ticks。每个指定bumps episode至少25个有效控制间隔内有真实正轮载荷；rough/ramp分别至少一个有效、具有对应地形正载荷的episode。接触从实际原生force记录核验，不从raw指令、地形标签或观测高度猜测。

覆盖不足不延长训练、不重抽episode；若两个final仍有效，可完成已列评估并报告，但不续段。无效或不足覆盖不等于科学假设已被否定。

只有开发/回归全部继续门通过才进入第二段。每臂从自己的首段final/Adam继续；新段source episode index从上一段最后实际开始的index+1起，partial尾部已消费、封存后丢弃，不补齐。新物理reset、stage2 PPO seed201011、command/measurement仍按累计source index映射。分别保存local episode index与source index；两臂若合法终止数量不同，须明示实际暴露不同，不伪称无缝物理续轨或完全相同采样。课程规则与所有门不变，只有这一次续段。

## 4. 记录、保存与每worker上限

采用C18实际控制数值；新训练记录须保存原有numeric/Gaussian及C18每拍pre/post、servo、nominal/support输入、action门、PI/z前后与连续性、请求/保护/实际执行力矩，以及5子步接触/编译几何关联。旧short_curriculum的STAGE_ARRAYS不足以独立证明z与新控制律，必须新增记录适配。clean reader核全部新tick和native记录；C15只核first1024的配对摘要不能替代此链。终止/bootstrap沿既有正确语义，episode合法任务失败照常进入下一预定episode。

两臂训练归档允许共同使用 `AtomicCourseNativeGuard` 原 `heldout` 模式的完整 contactForce 采样路径；这只读取同一实际5T的力/几何，不额外积分或改变控制。metadata必须分别写明 `env_mode=train` 与 `archive_force_sampling_mode=heldout/full_contact`，不能把训练数据命名为最终保留评估。原guard的train模式不采contactForce，因此不能用它的未采样计数证明正轮载荷覆盖。增加记录的写盘/读取时间已包含在1440 s训练host和900 s reader上界中，不另加运行。

每worker：32768 controls/163840 normal native/+2 compiler；一次learn、32 train；rollout forward≤32768 rows，evaluate_actions≤512×256=131072 actor+critic rows；边界value-only≤32、TimeLimit value-only≤32；backward/optimizer≤512，分组底层clip≤1024。单批KL检查前向已含于512次，不另开前向重算。

warm-load1、final strict-reload1，共PPO load≤2/torch.load≤6；warm/probe、save动作probe、reload probe各32 rows，共3 batches/96 actor rows。每worker总actor≤163936、critic≤163904。成功final save1；另仅允许1个独立失败checkpoint槽，save attempted≤2。失败checkpoint永不参加正式评估。唯一final取该段预算末端，不保存一串候选择优。

所有真实计数保存attempted/returned、API calls与rows；scalar `_n_updates`、实际完整epoch、进入epoch和optimizer次数分别列，不伪造固定更新数。原子提交close/fsync→rename→manifest，软停止在完整control/minibatch边界封存，首异常保留；归档失败不得finally再次写同名路径。host/reader时限见plan/spec。源码GO前的纯接口检查不允许项目checkpoint调用或机器人步骤；所有实际新训练步骤属于独占worker预算。
