# continuation15：分组裁剪的配对继续训练实验

2026-09-29，Astra。用户已授权完成诊断后的实质 RL 推进。continuation14 已实际完成并通过只读审阅，U 与 C 同时成立；据此选定本实验。**本文件冻结科学设计和执行上界；源码、来源、纯接口检查、独立 host 与实际 GO 尚须在首次调用前完整冻结，不将本文件当作已执行证据。** 旧 77 个源、模型及原始记录不改。

## 1. 唯一主要问题与对照

检验：在相同已训练 policy 与 Adam 状态出发、相同新增训练预算下，将 actor/critic 从一个全局梯度裁剪组分为两个组，是否改善实际跟踪而保持能力、安全、成本和制动。

两臂为 `global_continue`（原全 policy L2 clip=.5）和 `grouped_continue`（actor L2 clip=.5，critic L2 clip=.5）。两者用同一 loss、同一个 Adam optimizer、每 minibatch 一次联合 backward 和一次 optimizer.step。只改变 clip 组；不改 reward、vf_coef=.5、学习率=.0003、network、观测/动作、课程规则、controller、entropy、gamma=.99、GAE lambda=.95、PPO clip=.2、target_kl=None、batch/epoch。

不能只把 treatment 与未经继续训练的旧 final 比较，否则无法区分新增训练收益与裁剪机制。这里必须有同预算 global control。旧 final 不新增物理评估，因此本实验不能声称超过旧 final；可声称的比较限于新 seed 的 global/grouped/zero。zero 是实际零残差控制器，不调用策略。

单训练 seed 配对提供机制和开发证据，不是跨 seed 统计可靠性。16k/臂沿用原局部课程 first8 平地 warmup，后续地形暴露有限；不能以此宣称全地形泛化或彻底解决 critic。

## 2. 一次总预算

训练合计 **32,768 controls**，不是每臂 32,768。每臂 `BudgetSpec(total_controls=16384,n_steps=1024,batch_size=256,n_epochs=4)`：16 rollout、16 train、64 epoch、256 optimizer step、256 backward。两臂合计 512 optimizer/backward。

| 阶段 | controls 上界 | normal native 上界 | 冷构造 / compiler native 上界 |
|---|---:|---:|---:|
| global training 新 worker | 16,384 | 81,920 | 1 / 2 |
| grouped training 新 worker | 16,384 | 81,920 | 1 / 2 |
| 两新 final floor + 三 actor 六任务，新 eval worker | 30,600 | 153,000 | 1 / 2 |
| **总计** | **63,368** | **316,840** | **3 / 6** |

每个真实 control 固定 5 normal native。compiler native 另记，不混入 5T。三个 worker 串行；eval worker 内先两场 floor，再正式任务，复用同一物理 owner/已编译世界，不因 actor 切换重新构造引擎。任何另开 floor 进程都需在执行前修订构造账，不能暗加。

未通过门的后续场次不执行，额度不退款，不补步、重试、重抽 seed 或回跑旧 65,536/旧物理批次。合法任务失败与软件/归档失败分别记；失败仍是本次实验结果。

### 模型调用上界

采用每臂一次 warm load、保存后一次 strict reload、eval 各新 final 一次 strict load：**PPO load≤6**，同一 ZIP 内 torch.load 成员另列（按每次至多3，总≤18）。如实现可减少 reload，不扩大以下额度；每次 load 入口必须明确计数。

确定性 probe 使用预先固定的 32-row 旧 final probe observations，不新做物理：每臂 warm-load probe 1 batch、save 产生 actions 1 batch、save 后 reload 对照 1 batch；eval 两新 final 各 1 batch。总≤8 actor probe batch / 256 actor state rows。旧 `save_final_and_verify` 的 `reference_model=model` 还会重复预测一次 reference actions；新save15需避免该重复（保存时原actions与policy hash已可绑定），不能不计入预算。不得调用旧保存helper后又在同训练worker重复一次无必要load。

训练全部 model 计算单列：每臂 rollout policy.forward≤16,384 次/rows；train evaluate_actions≤256 batch×256 rows=65,536 actor+critic rows；rollout 边界 value-only≤16 rows，timeout value-only≤16 rows。两臂训练合计 actor rows≤163,840、critic rows≤163,904，value-only≤64。timeout 上界来自每次至少1,000 controls；终止不加 timeout value，不能为多 bootstrap 增预算。

eval 两策略正式任务各9,800、floor各600，因此 regular deterministic predict≤20,800 batch/rows；zero 六场合计9,800且0预测。连同 probe，全阶段 actor rows≤184,896；不在 heldout 调 critic。训练API调用、eval predict、probe、value-only 必须分别记，不能只用一个“推理次数”掩盖 train evaluate_actions。

learn≤2；PPO.train≤32；联合 backward≤512；optimizer.step≤512。clip 入口每 optimizer一次，共≤512；global 实际底层 clip≤256，grouped实际底层 clip≤512，总≤768。这些是裁剪函数调用，不是额外 backward。成功 final save 每臂一次，总2；每臂最多再允许一个明确标记的 failure checkpoint 归档槽，总 model.save attempted≤4，失败 checkpoint 无评估资格。无额外离线拟合、模型诊断或挑 checkpoint。

## 3. Warm-start 公平性与计数

父模型为 continuation14 核验的唯一旧 final ZIP SHA256 `6cf2db80be7b990efc8be40eff307e58351eae839970b0c78ce5c9b5193f8e70`；policy state `c20a8841dcf2174c2db584edca951adaafa12688b026172669ef0e07ad955212`；Adam state `bd4c898d33f4a121489f393dfe0f083737305f76fb139ddef74db2bb94878bf9`。

新 audited PPO subclass.load 恢复 policy、log_std、Adam moments/step/param groups；不得调用旧 build_audited_ppo，它会将 actor head 归零。两臂载入后、绑定env/设置seed后、第一步优化前均保存 hash，匹配父状态且两臂匹配。参数不得重新初始化，Adam moments 不清零，lr 不变。

parentage 保存父 num_timesteps=65,536、_n_updates=256、optimizer已完成1,024，以及父 manifest/metadata/SHA。新阶段 num_timesteps/_n_updates 置0只用于本阶段计数；常量 lr schedule在两臂相同，必须静态确认该重置不改变 lr。完成时新阶段16,384/64 epochs/256 steps；每条分支累计81,920 controls/320 epochs/1,280 steps，不能把两条分支的共同父训练重复计入新增工作。

加载后绑定新 curriculum 和新 `LearningAudit`，记录 initial snapshot 与 warm-start construction。新 seed：PPO **151001**，command selector **151002**，measurement stream **151003**；两臂相同，且与旧886xx不同。用原 episode局部规则，source_episode_offset=0，不改变 first8 warmup 或后续8-slot cycle。每个 episode 的 schedule、seed和raw commands在执行前确定，不根据一臂表现选择。

两臂独立 fresh runtime、相同reset状态/geometry/随机源。前1024 controls处于首次参数更新之前，应按数组核对两臂 first-rollout observations、Gaussian actions、rewards、done和reset身份的一致性；若不一致，不将两臂视为严格机制隔离，并保存原因，不回跑。算法分支在第一次 clip 才允许造成参数差异。

## 4. 真实梯度、裁剪和参数步长证据

actor组为 pi MLP、action head、log_std，预期7个 tensor；critic组为 vf MLP和value head，预期6个 tensor。要求完全覆盖可训练参数、identity不重叠、无可训练共享 FlattenExtractor；不把共用无参数展平误写成共享表示。

只在 audited super.train 的作用域拦截原 `torch.nn.utils.clip_grad_norm_`；global 分支调用原函数一次并保持其返回值；grouped 调原函数两次（actor/critic），不递归调用包装器。finally 恢复入口。与optimizer pre/post hooks匹配，一次clip入口必须对应恰好一次真实step；发生异常记录已发生的partial次数。

每个真实 optimizer step 保存：rollout/epoch/minibatch/全局step，actor、critic、joint的preclip与postclip L2，各组预期系数及实际范数比，finite/缺失梯度计数，Adam前后actor/critic参数delta L2及相对参数norm；至少记录每组gradient摘要hash。联合loss的梯度因参数完全独立，actor组包含policy+entropy梯度，critic组包含vf_coef加权value梯度，不另做actor/value backward。

每 rollout固定保存第一个和最后一个 minibatch 的完整pre/post梯度（每臂32个已预定step），用于独立纯读回复核；其余step保存norm/hash即可。所有step的clip比例需与实际实现相符，建议rtol=3e−5、atol=1e−8；global postjoint≤.50001，grouped各组≤.50001且joint≤sqrt(2)*.50001。grouped joint允许超过.5，这是预定机制，不是预算违规。

不要以clip factor冒充参数步长：必须另报实际Adam delta。保存既有KL、clip_fraction、value_loss、EV、std和stage optimizer计数。target_kl仍None，不新增KL早停算法；非有限值、物理硬门和来源失败照常停止，未达预算不伪称完整更新。

## 5. 新 seed floor与完整评估

三actor为 `zero`、`global_continue`、`grouped_continue`，所有正式任务共用同一组新物理seed、commands、spawn、compiled geometry和评分窗口。每个actor×task必须重新reset到匹配状态，逐位核 qpos/qvel/ctrl/warmstart/99D initial obs；不拿模型名称相同代替配对身份。

eval worker先分别对两个新final执行唯一floor（每场≤600，seed151090）：flat，0–174 settle，175–424 raw vx=.4，425–599 raw stop，yaw=0，clearance=.455。floor门：完整600/3000返回、来源/观测/动作finite、无warning/geometry问题、原安全10° roll/pitch、clearance≥.28m、无nonwheel/地图越界；[325,425)真实COM均速与.4偏差≤.2m/s、vx RMS误差≤.25m/s、yaw RMS≤.15rad/s。不延长短停止窗口，也不以600控制取得完整制动或速度资格。

某新final floor失败则跳过其所有正式场；不调模型或重试。如果两个新final都失败，eval在两floor结束后停止。只有一个通过时，可完成该actor及zero的六任务，但机制比较未完整，不作promotion成功。floor不是挑选checkpoint的环节。

正式六任务保留旧数值脚本和评分定义，**只更换独立seed**；旧成绩不拼入本次结果：

| task | terrain / vx / yaw amplitude | seed | controls | hold / release / final |
|---|---|---:|---:|---|
| flat_0p6 | flat / .6 / 0 | 151101 | 1600 | [295,695) /695 /[1500,1600) |
| flat_1p6 | flat /1.6 /0 |151102|1600|[495,895) /895 /[1500,1600) |
| flat_1p2_yaw |flat /1.2 /.3|151103|1600|[415,815) /815 /[1500,1600) |
| bumps_0p4 |bumps /.4 /0|151104|1600|[255,655) /655 /[1500,1600) |
| rough_0p35 |rough /.35 /0|151105|1600|[245,645) /645 /[1500,1600) |
| ramp_0p45_complete |ramp /.45 /0|151106|1800|[600,1000) /1355 /[1700,1800) |

每actor合计9,800；三actor29,400，加两floor1,200为30,600。按task表顺序进行，每task内zero→global→grouped；不因结果改变顺序。合法任务失败保存终止记录后可按合同继续下一场；引擎/归档/模型身份失败则停止worker，不能把损坏记录当作合法物理失败。

每case直接复用旧纯数值scorer的安全、COM speed/yaw、terrain channel、真实ramp geometry、release后stopping与final窗口门。真实1.6能力必须包含COM均速/RMS/超过1.5的占比，不能用指令或轮速代替。模型release后effective动作归零时，制动不归功于RL。

## 6. 预注册成功与拒绝门

分开报告工程有效、机制成立、物理收益、相对zero的RL收益。所有 ratio 分母为0时不加epsilon制造结果；缺失或undefined保持不通过/未定。

**工程完整性：** 两训练完整预算、identity/count/native/optimizer闭合；两newfinal唯一；floor与18正式任务按合同完成或合法失败，记录可独立读回；两臂配对身份成立。任一核心证据无效，不能做机制promotion。

**裁剪机制：** 每step真实pre/post与原/global或grouped公式一致，actor与critic无共享。全体统计显示预期通路是否实际发生；记录global实际actor保留率，以及grouped同一pre梯度下假想全局clip的actor保留率与实际grouped保留率。后者仅用于同一实际梯度的代数对照，不是反事实训练轨迹。必须再报告实际Adam参数delta；loss/clip数据单独不算控制收益。

**物理promotion门（grouped 相对 global）：**

- grouped六任务全部 task_passed=true、完整horizon、安全与制动各门通过；不允许global通过而grouped失败。
- 至少4个双方task通过的配对；原drive归一化跟踪SSE在这些共同通过任务中的 pooled grouped/global≤**.85**，且至少4对各自≤**1.02**。定义沿用原评分的vx/.25、yaw/.4平方和，窗口不事后选择。
- 四个诊断主任务（flat.6、flat1.6、flat1.2yaw、bumps.4）各自hold SSE grouped/global≤**1.02**；其中flat1.6真实vx hold RMS grouped/global≤**.90**。这防止只改善加速而保持继续变差。若global对应SSE或RMS恰为0，则仅双方0可满足无退化门，但不能据此建立相对改善。
- 六任务共同完成drive区间内归一化力矩平方的pooled grouped/global≤**1.20**（成本代理，不称能量）；同时报告每task比值。安全、停止距离/耗时、速度与yaw的绝对门仍优先，不能以成本门放宽它们。

以上全部成立才称本次配对开发实验证实grouped相对同预算global有跟踪收益。若global有合法任务失败而grouped通过，只另报能力转换；缺少至少4共同通过任务或hold比较时，不冒充此tracking promotion通过。若仅loss或gradient改善、物理门不满足，则拒绝将grouped设为默认；不自动加训练预算。

**相对zero的RL贡献：** 分别对global/zero和grouped/zero应用原六对branch A/B和cost门（原branch B pooled SSE≤.85、至少4对≤1.02；cost≤1.20；能力/安全先决）。两条比较各自报告，不因grouped赢global就宣称赢zero。只有相对zero完整贡献门通过才能作对应表述；单seed仍不代表可靠部署。

无本次旧final对照，不能写“相对冻结final提升多少”；旧四对退化仍保留历史事实。GUI、15mm台阶、侧移、跳跃、实机不属于本实验资格。

## 7. 时间、归档与停止

依据旧65,536训练约1,348 s，16,384线性估计约337 s，但不把它当保证。每训练worker软停止600 s、close deadline720 s、host硬750 s；eval worker软停止1,800 s、close deadline2,040 s、host硬2,100 s。三个host总硬上界3,600 s，另每host最多5 s进程回收；全阶段时间只上限，不要求花满。限制自各worker启动算起，包含import、构造、load、数据写盘和封存；源清单仅含实际依赖，避免重复CUDA副本无关哈希消耗。

soft stop只锁存，不在gzip/当前tick内部抛异常；物理owner完整tick后停止发新调用，在已完成边界封存。正在进行的minibatch允许完成已进入的一次optimizer原子步骤后停止，不启动新的minibatch/rollout；不能为凑满计数继续。独立host兜底，无进程残留。所有失败/停止计入已消费预算，首异常独立保留。

复用已测试archive13的临时块close/fsync→原子提交→manifest最后登记纪律及幂等封存，桥接training和evaluation；不能直接复用旧guard.finish_segment的已知中断覆盖路径。保存足以独立核验的control/native/force链、numeric+Gaussian、梯度、schedule/seed、checkpoint、C/Python attempted/returned及合法失败前缀。compact记录不能未经资格证明就代替所需force/native证据。

每阶段开始独占reservation、冻结GO/source/dependencies/旧父模型/77源/recipe和score；阶段后同集hash、第一异常/cleanup/耗时/退出/剩余额度废止。原文、失败、旧plan和新revision各自保存，不覆盖。最终独立读回仅解析已保存数值，不再做物理或加载模型。

## 8. 实现分工与最小改动边界

- **Sol learning：** 新learning15.py，warm-start/audit、scoped clip、真实pre/post及Adam delta、必要synthetic纯测试。不得改旧learning11；global分支必须保留原SB3更新数值。
- **Sol recipes：** 新recipes15.py，训练原规则加共同新command seed151002、offset0；新heldout/floor脚本与门；必要纯测试。原recipe只读。
- **root：** 新training/eval worker、archive13桥接、host/计数/来源、唯一实际执行、保存和独立读回。root决定常规实现细节，无需再问例行许可；不得变更本科学因素或暗加执行额度。
- **Astra：** 14结果审阅、本合同、首次物理前集中静态审阅和实际结果只读终审。未通过处给具体修正，不反复跑旧批次。

实现与必要纯接口检查完成、计数与来源GO冻结后，按用户已授权内容连续完成训练和有条件评估，不以交付计划代替实验。失败也须完整闭账并据实给下一决策，不能为了得到正结果重复本预算。
