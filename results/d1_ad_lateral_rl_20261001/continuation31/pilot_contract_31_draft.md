# C31：有界事件级横移 RL pilot 草稿

状态：实施设计草稿，尚无模型、物理或训练 GO。C30 六场配对触发门已通过；本文件不改变 C30/B22 的冻结源或记录。主审实际模型 gpt-6-astra，reasoning effort ultra；root 独占执行与测试。

## 1. 目标与最小变更

C30 固定参考已将左右周期从 10.55 s 降至 9.69/9.71 s，尚无学习。本 pilot 检验：在同一 30 mm 落脚目标与既有安全投影下，事件级学习是否进一步缩短周期，并保留位移、落脚、停止和取消能力。不能把 37 mm 实际超调作为额外正奖励；不承诺 2 倍或 0.01 m/s。

只新建 C31 模块：复用 C30 headless owner、同 env/plant、B22 载入与 99D/16D rolling 接口、三 MjData、两条 copy 边、真实 5T 记录、collision_bounds24、侧步 IK/接触门、宏分量与独立 reader 子函数。B22 不训练；side 拍仍 0 次 B22 predict。早降继续禁用。没有 GUI/渲染性能分支，没有重新执行 C30 六场。

## 2. 真实宏接口与动作

在旧 `_prepare_leg()` 产生 teacher plan 后、额外 IK/动作投影前，只调用一次 `sample_latch31(observation54, teacher_plan, cycle_index, local_control_index)`。动作从当拍实际 control i 开始；前宏为 [previous_latch,i)。每周期最多四个真实动作，末宏延至实际 handoff 后 400 拍保留结束。100 Hz 记录不会变成额外 RL 样本。

定义 C30 固定候选 f：沿 teacher body_to→body_from 回缩 min(8 mm, span)，在初始机身 yaw 坐标表达 f_xy，f_lambda=.5。新 actor 输出 latent z∈R³：

- proposed_xy = f_xy + .004*tanh(z_xy)，再按原合同限制各分量 ±.012 m；
- proposed_lambda = .5 + .5*tanh(z_lambda)；
- 继续原一次额外 IK、关节范围、8 mm IK误差、28 mm tripod margin、120 mm shift 限制；XY拒绝回 teacher，lambda 仍独立有效；不追加 IK 搜索。

Gaussian latent log-prob 用于 PPO；保存 z、均值、固定标准差、log-prob、proposed/projected/consumed 和投影原因。不得在 fallback 后把 consumed action 当作原采样动作反算 likelihood。确定性 z=0 明确调用原 C30 fixed 路径，避免 +0 运算破坏来源桥；teacher-zero 模式直接旧 super 路径。非零采样须完整走新权限，不伪称首训练周期与 fixed 同轨。

## 3. 固定 54D 观测

全部来自本拍尚未积分 measurement、原 teacher plan 与已保存前动作；不新增 engine 查询，不使用 `_targets` 后未来缓存。按以下顺序保存 raw、normalized、clip mask；连续项 clip[-5,5]，离散项原值，非有限值即停止。

| 索引 | 量与固定尺度 |
|---|---|
| 0 | direction，±1 |
| 1:5 | 当前 leg one-hot，四腿固定原编号 |
| 5 | macro_index / 3 |
| 6:8 | 相对本周期侧步起点的 XY，在初始 yaw 坐标投影 / .12 m；Y不按方向镜像 |
| 8 | (base qpos z − .455) / .05 m，平地专用 |
| 9:12 | 世界 roll、pitch、wrapped(yaw−initial_yaw) / .12 rad |
| 12:15 | free-joint 平移 qvel 的初始 yaw 坐标分量 / [.2,.2,.1] m/s |
| 15:18 | 原 free-joint angular qvel[3:6] / .5；按模型原坐标命名，不伪称 COM 速度 |
| 18:30 | 四腿每腿前三个关节的 qpos，按实际 compiled range 中点/半宽归一化 |
| 30:46 | 实际 actuator joint dof 顺序的16个 qvel / [4,4,4,20]×4 |
| 46:48 | teacher(body_to−body_from) 的初始 yaw XY分量 / .12 m |
| 48 | teacher_shift_time / 1.5 s |
| 49 | plan_com_height / .5 m |
| 50:53 | 前一次实际 consumed_action / [.012,.012,1]；首宏为零 |
| 53 | .01*(当前 local_i−side_start_i) / 15 s；首宏0，不用 `_advance` 已预增的 self.time |

关节索引由 compiled actuator/joint map 验证，不能只凭数组碰巧排序。没有 running normalization 或训练后改尺度。terminal next observation 如不满足锁存前观测结构，只保存 raw terminal state；terminal value 严格为零，不伪造第五腿。

## 4. 一个 pilot 与实际优化次数

一进程最多 64 周期、256 真宏、140800 controls、704000 正常 native；任一帽先到即停止，不补步、不续段。固定64周期列表左右严格交替，32+32，同 C30 名义初态；policy seed=310031，环境 seed沿271001。随机策略动作提供探索，seed不冒充独立物理随机化。每周期200准备+侧步≤1500+400保留，控制上限仍2200。

actor=独立 Linear(54,3)，value=Linear(54,1)，CPU float64；权重/偏置全零初始化，无隐藏层，latent std 固定 .35，不学习方差。deterministic 初始策略等于 fixed 候选，训练使用真实 Gaussian 采样。Adam 的 actor/value 两参数组均 lr=.001, betas=(.9,.999), eps=1e-8；clip_ratio=.2；无熵奖励。actor 与 critic 分别按各自梯度范数裁剪到 .5，不跨组做 global clip。采用已核本机 PyTorch clip_grad_norm_ 公式 factor=min(1,.5/(raw_norm+1e−6))，foreach=False、error_if_nonfinite=True；此1e−6与Adam eps1e−8分开。两组共享同一次 optimizer.step 调用，最多64次调用，两组各自 moments/step 分开记录。独立裁剪沿用 C14 后已明确的解耦原则，避免零初值 critic 的大梯度缩小 actor 更新。

每8个已关闭周期一批，实际宏 n≤32，失败短周期不填充、不重复。最多8批；每批2epochs、batch8（末batch可不足8），最多64个真实 optimizer steps。每批保存实际 epoch/minibatch/forward/backward/step 数；KL早停时不宣称名义64次已发生。完成整 pilot 需每批至少一次有效优化，且 actor 至少一次非零梯度与参数改变证据。

SMDP gamma=1；delta_i = r_i + V(next_i) − V_i；terminal V(next)=0。GAE_i = delta_i + .95**elapsed_seconds_i * GAE_(i+1)，只在同一周期逆推，绝不跨 reset。value_target=未归一化GAE+old_value；整批实际 n 个 advantage 做一次归一化：(A−mean)/(population_std+1e−8)，population_std=sqrt(mean((A−mean)²))；eps不放根号内。loss = mean(PPO clipped surrogate negative) + .5*mean((V−target)²)。

每 minibatch 更新前核该 minibatch 实际行的 fixed-std Gaussian 解析 mean KL；这次 actor forward 同时供 loss 使用，不为 KL 重复调用；即使 guard 拒绝也计实际 forward 尝试。最多64个 minibatch 尝试（不是64成功更新再加额外无界尝试）。> .03 则正常停止该批余下优化。每次 optimizer 后在本批全部 n 个实际 observation 只做一次 actor/no-grad 复算 KL（不做 critic）；> .10 或任何非有限 loss/gradient/parameter 即学习失稳、整 pilot 停止，不能挑更早checkpoint作成功。每个 minibatch保存 before/after parameters、两组各自裁剪前/后gradient及norm/clip factor、Adam moments/step、loss/KL；独立 reader 可用纯数组复算。只 final checkpoint用于最终比较，无 best-checkpoint选择。

## 5. 奖励与失败成本

沿 C30 全部真实宏分量，使用实际 post endpoints 与5×16力矩：

r = 10*progress − elapsed_s − .05*backtrack_normalized − .02*longitudinal_normalized_square_integral_s − .02*lateral_goal_error_normalized_square_integral_s − .02*yaw_error_normalized_square_integral_s − .5*torque_normalized_square_integral_s + terminal_adjustment。

progress沿 min(s/.03,1) 势差、无下界截断，超出30mm不会额外加分。成功闭合且原完整任务门通过 terminal_adjustment=+5。受控任务失败为 −50−max(0,22−实际全场秒数)，后项是明确的终端不足时长惩罚，不是虚构执行时间/native。所有实际宏成本（含取消/失败已执行部分及实际400保留）保留；不得把成功判定前的残差片段删掉。

只有来源/归档/native/fence完整且已安全落脚交接、完成400保留、末100停止载荷合格的任务指标失败可以产生terminal failure奖励后reset。例如位移/目标误差未过，但真实安全链全部有效。完整性失败、native硬安全失败、未安全交接、预算/超时中止均停止整 pilot；不把不完整 native 区间造为完整transition。最后未闭合批保存而不更新。连续2个或累计8个受控任务失败亦停止，不追加周期救援。failure checkpoint只审计，永不作为合格final。

## 6. 多周期记账与关闭

env/plant/B22、3 MjData、geometry binding只构造一次，预期 compiler native +2；每周期只 env.reset，不重新构造。每次reset前封存旧 segment/宏/局部receipt；清除旧 `_skill30_interval_evidence`、last_skill30_record、macro events、Hybrid accepted_starts/intent/handoff 等。不得让旧5T接触证明进入新episode。SideAccess、C/Python、模型和全局control计数不能清零；每周期差分限制 start≤1/compute≤1500/prepare≤4，NativeWindow local index与新segment严格重新绑定。逐行保存 cycle_index、local_control_index、global_control_index、local_native_start、global_native_start，global=该场起点+local，5T关系可独立重算；原始全局计数不改写。复用旧单场纯函数时只能构造经映射校验的局部视图，不得伪称原全局账从零开始。

训练全局 static caps=原单场上限×64（详spec），包括extra IK；pure FK/native shape查询另记，不算integration。B22：load1、torch_load3、probe1×32rows、control predict≤38400，总predict≤38401/actorrows≤38432；B22 train/optimizer0。

lateral actor sampling≤256rows/value≤256；minibatch actor/value各≤512rows；post-update KL actor≤2048rows；final save前后固定16row probe各一次，actor/value各32rows。因此 training lateral actor total≤2848rows，value≤800rows，backward/optimizer≤64。final save1、reload1；另failure save≤1且不得同时冒充final。最多两套lateral参数对象（训练与最终独立reload）。所有actual而非仅nominal记录。现 checkpoint31 采用 weights-only Torch 权重保存/重载：训练worker全局 torch.load 帽为4（原B22严格probe阶段3＋lateral_final阶段1），eval同样4（B22 3＋lateral_load 1）。不得为侧移重载绕过原全局hook；B22分账仍3。原B22 32行probe沿旧 phase=probe，其他phase不能冒领32行资格。lateral训练另明确torch.save≤1（失败审计save额外≤1、不能同时作为final），各阶段attempt/return独立记录。

候选wall帽：training soft4200/close4500/hard4560/outer4800 s；source preflight240/postcheck180，均含在outer内，不增加控制。headless完整保存读回单worker hard3600 s。root可在source GO前核运行成本后收紧，不能运行后放宽。没有RTF门，也不把训练速度当GUI性能证据。循环每周期及时flush，无GUI deferred archive/join。

## 7. 唯一 final 与新保留条件

训练前封存最终列表/权重选择规则；只用最终同一个deterministic mean checkpoint，不看heldout再训练。新评估最多10场×2200=22000controls/110000normal，建议一coldworker复用同model/reset（compiler+2），逐场闭合；任何安全/来源失败即停。候选eval wall soft900/close960/hard1020/outer1500 s，独立reader hard1200 s。

1. nominal learned 左/右2场；与C30已封zero/fixed同方向做完整reset+200prefix+入口桥。桥失败则这些旧baseline不得直接作公平比较，也不得现场补跑；本pilot结论受限。
2. 真正未训练条件：initial yaw +.04 rad的left、−.04 rad的right；其余spawn/地形/目标/准备/保留不变，各zero、fixed、learned3actor，共6场。reset yaw须在prepare前真实设置且整个机器人姿态/数据/provider一致，先纯几何与来源reset桥；每条件三个actor完整初态/200prefix相同。保留条件与模型在训练前封存；C30名义初态不称heldout。
3. learned 左/右首个真实双速度重叠下一拍取消2场，沿≤300安全交接+400保留+末100原门。

评估模型帽：B22 load1/torch_load3/probe1×32rows/control predict≤6000，总predict≤6001、actorrows≤6032，B22学习0。lateral checkpoint load1、参数集1，固定probe actor/value各16rows；六个learned场最多24个实际actor事件，因此actor总≤40rows、value≤16rows，deterministic评估事件不调用critic、不采样RNG，save/train/backward/optimizer均0。四个zero/fixed场不调用lateral actor。所有帽按实际调用分类闭合。

新能力门完全沿C30。学习收益门：nominal和两个新条件每个方向均 cycle learned≤.90zero 且≤.95fixed；目标误差≤zero+.002m，保留≥max(.90,zero−.02)，全场tau²mean≤1.20zero且≤1.10fixed；所有完整任务、配对和取消通过。没有总体平均掩盖单向退步。纯速度阈值不是训练保证；若未过，公开真实学习结果，停止扩训，不能把固定8%改进冒称RL收益。

这10场是研究评估，不是GUI A/D交付。之后GUI只接实际合格的选择并明确actor/fallback身份，另有界用户输入验证；本合同不授权该GUI物理。

## 8. 分工与前置

root：new headless多周期owner/recorder/native与model budgets、版本host、来源/候选/实际执行/发布；Sol：observation/action_callback/小型SMDP PPO与checkpoint纯模块；独立Astra：saved-only新transition/reward/GAE/Adam/readback、条件配对与final gate；本Astra：合同/spec与源/结果终审。

必要最小验证：54D索引/归一化；mean0→原fixed、teacherzero→原路径；有界action+原projector；不同duration的GAE解析例；失败成本不省略；实际Adam一步/枚举KL早停；重复reset的旧interval隔离与局部/全局差分；共享真实bootstrap入口和完整finalize保存回归。这些纯/合成learner测试不是机器人实验，若调用合成Torch网络需诚实另计，不能声称所有model调用0。新runtime源不能在首训练后改。所有先决、checkpoint/heldout表与实际tests receipt哈希齐后才签单次训练 GO。
