# continuation14：冻结 final 的保存行为窗口价值与梯度判别合同

日期：2026-09-29。此文件为 Astra 静态方案，不是执行回执。方案作者仅阅读保存源码、JSON 和清单，没有加载模型、运行前向/backward、物理、测试或训练。root 是唯一真实模型调用 owner；Sol 可实现纯数据准备与纯数学。旧记录和 77 个冻结源均不改。

本合同足以在既定上界内完成下述诊断，无须增加 value 前向或 backward。结果仅用于选择下一项单因素研究候选，当前合同不授权新训练。

## 1. 问题、已有证据和四个可区分假设

要回答的是：当前 final 在固定的历史训练行为窗口上，价值预测是否呈系统尺度偏差；该代理目标的 critic 梯度是否足以单独触发强全局裁剪；偏差是否集中于早期行为或特定状态组。

已有保存事实来自 continuation13/diagnosis_13.md、bootstrap_source_audit_13.md 和 rl11/training_run_01：训练 reward 每步均值约 3.94；历史 value loss/EV 提示拟合问题；四对完整评估显示跟踪退化。历史逐转移 value、GAE、preclip actor/value 梯度未保存。TimeLimit 静态代码未见漏加或重复加证据，不能据此核验历史数值。

按优先级冻结四个假设：

1. **广泛的当前价值校准偏差。** 若成立，64-step residual 在多个训练时间组、包括最后组中有一致方向，且相对于奖励所定尺度达到第 7 节的偏差门。若只有早期组偏大，广泛偏差假设减弱。
2. **当前 critic 代理梯度可产生强全局裁剪。** 若成立，四个固定批次中至少三个（包括最后批）weighted critic-only 范数达到 5.0；相对于历史 0.5 clip cap，其单独裁剪系数不超过约 0.1。若四批均不超过 0.5，本诊断不支持该代理上的强制裁剪压力。
3. **训练时间或状态分布混合主导总体指标。** 若前期残差明显较大、最后组低于门，或误差只集中于足量的 terrain/速度/gate 组，则不应由总体负 EV 推出全局奖励尺度需要修改。时间分组仍混合了行为策略变化与课程变化，不将它称为单独的“策略陈旧”因果证明。
4. **计算路径、参数共享或数值完整性与预期不符。** 按源码，pi/vf 是独立 [128,128] Tanh MLP；共享 FlattenExtractor 没有参数，PPO 对全 policy 参数一次全局 clip。若加载后发现可训练共享参数、critic-only backward 产生 actor 梯度、有限值/头部 bias 解析校验失败，先停止解释，修诊断或核来源，不启动学习。

## 2. 预算、身份与禁止事项

| 项目 | 硬上界 |
|---|---:|
| PPO.load 尝试 / 成功 | 各最多 1 |
| 固定训练起点 | 512，四批各 128 |
| 每个奖励窗口 | 64 个 transition |
| value state forward 尝试 | 1024 个 row；重复状态仍逐次计数 |
| value API 调用 | 5：端点 1×512，起点 4×128 |
| critic backward 尝试 | 4，每批一次 |
| actor forward / predict / evaluate_actions | 0 |
| optimizer.step / learn / train / model.save | 0 |
| 新物理、引擎构造、reset、GUI | 0 |
| 自动重试 / 更换样本再运行 | 0 |

模型文件必须为旧 final_model.zip，SHA256 `6cf2db80be7b990efc8be40eff307e58351eae839970b0c78ce5c9b5193f8e70`；trusted policy-state SHA256 为 `c20a8841dcf2174c2db584edca951adaafa12688b026172669ef0e07ad955212`。以旧 final_checkpoint_manifest.json、metadata 与旧闭账清单交叉核身份。

使用新的诊断 loader，仅调用一次 `PPO.load(..., env=None, device="cpu")`，核 ZIP/metadata、state hash、99D float32 ±5 observation space、16D float32 ±1 action space、独立 Tanh 结构、无 VecNormalize、模型计数及超参数。`gamma=.99, gae_lambda=.95, vf_coef=.5, max_grad_norm=.5, clip_range_vf=None, n_steps=1024, batch_size=256, n_epochs=4` 必须符合旧清单。仅 `.eval()`，不生成 rollout buffer 数据，不调用训练入口。

旧 `load_and_verify_final` 会额外调用 32-row actor probe，因此本合同不调用它；本次不是重新运行行动探针资格验收，不宣称本次已复验 actor probe。不得为了审计而偷偷增加预测。

## 3. 纯数据准备与抽样冻结

读取完整的 64 个 numeric block、training_blocks_manifest.json 和涉及的 episode schedule。先按旧闭账清单核每个输入 bytes/SHA256，再用 `allow_pickle=False` 读取 numeric NPZ；不导入物理模块，不反序列化模型。要求总数 65,536，control_index 从 0 连续递增；所有所用数组长度、dtype、shape 与 schema 一致。

将控制索引分成四组：q=floor(control_index/16384)，q=0,1,2,3。候选起点 t 必须满足：

- `episode_tick[t] % 64 == 0`。只使用此预定 episode 格点，不看奖励或预测挑起点。
- t 到 t+64 共 65 行存在，control_index 和 episode_tick 每行加 1，episode_index 完全相同。
- t 与 t+64 在同一时间组 q；64 个奖励 transition `[t,t+64)` 的 terminated 和 truncated 全为 false。端点自身 transition 是否结束不影响端点前观测，但须仍满足同 episode 连续性。
- 起点及端点观测来自原 `input_observation99`，float32 [99]、有限且在 ±5 内。无 post-step observation 字段时，唯一合法端点是同 episode 的 `input_observation99[t+64]`；不以最后一个 input 或下一 episode reset observation 代替。

这些格点保证奖励窗口互不重叠；相邻窗口可共享一个端点/起点状态。共享状态重复送模型仍重复计入 1024 row 预算。候选不得跨 timeout、termination、数据缺口或时间组。末尾不足 64 步的窗口舍弃并计理由；不填零、不缩短窗口。

每组候选按如下 UTF-8 文本的 SHA256 十六进制字典序排名，取前 128：`d1-value14-selection-v1|{q}|{control_index}`。hash 冲突以 control_index 升序破同分。若任一组少于 128，纯数据门失败；不能改格点、扩大预算或重抽样。选定后执行顺序为 q 再 control_index 升序，得到固定四批。无须 RNG。

在任何 PPO.load 之前，以独占新文件冻结 selection JSON 和 data NPZ。JSON 至少含：算法/version；所有输入 hash；每组候选及排除数量；512 个 start/end control_index、episode/tick、block/offset、q；64 rewards hash、start/end observation bytes hash；固定批次边界；所有分组元数据；selection/data/方案/实现源 SHA256。root 的一次性 reservation 引用这些 hash 后方可加载。不得因之后看到 value 改样本。

奖励采用原 numeric `reward` float64，即 environment_returned_prebootstrap_reward_f64，64-step 累积用 float64。它与 callback 的 float32 reward 表示可能有舍入差异；本合同没有声称重建历史 PPO target。检验 reward_terms9 求和与 reward 的保存容差一致。数据身份或有限值失败属于 gate failure，不能简单删除坏行继续凑数。

## 4. 返回目标与解释边界

令 γ=.99，H=64，ρ=γ^64，D=(1−ρ)/(1−γ)。每个已选 t：

`G_t = sum_{k=0}^{63} γ^k * r[t+k]`

`Y_t = G_t + ρ * stop_gradient(V_final(obs[t+64]))`

`δ_t = Y_t − V_final(obs[t])`

正 δ 表示当前起点值低于此保存行为加 final bootstrap 的目标。名称必须为 **saved-behavior 64-step bootstrapped residual/target**。这不是历史 TD(λ)/GAE，不是 final on-policy 回报，不是 Monte Carlo 真值，也不是 critic 的独立无偏误差；旧行为分布、当前 bootstrap、时间截断和课程均参与目标。

由于已排除 64 步内的 episode 边界，不做 timeout reward correction、不额外加 terminal V、不使用下一 episode 值。不能由本实验重验历史 TimeLimit 处理。

1024-row 上界恰好覆盖 512 endpoints + 512 starts。不在所有中间状态评 V，因此不计算 λ=.95 的 64-step λ-return。若后来要此类新目标，需先新合同，最直接的上界为 512×65=33,280 row（可按唯一状态缓存减少）；历史精确 GAE 因历史权重和值未保存，增加预算也不能还原。

## 5. 五次 value 前向、四次 semi-gradient backward

加载后，先按原 hash_state 语义核 policy state；另记录每个 named parameter/buffer 的 shape、dtype、bytes hash，及 optimizer state hash（若 loader 已恢复该状态）。保存 actor-only、critic-only、共享参数的 identity 集合和实际参数数量。FlattenExtractor 共享对象不等于共享可训练表示；这里预期共享可训练参数为 0。

1. 为 512 个端点执行一次 `no_grad predict_values`，记录输出并 detached。不得用 `policy.forward`，因为它还走 actor。
2. 对四批各自：令所有参数 `.grad=None`，以一次启用 autograd 的 `predict_values` 得 128 个起点值。立即计算 `L_b = .5 * mean((V_start − Y_detached)^2)`，调用一次 backward。
3. 保存该批已计算起点值、未加权 MSE、weighted L、梯度统计；然后释放 graph、清 `.grad`，进入下一批。不能先 no-grad 算 starts 再重算用于 backward，不能跨批累计梯度，不能执行 optimizer.zero_grad 所属的训练包装器或任何 optimizer.step。

可将 Y_float64 转成与 value 同 dtype 的 detached 目标用于 loss；报告原 float64 Y 和实际 loss target float32，两者及差额有限。backward 解析校验使用实际 loss target 与实际 value。

每批必记：critic 总 preclip L2 范数 Gc、每层 weight/bias 范数、相对参数范数（分母 max(param_norm,1e−12)）、actor 梯度 None/零/非零数量、共享参数数量、value head bias 梯度。校验 `head_bias_grad = 2*.5*mean(V − target_float32)`，建议 `abs(error) ≤ 1e−5 + 1e−5*abs(expected)`。critic 参数不应缺失梯度；actor 梯度必须全部 None 或精确零；异常即停止解释。

不执行 clip_grad_norm_，只由保存梯度算 `α_c=min(1, .5/(Gc+1e−6))`。四批都是大小 128 的代理目标批次；历史 batch_size 是 256，因此不声称这些范数等于历史 minibatch 梯度。

如果 pi/vf 参数确实不重叠，在**同一个假想代理目标**下加入任意 actor 梯度 Ga，则总范数为 sqrt(Gc²+Ga²)，全局 clip 系数不大于 α_c。这个数学关系只证明可能的全局标量通路；没有测得 Ga、梯度夹角、历史竞争或 actor 的实际更新。Adam 状态还影响最终步长，不能把 α_c 直接叫 actor 更新缩小倍数。

不增加为诊断而做的 actor 前向或合成 actor loss。可由原有 value 前向 hook 记录 Tanh 激活，但它不是必要判别门，不为此增加调用。

## 6. 固定分组与必须报告的统计

除全体和四个 q 组，预先固定以下描述性组；组间允许重叠，不做事后挑组：

- terrain：episode schedule 顶层 terrain，若 choice.terrain 也存在则须相同；核 episode_index。
- 有效动作占比 f：64 步中 any(effective_action16 != 0) 的平均，分 f=0、0<f<1、f=1。称“保存有效动作非零占比”，不将中间保护 mask 当作真实动作损失。
- 速度：mean(abs(servo vx)) 分 `<.05`、`[.05,.8)`、`≥.8` m/s。
- 转向：64 步 any(abs(servo yaw)>.05 rad/s) 与否。

每组报告 n、episode 数、G/start V/end V/Y/δ 的 mean、median、std、min/max、5/95 percentile、MAE/RMSE、δ>0 占比。n<32 的组只列描述值，不能单独推动训练选择。四组 n=128 的批次梯度分别列，不能捏造 terrain-specific backward。

proxy EV=`1−Var(δ)/Var(Y)`，仅当 Var(Y)>1e−12；否则 null 并说明近常数目标。另报 bias、RMSE，防止 EV 对常量偏差不敏感。任何 R²/EV 都只针对本代理目标，不复制历史 train/explained_variance 标签。

## 7. 预先冻结的筛选门与决策

下列阈值是本次预注册的工程筛选门，不是统计显著性门或通用 PPO 定理。每组 g 定义 `S_g=max(1, median(abs(G/D))/(1−γ))`，`B_g=median(δ)/[(1−ρ)S_g]`，`E_g=RMSE(δ)/[(1−ρ)S_g]`。S 只由保存奖励定标，避免当前 V 同时放大目标和归一化分母。`median(δ)/(1−ρ)` 仅是“假设所有状态 V 都加相同常数”时的等效校准位移，不是已证明的真实 value error。

- **广泛欠估筛选门 U：** 全体 B≥.10 且 δ>0 比例≥.75；四 q 中至少三组 B≥.10，并且 q=3 也≥.10。广泛高估 O 完全对称（B≤−.10、δ<0≥.75）。满足才称“一致方向的当前代理校准偏差”。
- **大而非统一偏差 M：** 全体 E≥.20 但 U/O 均不成立。优先分组定位，不能据此只改全局奖励尺度。
- **强 critic clip 压力 C：** 四批至少三批 Gc≥5.0，且最后批 Gc≥5.0。若四批 Gc≤.5，记 C_low；其余记 C_mixed，不强行二分为已证/已否。
- **前期集中 D：** q=3 的 |B|<.10 且 E<.20，而 q=0,1 至少一组 E≥.20；或足量状态组 E≥.20 而对应其余足量组 E<.10。第二条件只作分布集中标志，不证明 terrain/动作的因果作用。
- **完整性 I：** 来源/预算/finite/hash/参数通路/头 bias 解析校验全部通过。I 失败优先于所有科学门；中断或不足四批只报告 partial，不形成 U/O/C 的完整阶段判定。

结果与后续候选对应关系：

| 结果 | 可支持的下一步 | 本次不能据此做出的结论 |
|---|---|---|
| I 失败 | 修数据身份、诊断边界或来源；保存失败，不训练 | 不能归因训练优化器或机器人 |
| U/O 与 C 同时成立 | 优先提出一次“actor/critic 分组分别 clip=.5”或价值损失尺度的单因素机制试验合同；两者只能选一个 | 不能说历史 critic 已挤压 actor；不能声称降低 vf_coef 一定改善欠拟合 |
| U/O 成立、C_low | 先有限离线 critic 拟合/校准可学习性候选；若再训练，单独检验 value 优化因素 | 不能凭当前残差直接增网络、改奖励或加总训练步数 |
| M 或 D 成立 | 保留现奖励/控制器，针对训练时间与状态组设计下一诊断 | 不能全局缩放 reward 或把地形换掉当作原因已证 |
| |B|<.10、E<.20 且 C_low | 当前代理没有支持“广泛价值尺度+critic 强裁剪”主因，下一优先仍是动作/更新幅度机制 | 不能称 critic 正确，也不能免除最终 on-policy 评估 |
| 其他混合结果 | 如实报告未区分，保留机制不确定性 | 不能挑最好一批决定配方 |

若后续选择分组 clip，它只改变裁剪耦合，保持网络、reward、vf_coef、actor/critic 学习率、课程、观测/动作、控制器及 target_kl 原样，并真实记录 actor/value preclip 与 postclip 及 optimizer 次数。若选择 value-loss 缩放，则不能同时加 target_kl 或分组 clip。本次诊断不产生最优系数，不能凭四次 backward拍定学习率/系数。任何不超过 32,768 控制的学习仍需单独合同；当前无训练执行、无新模型保存、无物理资格外推。

## 8. 执行闭账、失败和时间限制

纯数据先闭账。root 在 load 前创建唯一 reservation，标记该阶段预算已消费；阶段目录必须新建、不可覆盖。模型 worker 的软期限为启动后 120 s，独立 host 硬期限 150 s，均包括 import/load/计算/结果封存；在软期限或收到停止时只封存已完成批次，不开始新批。单独 watchdog 不调用模型。准备阶段最多 180 s；超过即 data failure，不自动启动模型。写盘失败/硬超时保留第一异常及 partial 文件，不用重试刷新预算。

所有计数在入口尝试前递增并持久记录阶段边界，分别记 attempted/completed/failed：load、value API batch、value rows（含失败 batch 全部预留行）、backward；再分别记录 actor、optimizer/learn/train/save/physics 禁止入口的 attempted 数。成功阶段应为 load=1、value batches=5、value rows=1024、backward=4、禁止项=0。异常必须保存首异常、耗时、已用额度、最后成功阶段、cleanup 状态；失败调用不退还额度，未消费额度不许可自动续跑。

外层 finally 在没有新模型调用的前提下重算 policy named parameter/buffer hash、optimizer state hash、输入文件 hash；必须与加载后/准备时匹配。`.grad`、eval 状态不是权重，单独记清理但不混入权重变化结论。最终回执包括所有预算、实际值、退出状态、finite 检查、参数 hash、冻结源 77 项前后比较、输出清单及 SHA256。不能把 partial 解释为完整通过。

输出最少为 selection/data 与准备回执、reservation、model_call_ledger、512-row value 数值、四批梯度 JSON、worker/host receipt、输入/输出 hash 清单、diagnosis_14.md、Astra 最终审阅。测试只允许 root/Sol 在真实模型阶段前完成针对采样、边界和数学的必要纯测试；本方案作者不运行测试。结果生成后不以重复真实 model 运行来“复核”，独立审阅只读已保存输出重算纯数学。
