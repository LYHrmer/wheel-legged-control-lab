# C22 训练、有限几何资格与 A1/reset 复用合同

该前置步骤只读保存资料并执行纯数学。不得构造机器人模型、载入策略、调用物理、额外reset、重跑旧轨迹或旧完整reader。来源必须绑定C20发布状态、原77及继承输入、公共C15父模型、A1有效reader/host/final。

## 唯一训练与公共优化

只训练B一个stage1，共32768controls、32个1024 rollout；batch256、max_epochs4、evaluate_actions/optimizer各≤512。从C15 grouped原final ZIP与Adam开始（SHA1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb），不重置actor/Adam，不加载失败B。仅重置本阶段计数；PPO seed201001、command201002、measurement201003与A1起点一致。shared reward、128×128 Tanh、lr.0003、gamma.99、GAE.95、clip.2、entropy.001、vf.5、actor/critic分别L2 clip.5保持C20。

共同target_kl=.03；实际batch KL>.045时该批只evaluate不backward/optimizer，并结束当次train；KL非有限或>.30、非有限loss/参数/梯度/Adam delta，或完整train零optimizer均中止阶段。实际entered epochs/evaluated/rejected/optimizer分别记账，不要求与A1的351步相同；第一个/最后一个实际optimizer完整梯度样本及其余范数/hash/Adam delta沿用C20。

训练env仍train，原生归档guard使用heldout/full_contact只读采样同一5native，不追加积分。保留全部C18 pre/post、PI/z/servo/provider/动作门、numeric/Gaussian、九奖励独立复算、真实力矩与5T contact/geometry链。新的reader只适配合同、来源、课程和有界reset门，不能退回first1024摘要代替全tick验证。writer原子提交、失败checkpoint隔离、host独占不退款、软停止与真实调用计数不变。

每worker上限见spec：PPO load2/torch6、learn1/train32、forward32768、evaluate512×256、backward/optimizer512、group clip1024、value-only64、predict probes3×32、save attempted2且success final1；actor≤163936、critic≤163904。唯一正式final只能来自完整预算末端。failure checkpoint不可评估；不追加训练段、诊断模型调用或checkpoint择优。

覆盖规则沿C20：每postwarmup槽≥3个完整1000-control且≥200有效servo/action门ticks；B指定yaw cycle0/2与bumps cycle0/1/2完整，endpoint正负各≥100、每指定bumps≥25真实有效正轮载荷控制间隔，rough/ramp有效载荷各≥1。覆盖不足与训练归档无效分开记录；可对有效final完成已列固定评估，但不能补预算或宣称覆盖收益。

## 有限课程与几何

完整有限域仍为A/B各source0..79、原七、开发四场、最终四场、floor，共176条/184800名义间隔。C21已完成一次全域预检：160训练+12评估通过、4新yaw失败。C22逐项复用已过172条的原始raw/hash、全部名义数组和条件性资格；只计算四条修订yaw各1600，共6400新纯servo、600s。不得将旧失败行标pass，也不得重算已通过的178400间隔。C21旧184800与C22新6400分别记账，累计191200。A始终只作来源/命令复用，绝不重训；最终策略分数未被读取。

实际servo/CAPS保持C20；逐段积分含释放后的servo衰减，不用raw净yaw或少数路径点替代。所有路径须满足中心线地图门；flat另按原compiled92世界geom、全部robot collision geom的径向包络、原11/6m边界和全部非floor盒逐段判交。评估horizon可为600/1600/1800，C22纯几何helper只泛化N+1形状，不能漏掉尾段或改变碰撞数学。非flat的bumps/rough/ramp本来要接触对应地形，不能套“避开所有盒”的flat资格；须证明同一reset姿态/几何与包络地图界、terrain/spawn/命令身份，真实载荷仍由运行5T核验。

未来source的geometry不能凭空记为实际。以C20 A1 flat episode0实际geometry和各地形episode0/11/12/13初始五数组、完整reset状态为基础；以B18实际geometry及已核唯一候选作为交叉证据。证明相同固定模型、nominal关节姿态、朝向、spawn与tick0零指令，不因后续命令/episode标签/seed改变reset几何。每地形使用自己的初态；禁止把flat的完整observation直接充作其他地形的observation。若仅对geometry做固定平移，必须由保存qpos及冻结reset源码证明同一姿态与该位移，并标为推导，不能修改world terrain坐标。

必要源码链包括：C18 controller18/residual18；course_impl08的full_drive_env_08/controller_08/loop_08/servo_08/observation_08/course_plant_08；upright11/world_upright_course_11；仓库d1 control_loop/control_context/state_provider/state_estimation/wheel_leg_controller/model/actuator_channel；实际模型/terrain描述与依赖身份。使用实际完整文件名，在回执列每条依据与SHA。oracle reset丢弃seed及env/loop/actuator/controller的reset→prepare语义须由源证明；无证明则不通过。

回执将未观测future reset写成`source_derived_expected_reset_not_observed_future_reset`，区分实际参考、变换依据和预期值。C22合并资格必须由旧172条有效来源与新4条全部通过构成；缺失、漂移或新失败均不启动B。原first80只是有限域；runtime source>=80在env.reset前停止，不抽替代episode。到预算末端既有零控制postbudget reset复用已资格schedule，不选新source。

每次真实reset后、该episode任何动作前，记录实际geometry及initial五数组、完整控制reset并与对应已封存预期核对；model/data地址允许不同，compiled拓扑/geom id/body/rbound/world rows与状态须一致。几何数值仅容许绝对1e-12舍入差；五数组须shape/dtype/bytes一致。读取当前已存在measurement geometry不增加forward或native步骤，不增加reset次数；同时保留原runtime flat预检。任何漂移保留失败并停止。

## A1 复用

唯一A为C20/train_A_1 final，SHA1a630229fe9b44abe75abe4e47c98be50a55a043f0163f20f57b0b97640904e5；冻结其session/worker/host、独立reader+host输出身份、final manifest/metadata、parentage与必要原始配对文件。有效reader可复用，不重复32768步或旧全量reader。

证明共同C15父ZIP/Adam、201001 PPO seed、201002 command seed、201003 measurement映射、warmup、网络/奖励/优化/C18控制/99D16D、5native、地形/出生点/episode规则未变。新finite-source guard只是已观察A轨迹0..32从未触达的外层拒绝条件；它不能被用来声称更广泛分布等价。

新B完整读回后，与旧A初始qpos/qvel/ctrl/qacc_warmstart/observation、完整PI/z/stop/servo/provider/previous-action reset状态及首1024所有numeric/Gaussian数组逐位配对。跨臂命令分歧后不要求轨迹一致。B实际optimizer数允许不同；共同规则与每臂实际计数必须保存。复用不能仅信一个bool，更不能载入失败B接着训练。

## C21 条件资格复用边界

引用 `spec22.finite_geometry.prior_preflight_path` 的失败总回执及全部输入，保留其 `passed=false`。只复用其中160条train和12条已通过eval；新四条yaw替换的是当前合同的命令版本，不删除历史失败。复用要求共同servo/CAPS、几何门、reset语义、同一compiled模板与原始initial/control状态不变，新旧172条raw SHA逐条一致。geometry helper若因模块名版本化，应以源码/AST证明数学等价；不能借重命名改变判据。新C22回执同时绑定旧失败总回执、新四条完整路径及所有源SHA，才可声称当前有限集资格通过。

A1学习复用仍比较C20全部公共学习字段；允许的新增差异只包括C20→C21已核B cycle1修复、0..79外层拒绝条件、C21→C22四条eval yaw窗口，以及明确的模块/合同身份。旧七/floor、训练奖励/网络/优化和B训练表不得随评估修复变化。旧A reuse回执可作来源证明，但新C22必须核这些列举差异，不能仅复制旧passed。
