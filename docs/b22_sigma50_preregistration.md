# 50：初始探索尺度的三配对 seed 诊断

用户已批准《优化 50》中的控制与训练分支。本文件固定训练分支，执行资格仍由独立来源登记、完整准入和逐阶段回执确定；本文不是已运行声明。

唯一变量为完整 B22 warm-start 后的初始 Gaussian 标准差：baseline 保持原值；candidate 对全部 16 维 float32 `log_std` 加一次 float32 `ln(0.5)`，随后按原 PPO 正常学习。不缩放环境动作、不修改 likelihood、不冻结 log_std，也不持续钳制到对照的半值。父模型为 `runtime/d1_b22/checkpoint/final_model.zip`，SHA `7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691`；完整 policy、critic 和 Adam 先由原 loader 验证。

固定 α=8、β=0、FF=3.9、`yaw_emphasis`、CPU 单环境、1024 rollout、batch256、epochs≤4、32768 controls/臂、原 KL 和分组裁剪。训练种子为 201051、201061、201071；每个 seed 依次 baseline、candidate。command_seed=201002、measurement_seed=201003、source_episode_offset=0。均值网络、critic、Adam moments/step/lr 和 Python/NumPy/Torch CPU RNG 在初始化干预前后必须逐位不变。

## 来源与初始化证据

- 旧 v27 清单与全部冻结源保持原样。独立来源审计为 `results/b22_sigma50_source_audit_20261024_01/report.json`；全部4072项均登记，唯一允许差异是已批准的 libXtst `0c81be070f7857bdcab6bd528161d1049c374f1237aabf76d92e097b2d8ba16a` → `4198b32dc9fba7dfc0ee2bb1b6bc85067c89adfc68d0279059ad82c207cc3c35`。任何其他漂移阻断运行，不能删除条目或自动接受当前值。
- 新登记绑定当前 HEAD、所有新代码/测试/本文、完整父 checkpoint/probes、旧适配器与冻结算法、来源审计及其逐项 ledger。执行前、每阶段与结束均核对来源；不调用任何旧一次性 parent，不恢复48。
- 47 的父/worker源码仅在新入口的私有命名空间中派生；指定 AST 位点、精确旧表达式、出现次数与逆向复原均必须匹配。原执行 guard 被禁用，所有新路径和新合同身份显式记录。
- 原 warm-start 的审计快照保留，新增真实干预后审计。`initial_sigma50/proof.json` 与 `tensors.npz` 保存干预前后完整 policy/Adam/actor/log_std/RNG 的 typed tree 和张量，保留 Torch tensor 的 `T` 哈希标记。独立 reader 从实际数据重算父与新初态 hashes，验证只有预定16值变化。辅助 `receipt.json` 记录加载审计、新审计及额外模型/物理计数0。
- 逐拍命令由原独立 reader 校验；另记录每个 source_episode_index 的完整命令 SHA、reset/provider measurement seed、实际 tick 数和终止原因。两臂比较共同 episode/tick 的外源规则，不能要求干预后动作、状态或首1024拍相同，不能称为三个独立环境噪声重复。

## 预算与不可复用规则

六臂开始前必须完成独立模型准入，计划为 `docs/experiments50/training_admission50_preregistered.json`，结果只能写入全新 `results/b22_sigma50_model_admission_20261024_01`。baseline_s51/candidate_s51 各经一次与正式 worker 相同的完整 B22 loader、私有 learning21 seed 绑定及初始干预 hook；Gym stub 只提供原 99D/16D float32 空间，reset/step 均在调用前拒绝。硬预算共 PPO.load2、torch.load6、确定性 predict2次×32行；learn/train/backward/Adam.step/save、MuJoCo构造与native步均0。父探针逐位相同后销毁模型，模型进程退出后才由无Torch的新进程读回两份初态证明。准入不消耗六个训练 attempt，不能恢复为正式训练。父入口必须绑定这份 COMPLETE 回执、独立读回及全部来源身份；pytest 通过不能替代真实准入。

准入单元为 `control50-wheel-model-admission-20261024-01.service`，资源门与训练一致；总时限900秒、外层1200秒，输出≤512MiB。完整来源复核先于载入；结束复核在模型进程退出后执行，避免模型常驻与来源页缓存同时占用内存。

六次训练共196608 controls、983040 normal-native，环境构造最多12次 compiler-native；bootstrap 0模型构造、0推理、0native。新增初始化证明只遍历已加载张量，额外 model load、forward、optimizer、environment step均0。每臂仍原调用账本：load2、torch_load6、save1、learn1、train32、forward32768、predict3；实际 evaluate_actions/backward≤512，predict_values遵守原rollout/截断计数。新的审计对象不是新模型。

每个完整模型导出后评估 zero/raw50/mirror50 的原六任务，18场/29400 controls；采用完整43链时额外 raw/mirror 两个600拍 floor，总30600 controls/模型，六模型183600 controls/918000 normal-native，构造最多12 compiler-native。floor全保留、不筛选模型。导出真实调用按独立评测入口的数值协议另记，不能混入训练账本。所有训练、导出、评测、读回串行。

输出固定 `results/b22_sigma50_20261024_01`，release 固定 `results/b22_sigma50_plan_20261024_01.json`；目录必须全新。失败/partial保留，不覆盖、不续跑、不补seed、不用旧模型填空；后续未启动阶段明确 NOT_RUN。任何不完整臂阻止主比较，不因某个中间 checkpoint 看起来更好而评测它。

训练父进程及所有子进程处于 `control50-wheel-train-20261024-01.service`：Nice≥10、CPU单线程、MemoryHigh2GiB/MemoryMax3GiB/swap0；4GiB有效内存持续10秒准入，运行保留2GiB。父训练/读回总上限10800秒，外层监督12000秒；每臂worker1200秒、reader600秒。单臂归档≤3GiB，总归档≤20GiB，启动可用磁盘≥22GiB、运行保留2GiB。不停止用户其他程序，不改变资源门。

## 事前判定

主actor固定mirror50；raw50为次要结果，全部报告。主R为原六任务固定drive窗口的合池归一化SSE_policy/SSE_zero。两臂分别报告三个R、均值、极差(max−min)，同时报告各配对差。只有全部六次训练、导出和主评测完整可比、分母有效，且 `range_candidate < range_baseline`、`mean_candidate <= mean_baseline`，才记开发诊断方向通过。n=3极差不是置信区间或显著性证据；旧六任务及已解封案例均是开发资料，不产生普遍性能或部署升级结论，不替换默认模型。

性能资格独立按原门逐项报告。六任务为 flat_0p6、flat_1p6、flat_1p2_yaw、bumps_0p4、rough_0p35、ramp_0p45_complete，原seed181201–181206、完整时域与速度/姿态/停车/地形阈值保持。共同前提为有效记录、policy完整且安全、无zero-pass/policy-fail回退；六对共同drive时域归一化力矩平方成本≤zero的1.20。A分支至少一次真实zero-fail→policy-pass且无新增失败；B分支至少四对共同任务通过、这些对合池SSE≤0.85、至少四对逐对SSE≤1.02。不加epsilon、不删除失败对，成本不称为能耗。诊断方向通过不能覆盖任何资格失败，也不能选择最佳seed为整组授予资格。

预计训练/读回约95分钟、导出1–2分钟、评测/读回约88分钟，总运行约3–4小时，开发与资源等待另计。上述为预算估计，实际值以完整回执为准。
