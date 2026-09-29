# 阶段 15–16 最终只读审阅

2026-09-30，Astra。审阅对象为已经冻结、执行并完成独立读回的两臂续训及阶段 16 评估。本次终审只读取既有源码说明、合同、JSON 记录并计算文件哈希；没有加载模型、执行模型计算、运行测试、优化器或物理，也没有修改冻结源。

**裁决：工程交付成立，实际裁剪机制证据成立，完整任务晋级失败，整体 RL 贡献未通过。** 分组臂相对同预算全局臂的若干跟踪改善可以报告；不得将其改写成通过六项任务、优于零残差控制器或取得新默认 GUI 资格。唯一下一方向为保存数据上的 yaw 退化定位诊断，尚未执行，未预占新增预算。

## 1. 工程与执行账

两臂各完成 16,384 新训练 controls、16 rollout、64 epoch、256 联合 backward 和 256 Adam step；两臂来自同一父 ZIP、policy 和 Adam 状态，未重新初始化 actor 或 optimizer。种子、课程、学习率、网络、loss 和控制器相同，首个更新前 1,024 controls 的 52 个 numeric 数组及 6 个 Gaussian 数组逐字节一致，无差异。首个 minibatch 裁剪前 actor/critic 梯度哈希也一致。这满足本次配对机制比较的来源条件；两条训练轨迹在第一次更新后允许分离。

阶段 15 原评估仍是一次失败记录：live worker 导入离线核验器，触发其正确的“physics/policy already imported into E verifier”保护，发生在评估模型加载和控制调用之前。实际消耗 2 compiler native、0 control、0 normal native、0 policy load；另存 2,573 次静态 mj_forward。该集成错误是原静态 GO 和 synthetic 检查的漏项，不能归咎于离线保护。原额度不退款，原失败和冻结文件均保留。

阶段 16 独立合同将 floor 读回放入干净、受控的子进程，保留输入身份与导入保护。它执行既定的两份训练产物和此前尚未物理执行的新 seeds；没有重新训练、挑 checkpoint 或根据评估结果改门。两场 floor 和 18 场正式任务全部跑满，独立离线读回 exit 0，模型调用与物理调用均为 0。77 份旧冻结源全部匹配，执行来源前后哈希一致，cleanup_errors 为空，全部所属进程已回收。

| 实际执行 | controls | normal native | compiler native |
|---|---:|---:|---:|
| global 续训 | 16,384 | 81,920 | 2 |
| grouped 续训 | 16,384 | 81,920 | 2 |
| 原阶段 15 失败评估 | 0 | 0 | 2 |
| 阶段 16 完整评估 | 30,600 | 153,000 | 2 |
| **合计** | **63,368** | **316,840** | **8** |

累计 reservation 为 93,968 controls / 469,840 normal native + 8 compiler，其中原评估未执行的 30,600 controls 预留已消费并废止，不计成实际步数。本轮合计 6 PPO load、18 torch.load、512 backward、512 optimizer step、2 learn、2 成功 save；actor rows=184,896，critic rows=163,904。两训练 host 用时 153.991/150.526 s，失败评估 11.968 s，阶段 16 host 494.433 s，均在各自冻结上界内。最终只读评分另耗时 344.914 s，单独记账。

两份新策略的 ZIP 身份为：global `db324c8ba2c7a4821352f1d564b47d4fcffb97784d751e9ea20b23d525674d2b`；grouped `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`。阶段 15 plan SHA256=`9552666e47010f3842d690bda6faa480fa6e8e3fc2ca2739a89755009856d248`；阶段 16 plan SHA256=`24d16b5362a5d95d0e12e88026b452890f7f72e4a33f4e0596969324e5ce1d00`。

## 2. 裁剪机制得到什么证据

唯一训练因素是一个全局 L2=0.5 裁剪组，改成 actor 与 critic 各 L2=0.5 的两个组。二者仍使用一次联合 backward 与同一个 Adam step；不存在可训练共享参数。grouped 联合梯度范数允许超过 0.5，是预先定义的因素，不是隐含预算放宽。

首个共同 minibatch 的 actor/critic 裁剪前范数为 4.694812 / 640.141084。global 裁剪后 actor 范数为 0.003666915，grouped 为约 0.5。这直接证明该共同更新中较大的 critic 梯度经全局范数上限压低了 actor 梯度。不能把此证据倒推成阶段 14 保存窗口中未测量的历史 actor 竞争。

| 256 次实际更新的统计 | global | grouped |
|---|---:|---:|
| actor 实际梯度保留系数，中位数 | 0.000795092 | 0.174494408 |
| actor Adam 参数变化 L2，中位数 | 0.002935263 | 0.025494743 |
| critic Adam 参数变化 L2，中位数 | 0.003853471 | 0.003878939 |
| 首个 rollout 更新的 approx KL | 0.0483523 | 0.6323433 |
| 首个 rollout 更新的 clip fraction | 0.3405762 | 0.8164063 |

实际 actor Adam 变化中位数比约 8.69；它不是梯度保留系数之比。首次 actor Adam 变化约为 0.002337 对 0.046798，提示改变裁剪尺度后出现较大的初始更新；Adam 历史状态是否参与放大是待检验解释，没有清零 moments 的反事实实验。后续两臂经历不同参数与数据轨迹，跨臂中位数不能解释为每个相同 minibatch 的恒定因果倍率。

较大的首次 KL 说明要警惕策略位移，但本合同 target_kl=None，没有违反预注册早停门。它既不是控制收益，也不能证明本次最终 yaw 退化的原因：两臂都出现该任务退化，global 的首次 KL 较小；没有更新时点干预、动作通道消融或中间 checkpoint 物理比较。不得据此写“分组首次 KL 导致 yaw 失败”。

两臂解释方差仍近零或负，末次约 -0.003394 / -0.003135，未证明 critic 已拟合好。value loss 的跨更新变化混合了目标与访问状态变化，不能当作固定数据上的拟合改善。阶段 14 的 64-step 保存行为窗口自举残差也不能提升为当前 on-policy 价值真值。

## 3. 完整任务成绩

两场低速 floor 通过，只提供继续完整评估的资格。18 场正式任务均完整、安全与最终停止通过；“跑满”“保持安全”“最终停稳”与任务速度/yaw 门分别计分。

| 任务 | zero | global | grouped |
|---|---|---|---|
| flat 0.6 | 通过 | 通过 | 通过 |
| flat 1.6 | 通过 | 通过 | 通过 |
| flat 1.2 + yaw | 通过 | yaw RMS 失败 | yaw RMS 失败 |
| bumps 0.4 | 通过 | 通过 | 通过 |
| rough 0.35 | 通过 | 通过 | 通过 |
| ramp 0.45 complete | 速度失败 | 速度失败 | 速度失败 |
| **通过数** | **5/6** | **4/6** | **4/6** |

转向 hold yaw RMS 为 zero 0.11377548、global 0.12466550、grouped 0.13036747 rad/s，绝对门为 0.12。两学习策略在本次配对 seed 上失去了 zero 已通过的任务资格。grouped 相对 global 的联合 hold SSE 比仍为 1.017133，符合 ≤1.02 的相对门；这不覆盖 yaw 分量的绝对失败，不能用 vx 的改善将其抵消。

坡道三者的真实轮载荷与整轮几何清除条件都通过，但 hold COM 均速分别为 0.49719613 / 0.51663338 / 0.51007042 m/s，目标 0.45±0.04；对应 RMS 为 0.05025549 / 0.06915877 / 0.06236857 m/s，门为 0.05。因此几何跨越完成，完整坡道跟踪任务失败。zero 同时越过均速和 RMS 门，不能四舍五入为通过，也不能将三者共同问题全部归因于 RL。

平地 1.6 m/s 三者保持段 400/400 样本均超过 1.5 m/s，任务通过。grouped 的 hold vx RMS=0.01404623，相对 global 的 0.03083210 明显下降，但仍略高于 zero 的 0.01240063。grouped/zero 的完整 drive SSE 比=0.88552979，是包含加速等阶段的窗口成绩，不能写成稳态保持超过 zero。释放后的有效 RL 残差归零，最终制动不得作为 RL 的独立贡献。

## 4. 按冻结门裁决

| grouped/global 门 | 结果 | 裁决 |
|---|---:|---|
| 四共同通过任务 pooled drive SSE ≤0.85 | 0.72644454 | 通过 |
| 至少四个逐任务 SSE 比≤1.02 | 4 个 | 通过 |
| 四指定 hold SSE 比≤1.02 | 全部通过 | 通过 |
| flat 1.6 hold vx RMS 比≤0.90 | 0.45557170 | 通过 |
| 六任务完整 drive 力矩平方成本比≤1.20 | 0.95115357 | 通过 |
| grouped 全六项任务通过 | 4/6 | **失败** |
| **完整 promotion** | **false** | **不晋级** |

共同通过集合严格为 flat 0.6、flat 1.6、bumps、rough；上述 pooled 结果不包含失败 yaw/ramp 的 SSE，不得称“六任务跟踪普遍改善”。力矩平方值是归一化成本代理，不是机械能或电耗。此次相对门改善是有效开发证据，绝对门失败仍优先。

相对 zero，两臂整体 RL 贡献均为 false。global/common-pass pooled SSE 比=1.43080422，grouped=1.03939991；后者仍比 zero 高约 3.94%。grouped 只有 flat 1.6 一个逐任务 drive SSE 比满足≤1.02，未达到四项，并且没有任务由失败转为成功，反而发生 yaw 资格退化。六任务成本比 global=1.22512977、grouped=1.16528656；grouped 的成本门通过，但仍比 zero 高约 16.53%，不能补偿任务和跟踪门失败。

结论保持 `qualified_for_default_GUI=false`。旧 final 未在本轮新增物理对照，所以不能声称任何新策略优于旧 final。单对训练 seed、每任务单 seed、有限新课程暴露只能支持这些固定开发场景；不外推可靠性、实机、自由键盘、15 mm 单台阶、侧移、跳跃或自救。历史 GUI 资格按其原模型和原范围保留。

## 5. 唯一下一科学方向：保存数据上的 yaw 退化定位

**主要问题：两种裁剪续训策略为何在本次 yaw 保持段均较 zero 增加误差；保存的实际有效残差及对应训练命令覆盖能否定位一致的时间段和动作通道？** 这是一个尚未执行的诊断方向，不是认定某残差通道已经构成原因。

优先这个问题，是因为两臂共同出现了本来通过的任务退化，直接阻止整体 RL 贡献；它比“grouped 首次 KL 大，所以继续调 actor norm”具有更直接的当前物理证据。坡道 zero 也失败，暂不把坡道基础控制与 RL 转向退化混成一次训练调参。critic 解释方差问题保留为已知事实，本轮不同时启动第二条优化路线。

后续若获独立执行安排，应先冻结既有三方 yaw 记录和两臂训练 schedule/numeric 输入，只进行文件解析与数值汇总：沿原 raw/servo yaw 转换点及预注册 [415,815) hold 窗口对齐，完整报告有符号 yaw 误差、RMS 与各预定子段贡献；核对从策略动作、限幅/映射到实际有效左右驱动及转向通道的现存记录，区分实际执行残差、基础控制与被屏蔽动作；统计相应 vx/yaw 命令区间在已存训练记录中的暴露。若字段不足，明确标为不可判定，不补造力或反事实轨迹。

诊断的可交付判断应是：误差集中于命令转换还是持续保持；两学习臂是否存在同方向的有效动作偏置或饱和；相关工况有无训练覆盖。覆盖不足只能支持“覆盖缺口”假设，覆盖充分会削弱它；时间相关的残差与误差只能定位待检验机制，不能证明动力学因果。三条实际轨迹更新后的状态不同，禁止将它们逐时刻相减解释为同状态动作消融。

**本审阅没有运行上述诊断，没有建立新训练/模型/物理 reservation，新增执行预算为 0。** 后续训练变量应由该诊断结果选择并另立单因素合同；当前不冻结 actor 上限、奖励、课程、critic 拟合或 KL 早停修改，也不自动追加长训练。

## 6. 中文报告抽核与证据绑定

已抽核 `main_route_execution_20260930_15_16.md` 当前稿的 4/6 与 5/6 任务数、yaw/ramp 失败原因、分组相对全局改善、相对 zero 不成立、窗口差异、成本代理含义、原失败记录、reservation/实际区分及能力边界，与冻结合同和原始独立读回一致。未发现阻止发布的科学表述。该结论覆盖下表所列稿件版本；公开打包的完整性与实际上传仍由 root 完成，不能由本文的稿件审阅代替。

下列 SHA256 由此次只读审阅直接读取文件计算，路径相对工作目录 W。JSON 汇总用于便于阅读，核心裁决已对照独立训练/评估读回，未重新评分或执行模型/物理。

| 证据文件 | bytes | SHA256 |
|---|---:|---|
| `continuation15/astra_contract_15.md` | 16929 | `bb39f6b12d144c60c4cbd78a1305c83ddda87af74a72774c21f82bb88ea774c2` |
| `continuation15/go_review_15.md` | 7800 | `b24379a4785964dd9e1d208f0ecb751c320e78efe80962a08a0ac0e468fd9c54` |
| `continuation15/training_readback_15.json` | 19433 | `a01d90dec9ef44c640f02a46666db80f11e6e61cc53d553a7c8198f8b87dd23b` |
| `continuation15/eval_01/worker_receipt.json` | 6464 | `f19e5b4d324d9b4b763486be3434f9014677974ae0793c3dbb53d12b281e1414` |
| `continuation15/eval_01/host_receipt.json` | 349 | `040d3d4c6d8ba42b8386bcd9fe70ac0811f43eae142fae351789fa7bb5fe979a` |
| `continuation16/astra_contract_16.md` | 10227 | `9401e1dbadf5236b3b55b4d7defe0ad814015154ac00ae9320818f1dfebd5225` |
| `continuation16/go_review_16.md` | 5454 | `42d2243c95c203b782a1094c9fac36c6f6eeb5b543717419c1bd751efcc54849` |
| `continuation16/eval_01/worker_receipt.json` | 41496 | `41300d1c14f287eb5fa0b824c5222ada3b26bd5354ff8c92a1a9b854d5d747c3` |
| `continuation16/eval_01/host_receipt.json` | 273 | `11113672c0e559c9dacefcc7cf3ba1436dcef234c19dbcc6b27d6c4d4a3f6c1a` |
| `continuation16/eval_readback_16.json` | 572469 | `fdadeffa7683a19627f759001304ace481084a134b97eec2e866b2bd82646d36` |
| `continuation16/eval_readback_command_receipt_16.json` | 860 | `5ac466c2b96497b1834ddd2e26f445024b829d7f025d51a5d34b39992d055b1e` |
| `continuation16/result_summary_15_16.json` | 19217 | `76348d19762af22730757ecec1d59e62dcd466de2267486d9ed0d8274a7271e5` |
| `continuation16/execution_summary_15_16.json` | 16293 | `0790375ad3677ae766af78be4b2cc7e4e044a7b22f38d02dec87a22b55dd36e5` |
| `continuation16/frozen_source_post16.json` | 221 | `b870e4f7329132c0f47861cbdaf9d99a9be934a27d4dec7a2e75bd886f1f113d` |
| `continuation16/main_route_execution_20260930_15_16.md` | 9812 | `43063d394f7b09315c6f73dd1acfb2c0846f113a7182257a227201b4ee05ad2a` |
