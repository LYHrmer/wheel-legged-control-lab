# continuation16：隔离 reader 后的独立有界评估尝试

2026-09-30，Astra。用户已授权诊断后的一项完整 RL 推进及必要的常规集成修复、验证与记录。两臂训练已经完成；continuation15 的物理评估尚未完成。本合同允许在修复确定性的进程边界缺陷后，对这两个已冻结 final 完成一次新的有界评估尝试。**这是新增 reservation，不能记作 continuation15 原额度内的无成本重试。本文件是设计合同；新实现、唯一 import-boundary 检验和精确源码 GO 须在首次 cold construction 前完成。**

## 1. 原失败与新增授权范围

continuation15/eval_01 已失败关闭：eval15 在已导入 MuJoCo/SB3 的 live worker 内导入独立 read_eval15，继承的 verify_course_e_08_03 导入保护抛出 `physics/policy already imported into E verifier`。保护本身正确，缺陷是调用者把 offline reader 放入 live 进程；15 的静态 GO 和 synthetic 检验漏掉了这一跨进程约束。不得删除保护、让 verifier 接受污染进程或把失败归为机器人任务失败。

实际 worker 5.263475785 s、host 11.967751863 s；PPO load、torch.load、actor inference、control、normal native 均 0；cold compiler native attempted/returned=2，另有 2,573 次已记账的静态 mj_forward，不能抹去构造开销。C 指针关闭，source postcheck 完整、changed_sources=[]、cleanup=[]、无存活子进程。原 eval reservation 永久消费且不重试；15 的 18 个 Python 源、GO、plan、模型及全部记录保持原样。

新增尝试是完成已经授权的配对验证所需的集成修复。没有看过任一新 seed 的 floor/task 物理成绩，不选择模型、不换 seed、不继续训练；因而不是依据物理结果扩大试验。工程上这是修复后的再次评估尝试，必须以 continuation16/eval_01、独立 reservation/plan/GO 和新增预算明确呈现，不能仅改目录后声称原 15 已成功。

## 2. 已确定的唯一输入

两模型只读复用，禁止重新训练、改权重、改 Adam 或挑选其它 checkpoint：

| actor | final ZIP SHA256 | manifest SHA256 |
|---|---|---|
| global_continue | db324c8ba2c7a4821352f1d564b47d4fcffb97784d751e9ea20b23d525674d2b | 3bb5b675f1edb0d7ec33bc62cc23e94ecee8682384b67491d60fdfb503c16ea6 |
| grouped_continue | 1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb | bf912efcb7c8aa938b31c5160b3cec0c00891946fdfd00d0039cdd9395e326b3 |

模型分别位于 continuation15/{global_01,grouped_01}/final_checkpoint。metadata SHA256 分别为 e1570201327401fa3867563a27b54504b4832ac5f4548ed7af6e949ffdf31566、cce486dc3f0ae9f0db61babc196cb48b692644659523cab5a7aecbc9fcadeb6a。

15 plan SHA256=9552666e47010f3842d690bda6faa480fa6e8e3fc2ca2739a89755009856d248；training_readback_15.json SHA256=a01d90dec9ef44c640f02a46666db80f11e6e61cc53d553a7c8198f8b87dd23b，engineering_passed=true，首 1,024 controls 的 52 numeric 与 6 Gaussian 数组逐位匹配。15 eval worker receipt SHA256=f19e5b4d324d9b4b763486be3434f9014677974ae0793c3dbb53d12b281e1414；host receipt SHA256=040d3d4c6d8ba42b8386bcd9fe70ac0811f43eae142fae351789fa7bb5fe979a。上述身份和所有新源、77 旧冻结源、实际依赖及 C15 科学合同均纳入 C16 前后同集 hash。

## 3. E16 唯一预算及累计账

E16 **training/learn/train/backward/optimizer/model.save=0**。一个新的 live worker、一次 cold owner，compiler native≤2。normal controls≤30,600、normal native≤153,000，每 control 固定 5 normal native。两个 new final 各 strict load 一次：PPO load≤2、torch.load≤6、确定性 probe 各32 rows，共2 batch/64 rows。常规 actor predict≤20,800 次/rows（两 floor≤1,200，两策略六任务≤19,600）；zero 任务≤9,800 controls、0策略前向。E16 总 actor rows≤20,864，critic forward/value-only=0。记录 attempted/returned、行数、phase 和构造账，失败仍消费额度。

| 账口径 | controls | normal native | compiler native | PPO load 上界 |
|---|---:|---:|---:|---:|
| C15 原已消费 reservation | 63,368 | 316,840 | 6 | 6 |
| C16 新 reservation | 30,600 | 153,000 | 2 | 2 |
| **累计 reservation** | **93,968** | **469,840** | **8** | **8** |
| C16 开始前实际完成 | 32,768 | 163,840 | 6 | 4 |
| 若 C16 完整执行，累计实际 | 63,368 | 316,840 | 8 | 6 |

累计预留 actor rows≤205,760；若 E16 完整执行，累计实际 actor rows≤184,896。累计 critic rows仍≤163,904，backward/optimizer 仍为训练已完成的各512。未执行的15 eval额度不退款、不转入E16；任何门失败时未执行的E16场次同样废止，不补步、补seed、补模型或再开第三个尝试。

## 4. 科学 recipe 和判别门不变

完整沿用 C15/astra_contract_15.md §§5–6、冻结 recipes15/score15/readback15 的科学定义。floor seed=151090，case=floor_0p4_600：0–174 settle、175–424 raw vx=.4、425–599 stop；每 final≤600 controls。固定 last-drive [325,425) 均速误差≤.2、vx RMS≤.25、yaw RMS≤.15；完整600/3000、无warning/非wheel/地图越界、10° roll/pitch、compiled-terrain clearance≥.28，且记录有效。floor 全链独立核验，不以短停止窗取得完整制动资格。

固定新任务 seeds 151101..151106 分别对应 flat_0p6、flat_1p6、flat_1p2_yaw、bumps_0p4、rough_0p35、ramp_0p45_complete，数值脚本、spawn、窗口与 C15 相同；这些 seed 的物理控制从未在失败15 eval执行过。按原顺序先两个 floor，再各 task 内 zero→global→grouped；actor 初始状态逐位配对。某 final floor 不过则跳过其正式场次，两者均不过则结束。合法任务终止可按原合同继续；软件、来源、模型、归档或 reader 失败则停止。

grouped/global 的完整任务、安全/制动、≥4共同 task-pass 的 drive SSE≤.85、≥4个别≤1.02、四个 hold≤1.02、flat1.6 hold vx RMS≤.90、六对完整 drive 成本≤1.20及逐任务成本报告全部不变。零分母和缺失不制造改善。分别报告相对zero原贡献门。不得事后降低门，不能把已看到的 grouped 首次 KL=.6323 或训练梯度改善代替物理成功；没有旧 final 的本次物理对照，不宣称超过旧 final。

## 5. 最小、安全的进程修复

live worker 不导入 read_eval15 或其任何 inherited offline verifier。它只调用一个 stdlib bridge：floor 归档已原子提交后，以新的独立 Python 进程执行冻结 read_eval15.read_floor。输入只有绝对 archive 路径和冻结身份；输出独占写入带 actor/case/checkpoint/input identity 的 JSON/receipt。parent 核对退出码、输出身份和对应 floor 后再决定是否运行正式场次，不能自己重复执行 reader 以绕开失败。

child 清除 LD_PRELOAD 和 LD_LIBRARY_PATH，使用明确且冻结的 Python/PYTHONPATH；禁用 MuJoCo、Torch、SB3、Gym/Gymnasium、glfw、engine_binding 等物理/策略导入。复用原 NoPhysics 保护，必要的额外防护只能收紧；child 前后检查禁止模块未出现，/proc/self/maps 无 MuJoCo/engine DSO。child 不接触 checkpoint 反序列化，0 model load/forward、0 engine construction/physics；仅从已保存 arrays/contact records 计算。

最多两个正式 floor reader child，每个总时限≤60 s，包含最多5 s kill/reap；两者都包含在同一个 E16 host/worker 时间上界中。禁止 detached child；继承可回收的 owned process group，逐个记录 spawn attempted/returned、pid、argv、环境、输入/输出 hash、耗时、退出及清理。超时或输出失效立即终止本 eval worker，不改为跳过资格门，也不重启 child。若使用 RTK 外层，必须回收实际 Python reader，不能只杀 RTK。

最终独立 readback 也在 clean process 中执行。可以复用冻结 C15 科学 reader 算术，但新 stage/plan/run/contract 身份必须可核验；若保留15 wire-format schema以少改代码，必须另有显式 continuation_stage=16 和 C16 run_id/plan/contract 绑定，且最终报告不可称失败15 eval成功。新读回仅解析保存数据，不增 model/physics。所有 C15 文件保持冻结，修复仅写新 C16 文件。

## 6. 唯一针对性验证与执行 GO

只需一个选定的零模型 import-boundary 检验覆盖这次真实缺陷，root 执行且先于 cold construction：测试 parent 用假的 sys.modules 条目模拟已导入 MuJoCo/Torch/SB3（不加载真实模型或引擎）；真实 live bridge/eval模块导入不得拉入 offline reader；通过实际 bridge 的 child preflight 在 clean process 成功导入冻结 read_eval15/read_floor，同时核对 child 禁止模块和引擎映射均不存在。该检验≤30 s（另最多5 s回收），0项目checkpoint load、0 actor、0 cold construction、0物理；单次结果独立落盘，不重跑旧22项。AST/必要lint为静态检查，不扩大科学实验。

检验若失败，修复代码后必须保留失败记录、静态解释差异，不能借检验触发真实构造。首次真实调用前由 Astra 集中审阅 bridge/worker/host/source与调用预算，再将新源码、该检验结果及本合同写入唯一精确 hash GO。root 拥有实际命令与执行，Astra 不执行测试、模型或物理。

## 7. 时间与最终闭账

E16 soft=1,800 s、close=2,040 s、host hard=2,100 s，自host开始计入import、sourcehash、构造、模型load、reader child、正常任务及归档；hard包含source posthash，额外最多5 s仅kill/reap。C15+C16累计预留host上限5,700 s，另各已预留host回收至多5 s，不能把旧失败的未用时间当新额度。

继续使用 archive13 事务、有限值检查、完整contactForce/control/native链、C/Python/guard/model实际入口账、mapped ELF和前后同集hash。首异常与cleanup分别保存，父/子进程全部退出，未用额度失效。E16失败即如实关闭，不在本合同内再次尝试；两模型保持可归档但未取得额外资格。最终交付必须区分：两臂训练完成、15评估软件失败、16实际评估结果、裁剪机制、PPO更新幅度和物理晋级，不以计划或运行启动代替结果。
