# continuation15 最终静态运行审阅

**GO — 仅批准 astra_contract_15.md 所定的一次 global/grouped 配对实验。**

2026-09-29，Astra。已审阅最终 warm-start、clip、recipe、worker/host、保存、原子归档桥接、完整 heldout recorder、独立训练/评估 reader 及评分路径。root 可将本文件与下列精确源码、依赖、父模型、77 个旧冻结源一起生成唯一 plan_go_15.json，随后按合同执行 global_01、grouped_01，训练独立读回通过后执行有条件 eval_01。无需追加例行许可。本审阅本身执行模型 load/forward/backward、optimizer、训练、物理、测试均为 0；仅静态读取、源文件哈希及写入本审阅。

## 审阅结论

1. **公平 warm-start 成立。** 动态 AuditedPPO 使用继承的 load 路径，不调用归零 actor 的旧 builder。父 ZIP、policy、Adam/actor/log_std 哈希与父 65,536 controls / 256 epochs / Adam step 1,024 均核对；设置新 seed 后再次核对状态。仅重置本阶段计数，常量学习率在 progress=1/.5/0 核对。两臂保持同一初始权重、Adam moments、课程和 seed；first-rollout 独立读回比较完整数值与 Gaussian 数组及全部实际涉及的 episode/reset/schedule。合法提前终止按实际 done/tick 边界处理，不假定 episode 0 必须满 1,000。

2. **唯一算法差异是裁剪分组。** actor 7 tensor（含 log_std）与 critic 6 tensor 完整、独立且恰好覆盖 Adam 参数。global 保留原参数顺序、原 clip 调用及返回值；grouped 分别以 .5 裁剪两组，仍为一次联合 backward、一次 Adam step。新 optimizer hooks 在旧 LearningAudit hooks 后惰性注册，真实已完成 step 先记账，再做参数 delta/归档；失败有独立 attempted/returned。没有额外 actor/value backward。每步真实 pre/post norm/hash、Adam delta，以及预定 32 组完整梯度样本/臂可独立读回。

3. **调用与物理上界闭合。** 两臂各 16,384 controls、256 backward/optimizer，总训练 32,768；评估含两场 floor 和最多 18 正式场，总最多 30,600。全阶段最多 63,368 controls、316,840 normal native；三个 cold owner 合计 6 compiler native 另记。最多 6 PPO loads / 18 torch.load、8×32 probe rows；总 actor rows≤184,896，critic rows≤163,904。counter 在实际 API 入口限制次数和 batch 行数；worker 成功路径再核精确计数，reader 复核最终账。checkpoint15 删除旧 reference_model 重复预测，未引入隐藏第三次保存 probe。一次 learn/臂、16 train/臂、32 value-only rows/臂等上界不扩大；失败不退款、不重试。

4. **边界、来源和失败封存符合合同。** C/Python/guard 的 5T、construction、caller、CCD、指针关闭及禁用入口分开记录。archive13 原子提交桥接保留首异常，不重试已失败 payload；training 的 unbind/事件关闭次异常另记。Torch/NVIDIA mapped ELF 在 load 前、完成及 finally 核对 resolved frozen membership，并拒绝 deleted 映射，允许省去未使用的 repo/.local-deps/nvidia 重复副本。host soft signal 发给 session SHA/PID/process-group 核验后的实际 worker，不提前结束 RTK；独立 close/hard deadline 保留封存时间。实现从 host 启动开始计时，比合同文字中的 worker 起点更严格；750/750/2100 秒包含 source posthash，额外 5 秒仅用于失败 kill/reap，不能令超时结果成功。

5. **物理门和解释边界一致。** floor 从真实保存的 controller/native/contactForce/compiled terrain 链核验，除纯数值 gate 外还核 terrain-aware initial/native/control 最小 clearance≥.28。两个 floor 均失败时不读取未执行的 zero 场次；单臂通过时不给 grouped/global promotion。正式任务使用新 seeds，score15 与旧 scorer 的算术逐行保持一致，仅改 seed、schema 和提示文字。grouped/global promotion 使用预注册 ≥4 共同合格任务 SSE、四个 hold 门、flat1.6 RMS 严格改善及六对完整 drive 的成本门；零分母不制造改善，逐任务成本另报。reader 最终绑定 worker/host/session、实际 warnings、source/进程闭账及 C/Python/model 计数，不能用开场的空 warning 列表自证成功。

目前没有未解决的静态运行阻断项。本 GO 不是物理资格或正收益结论。root 必须依实际 receipts 和独立读回裁定；不完整、失效、undefined 或未过物理门均不得宣称成功，不得因本次结果追加预算。单训练 seed 与短课程不足以证明泛化；没有旧 final 的本次对照，不得宣称相对旧 final 提升。

## 已有验证证据

已读取 pure_tests_receipt_15.json：22 个唯一 pure/synthetic case 通过（learning 7、recipes 4、runtime/archive 3、training reader 5、eval reader 3）；robot checkpoint loads=0、physics controls=0。root 记录 18 个 Python 文件 AST 与 E9/F63/F7/F82 检查通过；不声称全量风格检查通过。该 receipt 明确记录最后的 compiled-terrain clearance 收紧为测试后静态审阅变更；本审阅已检查其只读算术和阈值，不把它写成重新运行过的测试。

## 精确来源绑定

以下路径均相对 continuation15。本表绑定最终 18 个 Python 文件、科学合同及 pure-test receipt。执行前必须与 plan 的冻结输入一致；科学或执行源码若变动，须保存新审阅记录，不覆盖本文件或沿用本次哈希。

| 文件 | bytes | SHA256 |
|---|---:|---|
| checkpoint15.py | 5350 | 5e4fe24479c779d1c9b5766fc9b5480953818ed37d605b31e28879b212ca87f1 |
| eval15.py | 3854 | 9432bff251647a33a5ff3f5dce96536c9116bf73d98a31e51cccda9a1d1808d1 |
| heldout15.py | 14804 | 29859ba57a4f441cda67ae40c78816ba30c900b343f7bb1d9bac77ec3456cd8c |
| host15.py | 13404 | 5cdcb62874872a831fb149769665ea1b2903e3e58fd6a1bb97d31d2328604c1e |
| learning15.py | 22598 | 2b200d1900805f2524a35ce0927e6534c3f12aa388a19ba39a93dce94bfcb51e |
| read_eval15.py | 45913 | 0c4264473f87dfa2afa5e7433bdd6d9ffc4b054b4bebf86e57e761ab49500a7a |
| read_training15.py | 26848 | 40ca71eaea43d927b83222d3069490ea9eec3c1d2496dbf588447972094dd620 |
| readback15.py | 15013 | ce38ed5f94918b7f839cf27e07c60d393be6458eed349c788d99b722f463e05d |
| recipes15.py | 10093 | fbda812b00b5926e027808ffce95c8ee185b1d2ab14e12f937486f5a7c2f6a85 |
| runtime_support15.py | 5516 | baa217fa07518ac9c554187a4616cb816119ab1f154b48349362964636670b7c |
| score15.py | 18933 | 1be3c8407bf63ec6289cc22a74b6a6bbac77b5be50c0c4304f338a2f94a149b9 |
| test_learning15_pure.py | 7188 | 7a2ccfe16720336b0486cbb8687dd15d9937b01e9a9d415dcf41c00697638d65 |
| test_read_eval15_pure.py | 5770 | ba8ff1010bdc2205dc864a4402de14d39afc91f6870153cbe8531553ce117129 |
| test_read_training15_pure.py | 4617 | 7aec39a215111c156aa01ea8ce4e51a5517812a32f12197c5e0e3984ae8444f0 |
| test_recipes15_pure.py | 4907 | c47215d2cd7b04b46e05cb0368fde53c0a1100f79810ec75a6cd5c52fbdea59f |
| test_runtime15_pure.py | 2659 | 91236126c9c27a85201f625cc81277890b8dbd835385f0de517645aab64efddf |
| train15.py | 7996 | 7a43d9e5742c665ad07dfa1622f9ecebb4b4b0b78b597795454a42914dcbb506 |
| worker15.py | 13320 | 7db4a3c71ab1579e6e7dc14c82c757ff8116e4255867069258017ce3d23f5857 |
| astra_contract_15.md | 16929 | bb39f6b12d144c60c4cbd78a1305c83ddda87af74a72774c21f82bb88ea774c2 |
| pure_tests_receipt_15.json | 5534 | ce678415d47da8149969801d0e85564c216ec9cabd3b795c8601687b562d9015 |

父 ZIP SHA256 为 6cf2db80be7b990efc8be40eff307e58351eae839970b0c78ce5c9b5193f8e70；父 policy 为 c20a8841dcf2174c2db584edca951adaafa12688b026172669ef0e07ad955212；父 Adam 为 bd4c898d33f4a121489f393dfe0f083737305f76fb139ddef74db2bb94878bf9。完整原生/依赖/旧源码输入身份由 root 的唯一 plan 与每阶段前后同集 hash 继续约束。
