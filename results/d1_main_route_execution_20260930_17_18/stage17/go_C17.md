# C17-C exact-source GO

2026-09-30，Astra审阅结论：**GO，仅执行qualification_spec_17.json规定的primary与mirror两个新worker，共24,000 controls。** B两个单因素修复已通过完整任务、独立物理读回和相同初态核验；本GO不预判组合结果。常规开发和有界验收在用户已授权范围内。

## 已满足的前置条件

`development_summary_17.json`来源闭合为true；四组初态比较的qpos、qvel、ctrl、qacc_warmstart、observation均逐位相等。baseline重现yaw RMS 0.13036746766567778与ramp均速0.510070419136454、RMS 0.062368568044230605，两个原任务均失败；yaw单因素RMS降至0.0947655026817745并通过；ramp单因素均速0.46985660182497285、RMS 0.02120257264488338并通过，坡面正轮载荷及四整轮清除通过。三个readback的记录、安全和最终停止检查通过。B实际账为8,600 controls、43,000 normal native、6 compiler、3 PPO load及9 torch.load，无训练。

此前已经审阅并执行的11项纯测试与导入/语法检查有效；下列运行控制、独立算术和reader来源与B冻结值完全相同，未重复测试。新的freeze_C17.py仅验证B前置、来源和C预算，绑定全部B归档及readback后独占写入C plan；没有新增模型或物理执行。readback_host17_v02.py保留295s等待和最多5s回收，总预算300s，原冻结v1不调用。

## 本GO绑定身份

| 文件 | SHA256 |
|---|---|
| qualification_contract_17.md | 4ab3a9e4f1346ae1f085a40a1853ed52df1dade025f3bde1b6b9821842bb0fcf |
| qualification_spec_17.json | 4c721267415d776c418c4aaf640a54f8d70592bfbf303202dcd2b28bfcc62690 |
| development_summary_17.json | e5d348905cd7dff86cb6466a7415b1b8a97953b7da8acaab833688078852d668 |
| development_baseline_readback_17.json | 41d196d83e322d8c39bf353a23ea24b69dd6d31f05aae15c9179bf3fba66585b |
| development_yaw_readback_17.json | d78dc80f63502e74e17f6d89459d8e1825be4b0a0311bc113b0a48460a3ff4f4 |
| development_ramp_readback_17.json | b51c901d747bd3bc833ffbdfa8b966d55aea42a200147e9a4407abd773507781 |
| freeze_C17.py | 55128141328f6b66bf35525b48393f72e05bc1c27c5237c79a85e0600f462ab4 |
| readback_host17_v02.py | 88b231051d6e0f3d72691ff3839587ca9db4b2664cad532b5ec0bcfb1ef01530 |
| readback_host17_v02_identity.json | 1de297b180887530c5d867227cb18af6049b990d6d693455725f4c278795d8b3 |
| pure_tests_receipt_17.json | ee1dce3e885167ea420d71184801ed1e2e64d354d1b8913c1d710a421af6b3c5 |
| go_B17.md | 785f8b23934441fc506ba8a9b7699e9ac76ba4abda447d93dd3d8f018d5e8e01 |
| controller17.py | a5ee7e809212313011180ce81b1c2ef70cd3e5807bb52a67e4d08e547f9e85d3 |
| residual17.py | 5cc145688772dcf252c7413c06e980ac047a4dd41f9462a4f50b568021cde42c |
| verify_control17.py | a1909c215367947775241f6b8199925dcf6da5d0e5aaa92453f4ce3161a85cd2 |
| read_eval17.py | d2617aba72eb6aadc92bd7fd1439ba2d41489419cade79c62b82270418babb6f |
| recipes17.py | 698a8534ce42d7a09935b31c99d6752d9b795abec7a68be4a52186ac4d8e4f3d |
| score17.py | 69a78562c79ca69102e505530a5d75a2f4d32b92383e81082bc38de9ac7f98dd |
| floor_bridge17.py | e9a272c56bcb1661dd68d9512c5fce97ad5b43912e8f70221439c4a54d0e236d |
| offline_floor17.py | 4c0fc95beeb23cdcbb72891de2293c70e9d81ae08167c5140db4a51c03934d93 |
| eval17.py | bf97aa2ca3614111ea4cdd54e18464a0572cc220520efd301d2b36d5c3ad0fa0 |
| worker17.py | 0d4bc77042d7adc59431d1e9bdb4ebcadaebbbabb057de9253dbf181aed3a975 |
| host17.py | aedc41b73b8c48fae8ca807f63f5c3747eaae0c21dd67e6a310addaf4f7e3932 |

唯一policy ZIP为grouped final，SHA256 `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`。B plan、全部历史输入、完整B worker归档和各reader/host receipts由freeze_C17.py进一步直接收入C plan的路径/hash表；本表不代替完整来源闭包。

## 固定执行及判定

两个worker都固定combined：yaw内部上限1.2、gain4不变；common active时采用已验证的共同积分误差及对应退绕方向，P、残差16通道与保护不变。primary为六个原任务、每项zero→grouped，floor171190与case171201–171206，20,200 controls；mirror为反号yaw、zero→grouped，floor171390与case171403，3,800 controls。exact WorldUprightCourseEnv、cold构造、原子归档和完整5T/contactForce独立核验照旧。

此C设计在任何C执行前已取消仅换seed的重复原yaw/ramp。当前oracle reset忽略seed的物理随机化，B已精确重现旧分数；新的seed只作为归档身份。七场景是确定性脚本验收，不声称独立随机试验或未见分布统计泛化。

预算总上限：24,000 controls、120,000 normal native、4 compiler、2 PPO load、6 torch.load、12,602 predict API、12,664 actor rows；训练/value/backward/optimizer/save全为0。primary soft/close/hard为1,200/1,320/1,380s，mirror为600/720/780s；各独立正式读回总≤300s。不得追加额度、修改冻结来源、原位重试或因结果改场次。

通过须grouped在六项原任务和镜像yaw全部满足原任务、记录、安全、完整horizon、几何及最终停止门。yaw RMS≤0.12；ramp均速0.45±0.04且RMS≤0.05以及原三坡面轮载荷/四轮清除。任何所需项失败应记录并继续新的有界修复，不用zero或B单因素通过替代。RL总体贡献按同combined基控的primary原六任务配对另判，失败仍记false；不把基础控制修复计为RL收益，不授予GUI/自由键盘/实机资格。

本次Astra仅执行只读文件/身份检查及写入本合同GO，没有执行模型、物理或测试。两个C worker与其独立读回由root按冻结计划执行。
