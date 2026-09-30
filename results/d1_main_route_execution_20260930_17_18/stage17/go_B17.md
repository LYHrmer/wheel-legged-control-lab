# C17-B 精确源码运行审阅

**GO — root可冻结并执行本合同的baseline、yaw、ramp三个独立开发worker，各一次；总上限8,600 controls。** 不授权本GO之外的训练、换模型、参数扫描或资格场次。B完成后按实际结果推进组合验收，不能以本GO或纯测试宣称转向/坡道已通过。

2026-09-30，Astra。已静态审阅最终controller17/residual17、独立verify_control17/read_eval17、recipe/score、floor bridge/CLI、host/worker/eval与development_spec；已读取root的 `pure_tests_receipt_17.json` 并将当前源SHA与该receipt逐项核对一致。没有剩余阻止本批开发执行的静态问题。

## 科学因素与调用路径

- yaw配置只将已有反馈的内部L从0.6改为1.2，gain4不变；ramp配置只将积分共同误差改为 `(servo_vx-pre_body_vx)/.087`，差分误差保留，同时按实际eI确定antiwindup退绕方向。P仍使用wheel误差，Ki3、dt.01、I±4、pre-common作用点、物理限幅及全部16RL映射不变。已对照residual17与冻结residual16的diff，未夹带奖励、观测、动作scale、地形、servo或评分门改变。
- installer只作用于never-reset的exact WorldUprightCourseEnv；保留plant/provider，替换固定新adapter和同类WorldUprightLoop，并断言nominal geometry只cache hit、不新增cold miss。所有真实physics仍走旧runtime、archive13和nativeguard。worker运行时继续核compiler2和每control5 native，不能由静态判断代替实际账。
- 新独立checker不调用live residual函数；它核pre-state速度/yaw、servo、nominal未限幅/限幅yaw及足横向映射，复算eI/PI/支撑/阻尼/保护。reader把variant绑定到plan/session，禁production test_actor_probe，重建stop latch和4维积分连续链；保留C16完整controller→实际16力矩→5T积分→contactForce→post metrics链。
- floor CLI已指向read_eval17，clean process的禁策略/物理导入、去loader injection、60s含回收、来源与输入输出hash保持。live worker不导入reader。此前审阅发现floor record_case缺少必需pair_initial的问题已修为None；未执行物理前即解决，不制造失败运行记录。
- 新recipe使用实际新seed并显式mirror身份，score只改变身份接口，原任务速度/yaw、安全、停止和坡道载荷/整轮清除门保持。generic combined/新seed接口已在本次冻结前实现，**接口存在不构成C资格预算授权**；未来C须先冻结自己的合同/spec/GO。

## 唯一B预算与来源

development_spec固定三配置controls=4,000/2,200/2,400，总8,600；normal native≤43,000，cold compiler≤6。每worker一次PPO load、至多3次torch.load与一次32行probe，总PPO≤3、torch.load≤9、predict API≤8,603、actor rows≤8,696。critic/value、train/learn/backward/optimizer/save全部0。

唯一策略为grouped final `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`，文件/manifest/metadata/probe纳入同集冻结。seeds=171090 floor、171103 yaw、171106 ramp；root已核历史未执行。旧77源、C15两训练和C16完成评估不改不重跑。baseline提供当前新seed对照；跨worker五初态数组须在独立开发汇总中逐位配对，跨worker物理地址不要求相同。

host各自独占reservation/output，失败不退款、不原位重试。时间按已绑定development_spec且不超过合同各600/720/780s上界；各floor reader≤60s计入所属host，最终offline reader各≤300s/总≤900s。source前后同集hash、首异常、清理和所属进程退出必须进入实际receipts；未用控制额度不得转给另一worker。

## 必要验证

root报告11项针对性pure tests通过；已读取receipt中pytest exit0（外层0.36549s）、C17 clean reader import probe exit0（0.21573s）、14个根Python源fatal ruff exit0（0.11550s）。receipt记录changed_sources=[]、controls/native/model_calls/optimizer_steps均0、passed=true。测试覆盖新积分误差方向、差分与baseline算术、停止/保护、eI退绕、独立checker拒绝被篡改字段，并由root补齐yaw与接入检查。没有重跑C15/C16旧批次。

## 精确身份

路径相对continuation17。以下身份全部进入root生成的唯一plan_go_B17；若其中任何文件改变，不能沿用本GO。

| 文件 | SHA256 |
|---|---|
| controller17.py | `a5ee7e809212313011180ce81b1c2ef70cd3e5807bb52a67e4d08e547f9e85d3` |
| residual17.py | `5cc145688772dcf252c7413c06e980ac047a4dd41f9462a4f50b568021cde42c` |
| verify_control17.py | `a1909c215367947775241f6b8199925dcf6da5d0e5aaa92453f4ce3161a85cd2` |
| read_eval17.py | `d2617aba72eb6aadc92bd7fd1439ba2d41489419cade79c62b82270418babb6f` |
| recipes17.py | `698a8534ce42d7a09935b31c99d6752d9b795abec7a68be4a52186ac4d8e4f3d` |
| score17.py | `69a78562c79ca69102e505530a5d75a2f4d32b92383e81082bc38de9ac7f98dd` |
| floor_bridge17.py | `e9a272c56bcb1661dd68d9512c5fce97ad5b43912e8f70221439c4a54d0e236d` |
| offline_floor17.py | `4c0fc95beeb23cdcbb72891de2293c70e9d81ae08167c5140db4a51c03934d93` |
| eval17.py | `bf97aa2ca3614111ea4cdd54e18464a0572cc220520efd301d2b36d5c3ad0fa0` |
| worker17.py | `0d4bc77042d7adc59431d1e9bdb4ebcadaebbbabb057de9253dbf181aed3a975` |
| host17.py | `aedc41b73b8c48fae8ca807f63f5c3747eaae0c21dd67e6a310addaf4f7e3932` |
| development_spec_17.json | `6220bd695a3890e8533d51fda61e2bb9f79a619853b5d934d0c747aa8afa018e` |
| intervention_contract_17.md | `890424e4f04b74eecf3a1f00596f1307de5afa8b3093bea5cf9d1f29ad560e14` |
| pure_tests_receipt_17.json | `ee1dce3e885167ea420d71184801ed1e2e64d354d1b8913c1d710a421af6b3c5` |

Astra仅做静态阅读、已有结果阅读、SHA计算与写本审阅，新增测试、模型、优化器和物理为0。实际成功须由三个worker的真实记录与独立读回判定。
