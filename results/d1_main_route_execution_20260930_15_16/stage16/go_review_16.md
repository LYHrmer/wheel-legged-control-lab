# continuation16 精确源码运行审阅

**GO — 批准本合同下唯一的新评估尝试，不批准训练或重复尝试。**

2026-09-30，Astra。已对比 C16 的 eval16/worker16/host16/read_eval16 与冻结 C15，对最终 floor_bridge16/offline_floor16 做静态审阅，并读取 root 的针对性验证结果和实际 child probe 记录。以下精确身份可冻结为 plan_go_16.json，随后 root 执行 continuation16/eval_01。C15 失败 reservation 及全部源和记录不覆盖、不退款；本 GO 不将 C15 的软件失败改写为成功。

## 通过依据

- 两个 final 在新 plan 生成时即锁定：host16 核对 C15 已关闭 eval session 的来源身份、训练独立读回的 manifest/ZIP identity、全部 checkpoint payload、metadata、worker/host receipts，再纳入 C16 冻结输入。执行时再次核对。使用的 global/grouped ZIP SHA 分别为 db324c8ba2c7a4821352f1d564b47d4fcffb97784d751e9ea20b23d525674d2b、1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb。
- live eval16 仅将原 floor reader 调用替换为 stdlib bridge；不导入 offline verifier。child 清除 loader injection，在 reader import 前安装禁止物理/策略模块的 finder，并在前后检查 sys.modules 与 native mappings。实际引擎 DSO 名确为 libepa01_engine.so，映射检查覆盖它。旧 NoPhysics 保护完整保留。
- bridge 正式路径最多被两场 floor 各调用一次。child 55 s 执行，剩余至总60 s用于回收；与 live owner 同进程组、不detach。RTK 后代按 PID 与进程出生时刻跟踪、终止和检查，host仍统一回收后代。bridge 将 case/schedule/states/geometry 与 archive manifests 做前后身份核验，输出 experiment_actor/checkpoint_sha256 与真实 floor receipt/schedule/folder绑定，不凭一个 gate 布尔值接受结果。
- worker 强制 arm=eval、training=False；新 stage16 session/worker/readback schema、run_id、plan/contract 明确独立。C16 reader 与冻结 C15 的差异仅是阶段身份和源文件定位；recipe、评分、完整 controller/native/contactForce 资格与晋级算术未改。clean child 与最终独立 reader 都只读已保存数据。
- 预算保持 E16 controls≤30,600、normal native≤153,000、compiler native≤2；PPO load≤2、torch.load≤6、probe2×32、常规 actor predict≤20,800，总 actor rows≤20,864；critic、learn/train/backward/optimizer/save 均0。两 floor 的结果决定后续场次，失败不重试、不换seed。host 1,800/2,040/2,100 s 的软停止/封存/硬上界与旧隔离 runtime/计数/原子归档/来源检查延续，不增加隐含物理调用。
- 累计 reservation 明记93,968 controls、469,840 normal native、8 compiler；若 E16 完整执行，累计实际63,368 controls、316,840 normal native、8 compiler。多出的两次已消耗 cold compiler 留在 C15 失败账中，不伪称仍处于原15预算。

本轮没有未解决的静态执行阻断项。首次真实构造/模型调用后的成功仍须由实际 receipts、两场 floor 与有条件正式任务的独立读回确认。本审阅不预判 grouped 收益；其首次大 KL 和放大的 Adam actor 步长必须在最终物理结果中讨论，不能以裁剪机制成立替代晋级门。无重复训练、无旧 final 新评估、无额外checkpoint选择。

## 针对性验证证据

root 的 pure_tests_receipt_16.json 记录2个针对性 pure case通过（0.27 s，外层0.522666799 s），7份Python AST与E9/F63/F7/F82通过；项目checkpoint load、physics controls、compiler substeps均0。第一例用假的已加载Torch复现直接 verifier 的正确拒绝，再通过真实 bridge 启动 clean child；第二例检查 loader injection 和已加载模块的拒绝边界。没有重跑旧22项。

已读取 pure_test_artifacts_16/test_loaded_torch_blocks_direc0/probe 的实际记录：child exit=0，0.203162088 s，未加载禁止模块，loader injection移除，owned_live_at_exit=[]、live_owned_after_reap=[]，无model/physics。result.json SHA256=7faf6f83bd19b754833eb5681212cb6d3b901150c1ce1725c8f73c333c4eae98；其 reader SHA 与冻结 read_eval15 的 0c4264473f87dfa2afa5e7433bdd6d9ffc4b054b4bebf86e57e761ab49500a7a 一致。

Astra 本轮仅静态读源、读已有JSON、计算文件哈希和写本审阅，执行模型、物理、测试均0。

## 精确身份

路径相对 continuation16。后续源码变更须新建差异审阅，不能沿用或覆盖此精确绑定；本文件和下列输入均进入新唯一plan。

| 文件 | bytes | SHA256 |
|---|---:|---|
| eval16.py | 4118 | 21484642459afc0c0631aa8fc74e778a6a68649b0a53c4e3e63c4a79fb6cad31 |
| floor_bridge16.py | 15530 | cd8873ade278c578b0bfeae2095ca1572ff2198eaca254a5919a17909a4f9529 |
| host16.py | 14247 | 0909eeeaf5ccd83bcba3790df74a17d0380a41e0468c79786d9fe44524690a8c |
| offline_floor16.py | 4578 | 94b852eddfee30c6e7c5bca4fb426e8613a835169234b8c1407028d3c050ff12 |
| read_eval16.py | 46127 | e9c0cfb2301d8b1e0cf882d398c82d9363cec07ffb726549e24347ff5aa8a9c2 |
| test_floor_bridge16_pure.py | 1936 | 762a8150482e2f9902964e147a173b42c2bb71154b14d6942337eda6e24e8476 |
| worker16.py | 13426 | 60092ffcd810606d8674f63498353a10763150ff87b14ed7ddf58cce53918330 |
| astra_contract_16.md | 10227 | 9401e1dbadf5236b3b55b4d7defe0ad814015154ac00ae9320818f1dfebd5225 |
| pure_tests_receipt_16.json | 2513 | 1dc70d45c8607379a9d09c68e1531c4d27ce14e649ecd8a6bb70428f1d0ff95d |
