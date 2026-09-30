# C20 有界课程训练公开记录

结论：A 训练有效，B 在 18,000 个控制周期后的下一次复位被新命令的名义路径几何资格拦截；完整 A/B 收益比较未成立。唯一三段转向修复候选通过保存 B18 几何的离线检查，没有补训或后续物理评估。

- [中文报告](../../docs/main_route_execution_20260930_20.md)、[当前入口](../../docs/current_status_20260930_20.md)
- [结果汇总](execution_summary_20.json)、[实际执行账本](execution_ledger_20.json)
- [A 完整独立读回](train_A_1_readback.json)、[首 1024 步逐位配对](first1024_pair_20.json)
- [B 原始 worker 失败](train_B_1/worker_receipt.json)、[保存几何复算](failure_geometry_snapshot_20.json)
- [有界离线诊断](offline_corridor_20_02.json)、[名义路径图](figures20/nominal_corridor_20.png)
- [Astra 源码 GO](source_review_20.json)、[Astra 失败终审](failure_review_20.md)
- [训练合同](training_contract_20.md)、[评估合同](evaluation_contract_20.md)、[机器规格](spec20.json)
- [RL 纯测试回执](pure_tests_receipt_20.json)、[GUI 输入测试](gui20/input_test_02.json)
- [逐文件本地档案清单](local_archive_inventory.json)、[复制回执](publication_receipt.json)、[发布清单](publication_manifest.json)

复制子集约 108.5 MB，共 932 份源产物，含源码、合同、检查、读回、实际梯度样本、首 1024 数值/Gaussian 数据、A 最终 checkpoint、B 失败 checkpoint 和预先声明的少量 control/native 块。完整本地记录约 1.30 GB，其余原始块的 SHA256 和路径均在清单中。**本公开子集不足以重新执行完整原生接触读回。** 缺少的开发/最终样本对应未执行阶段，没有隐藏失败场次。

A final 未做 heldout 资格；B failure checkpoint 禁止用于正式评估或补训。实际已通过七个固定脚本的权重仍引用 [C15 grouped 父包](../d1_main_route_execution_20260930_15_16/README.md)。这些是带历史绝对路径的研究快照，依赖冻结工作目录与引擎布局，不是通用 GUI 启动器。GUI 输入模块仅有纯测试资格，R 请求表示仿真复位。

首次离线诊断导入失败与修正环境后的第二次启动均保留；第一版源码依据记录 SHA 校验恢复为 `reporting20/diagnose_corridor20_attempt01.py`。第二次实际运行 161,000 次纯命令 servo 积分，没有新增模型、训练或物理调用。离线静态通过不能推广为动态安全或连续课程全域资格。
