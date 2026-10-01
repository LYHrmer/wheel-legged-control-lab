# C30 增量公开包清单（草案）

本包是有来源哈希的**研究代码与保存证据快照**，不是可便携的一键运行程序。C30 是 headless 侧步参考试验；尚未接入已交付的 B22 GUI，也没有横移 RL 训练或速度优越性结论。六场结束后由 root 填入最终范围、数值和结案审阅；当前已保存并独立读回的仅 `fixed_nonzero_left_01`、`fixed_nonzero_right_01`、`zero_left_01`，不得把后续三场写成已完成。

## 应公开的新增源码（逐字节复制，不改名或改写）

- C30 合同：`continuation30/skill_contract_30.md`、`scope_clarification_30.md`、`spec30_draft.json`；各场对应的 `source_go_*.json`、请求与 reservation 作为来源身份。
- 技能和纯测：`continuation30/sol30/side_skill30.py`、`test_side_skill30_pure.py`、`test_airborne30_pure.py`。
- 实际 owner、归档和宏转移：`continuation30/root30/{worker30,runtime30,record30,macro30,make_candidate30,bootstrap_check30,launch_bootstrap30,test_headless30_pure}.py`。
- 独立读回：`continuation30/read30/{read30,reference_read30,geometry_read30,score_read30,reward_read30,test_macro_read30_pure}.py`。保留原文件字节和命令入口，勿仅发布合格布尔值。
- 最小上游接缝源码：`continuation27/sol27/hybrid_adapter27.py`、`continuation27/root27/side_runtime27.py`、`continuation24/kinematics24.py`。其余实际加载的旧 B22/C18/C26/课程模块、模型、MuJoCo 绑定和本机环境由各场 `source_go` / session 的 `inputs` 精确哈希标识；不在此增量包中复制 B22 大型运行 bundle，也不声称只靠这些文件能复现执行。

## 应公开的证据与失败记录

- C27：`bootstrap_failure_review_27.json` 及该次 `zero_left_01` 的 host/worker/bootstrap 失败回执；明确是环境导入失败，未取得侧步任务或物理资格。
- C28：`final_failure_review_28.json`、`zero_left_01` 的 host/worker、deferred archive 和保留下来的阶段回执；保留已实际执行数量及最终来源闭合失败，不把保存轨迹冒充合格运行。
- C29：`final_profile_review_29.json`、`formal_reader_failure_01.json`、`profile_left_01` 的 host/worker/阶段计时回执；这是有限诊断，非任务资格。
- C30 每场：独立 `independent_<case>_01.json`、`reader_<case>_execution_30.json`、对应 review（若已有）、host/worker/supervisor/segment、`macro_transitions30.json`、模型调用及来源回执。当前三场各自 `qualification_passed=true`、`record_valid=true`、`success_after_retention`；这只描述各场已验证结果。保留纯测与 bootstrap 回执、曾失败的纯测回执，不能只挑通过记录。
- 最终六场汇总/审阅在生成后再列入，缺失时公开包必须标为**阶段性草案**；不预填未来零/取消场的门或数值。

完整 `controls.jsonl.gz`、native blocks、states/initial NPZ 和大轨迹留本地；公开包可附每场首控制/native 小样本及宏摘要，但需逐文件 SHA-256/字节数、本地相对路径、所属场和生成回执的全量 inventory。公开子集不足以独立重跑完整读回或物理仿真，应直接写在包说明中。任何样本须从已封存文件按字节复制，不能重序列化。

## 依赖与署名口径

重执行依赖已封存的 B22 checkpoint、C18 控制链、C24 几何、C27 hybrid 侧步接缝、C26 保存数据读回、公用旧 FastSideStepController、精确 MuJoCo/NumPy/Python 环境及模型资产；具体来源以原 GO/session 哈希为准。不要把此快照宣传为独立安装包、硬件结果或 GUI 功能。

Claude Opus 5 先前仅编写 C27 保存数据离线分析模块 `claude27/analyze_saved_side27.py` 与 `test_saved_speed27.py`，由 root 执行其测试；C30 技能源码由 Sol 编写，方案与独立读回由 Astra 负责，root 编写 owner/归档接入并独占模型、物理和实际验证。发布时保留这些不同职责，不把离线分析贡献写成 C30 控制或实验执行。
