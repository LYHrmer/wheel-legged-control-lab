# C27–C30：侧移诊断、固定参考提速与 RL 前置验证

完整说明见 [侧移报告](../../docs/ad_lateral_skill_20261001.md)。C30 六场及配对门通过：左／右固定参考周期缩短 8.15%／7.96%，取消均 93 个控制拍安全交接。**横移 RL 尚未训练，B22 GUI 的 A/D 尚未接入。**

主要证据：

- [实际 Astra 终审](continuation30/final_review_30.json)、[六场配对审计](continuation30/paired_trigger_audit_30.json)、[实际审计执行回执](continuation30/paired_trigger_execution_30.json)。
- [事前合同](continuation30/skill_contract_30.md)、[权限澄清](continuation30/scope_clarification_30.md)、[首场来源 GO](continuation30/source_go_30.json)。六场 GO、请求、worker 和独立读回分别保留。
- [侧移参考源码](continuation30/sol30/side_skill30.py)、[控制记录源码](continuation30/root30/record30.py)、[独立读回](continuation30/read30/read30.py)、[配对审计源码](continuation30/pair_read30/audit_pairs30.py)。
- [C27 导入失败](continuation27/bootstrap_failure_review_27.json)、[C28 最终来源与 GUI 门失败](continuation28/final_failure_review_28.json)、[C29 诊断关闭](continuation29/final_profile_review_29.json)。失败不计成正式资格。
- [发布清单与原始轨迹库存](publication_manifest30.json)：302 个来源文件逐字节复制并核对 SHA-256；六场原始文件共 218 项库存，含未公开的大轨迹。

本目录是**研究源码与保存证据快照**，不是便携运行包。依赖冻结的 B22/C18、旧课程控制器、引擎绑定和原本机路径。公开内容不含完整 controls/native/states 文件，因而不足以单独重跑完整 reader 或仿真；原始文件仍在本地，库存记录其相对路径、字节数和哈希。静态源码、单场报告及派生指标不替代被省略的原始轨迹。

`publication_draft30.md` 是打包前的历史草案，保留其当时仅三场完成的描述；当前结论以最终审阅和六场配对报告为准。候选、失败测试和检查工具缺失日志均保留。`pilot_trigger=true` 表示达到后续独立 RL 合同的前置门，不代表训练已经执行。

方案与终审：实际 gpt-6-astra ultra；新技能实现：GPT-6 Sol；独立数值读回：另一 Astra；集成、测试与全部物理／模型执行：root。Claude Opus 5 的 C27 离线分析模块与执行回执另存，未参与 C30 控制场次执行。
