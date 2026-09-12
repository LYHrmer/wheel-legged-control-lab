# 正式预算结果的目录迁移复算摘要

验证通过：正式 study 的 4171 个文件、3296266633 bytes 曾完整复制到另一物理目录，使用归档分析器复算 3072 次更新与 200 个评测案例，并独立重读 24 个预算 ZIP 和 6 个根目录最终 ZIP。原与迁移副本在前后都逐文件 SHA 一致。

严格比较只允许分析顶层 study 与六个模型的 PPO 记录 directory 共 7 个路径字段变化；权重报告只允许顶层 study 变化。其余数值、案例、更新记录及参数/文件哈希链逐值精确相等，不使用数值容差。结果仍为 198 例完整、2 例地图边界终止、192 例质量达标。这是对同一批实验数据的复算，不能增加独立实验或案例数。

`verification_record/` 原样保留报告、输入 SHA 清单、命令、读取 guard 及回执、比较源码和原 verification manifest。原 README 描述的是当时本机工作目录，里面的绝对路径是历史记录；此公开摘要组没有复制 3.30 GB 的 `relocated_study/`，也没有复制整份重复分析 JSON。正式数据仍在 `results/d1_budget_study`，原分析和权重结论在 `d1_budget_study_analysis` 与 `d1_budget_study_weights`。

`archived_readers/` 保存本次实际执行过的三个分析模块及权重读取脚本。完整输入的每项 size/SHA 见 `verification_record/input_sha256.json`；原始完成回执及其校验和保持不变，本组另用 `delivery_manifest.json` 对实际复制成员提供完整清单。轻量读取 guard 阻止原仓数据/代码读取和迁移输入写入，分析与权重分别记录 11275、142 次新目录 open，无阻止项。

本验证使用同一机器和已安装依赖，没有做依赖重装、跨平台测试或 GitHub 回下载，也没有重训、仿真评测、优化器重放或 Claude 调用。它不替代后续 Release 附件下载、join 与成员 verify。比较源码快照依赖原工作目录的通用清单函数，作为审查证据保留；复算入口以本组归档读取脚本、正式 study 和全新输出目录为准。
