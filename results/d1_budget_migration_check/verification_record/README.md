# 正式 D1 结果迁移复算

状态：`migration_recomputation_passed`。完整结论与逐字段差异见 `report.json`，该报告 SHA-256 为 `c4ad9bbfca985ba7cb4cdac9d7534c16674ff9cc8527d6ec58386e11a96a6982`。

`relocated_study/` 是原正式 study 的完整实体副本，共 4171 个文件、3296266633 bytes。复制前后原文件与目标文件均按 SHA-256 核对；完整输入清单为 `input_sha256.json`。没有改写记录里的原始绝对路径、模型、源码压缩或失败案例。实际使用 `reference/` 中从正式 analysis/weights 目录复制、并经原 manifest 验证的归档代码复算，未读取当前 scripts/src 中的实现。

两条真实执行命令与依赖路径见 `commands.json`。输出分别为 `recomputed_analysis/` 和 `recomputed_weights/`，其完成 manifest 已逐项验证。分析器覆盖全部 56 条命令、77 个冻结源码文件、3072 次 train-call、200 个评测案例；权重工具重新读取 24 个预算 ZIP 和 6 个根目录最终 ZIP。案例仍是 198 完整、2 个地图边界终止、192 质量达标。

比较采用类型和值严格相等，不使用浮点容差，不忽略整个子树。预先允许的变化只有：

- 分析顶层 `study`，以及六个模型各自 `training.<model>.ppo_math.directory`，共 7 个字段。
- 权重报告顶层 `study`，共 1 个字段。

这些字段必须从原 study 前缀精确替换为本次迁移前缀；协议、账本中记录的历史路径也必须原样一致。全部其他字段逐值完全相等，包括数值、案例配对、3072 个更新记录和参数/文件哈希链。归一化后分析的规范 JSON SHA-256 为 `996849cb52d44babc689ed6121a5de1acad8a829f3fee323d62d6dd569deb8a5`。

`migration_read_guard.py` 通过 CPython audit events 拒绝读取原仓数据/代码（仅允许共享 `.local-deps`）及写入迁移 study。分析记录了迁移目录内 11275 次 open，权重核验记录了 142 次；两个 guard 回执均无被阻止的访问。原始和迁移数据在两项工具结束后再次通过完整文件集合及 SHA 核对。

本次仍使用同一机器和已安装依赖，检查的是结果文件及归档读取代码能否迁移到不同目录；没有验证依赖重装、跨平台或 GitHub 回下载。轻量读取检查也不是操作系统级沙箱。没有重新训练、仿真评测、优化器重放、Claude 调用或远端写入。完整发布包的下载、join 与逐成员 verify 仍由后续发布流程单独完成。

`comparison_source.py` 保存本次严格比较逻辑的源码快照；实际执行脚本为相邻 `release/compare_migration_check.py`，其通用文件清单函数来自同目录的 `archive_course_evidence.py`。此目录的实体副本无需再次加入正式结果 Git 提交或原始附件；发布说明可引用本验证报告及输入清单。
