# 侧移驱动最终集成 QA

全量 pytest **3121项通过，0失败/错误/跳过**，372.99秒；206条依赖库警告。当前源码 `ruff check src scripts tests` 通过。测试期间源码未变化，正式实验冻结77文件的前后哈希均匹配。

全仓 `ruff check .` 返回57项问题，全部在 `results` 的记录脚本中；因此全仓静态检查不能报告为通过。本次保留记录原貌，详细路径、规则和位置见 [ruff_findings.json](ruff_findings.json)。

真实快步集成探针通过公开输入接口注入失焦：空中取消后60拍完成落脚，再运行100拍原驾驶控制。合计482拍，legacy计算300次、fast计算182次，物理步进和状态更新各482次。无跌倒或异常机身接地，最大绝对roll/pitch为0.012180rad；没有重置机器人位置。该项是实际仿真测试，不是桌面焦点或人工验收。

默认profile为fast；`--side-step-profile conservative`直接选择原保守SideStepController。原保守控制器SHA保持为42b488660a300077eb7c89f37d6d8649f360ade5852f9ce4f9c2c317ef4d134e。

GUI04收到测试脚本序列之外的S/H/Backspace事件，随后失焦并取消A。输入和焦点变化来源未知；没有证据将其归因于用户或某个工具。该次尝试在abort_hold期间停止，不能作为完整落脚验收。详细事件及原始文件哈希见 [gui04_diagnosis.json](gui04_diagnosis.json)。

侧步读取simulator truth，适用范围是平地仿真；这里没有PPO效果、硬件鲁棒性或人工驾驶通过的结论。

文件：

- [精确计数、命令、版本、原始证据哈希](qa_summary.json)
- [全量测试原始输出](pytest.log)、[当前源码Ruff输出](ruff_sources.log)
- [快步失焦落脚探针摘要](fast_focus_handoff.json)
- [测试时完整源码哈希](tested_source_sha256.json)

大型NPZ及逐拍数组保留在原work证据中。本目录的manifest只覆盖其中明确列出的QA文件；后续GUI证据子目录应保留各自manifest。
