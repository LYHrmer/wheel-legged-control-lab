# 2026-09-23：零调用诊断完成，最小原生 CCD 逐位复现坏接触

完整目标仍是“稳定直行 → 可靠越障 → 提高速度”，尚未完成。本轮未新训 RL、未重跑任何旧物理批次。R 仍是仿真 reset，物理自救未实现，新 RL GUI 仍无代码。

工作目录 `W=/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923`。续进先读本文、`W/continuation_state.json`、`W/final_publication_receipt_01.json`，再读[公开结果包](../results/d1_driving_stability_development/contact_geometry_forensics_20260923_01/README.md)。9月14、21、22日交接和旧工作目录原样保留。

## 实际分工与授权

实际 GPT-6-astra ultra 主要负责方案、指挥和独立复审；用户已明确让实际 GPT-6-sol 接替额度不足的 Claude 执行有界模块。主代理集成、计数、执行全部实际查询和验证并上传。三次早先 Claude 请求均连接失败，0成功响应/0生成代码；用户另报告额度不足，此后不再调用 Claude。不能把该故障笼统记录成成功的 Opus 工作。

常规开发、验证、GitHub 上传已获授权。所有 shell 命令加 rtk；本仓库无 .codegraph，不自行建索引。SSH443 上传方法继续沿用9月22日交接。不要用父提交 CI 代表本次提交；准确 HEAD、远端 main 和对应 CI 以 W 的最终发布回执为准。

## 完成的两份有界工作

上一终端零调用合同 SHA256 `157f0a33b386cea0ef99556a0d693370126ae57479db0460e4b211dd7b27e7d9` 已完成。新增纯诊断入口、真实失败 fixture 和 12 个案例，根代理仅运行 1 轮，全部通过。两场完整离线诊断恰好 2 遍，三份结果文件逐字节相同，receipt 只差 pass 计数；扫描预算 2/2 用尽。本阶段 MuJoCo 导入尝试、构造、静态/动态调用均 0。

两场记录的 `raw_record_links_valid=true`。单箱46387条候选中仅 native3486/contact5 几何失败；11032条正载荷箱体候选均满足原几何规则，仍不能替代所有候选门。原 `single_15mm_box/score.json` 保持 record_valid=false/task_passed=false，未改判。

拟议 kernel01 因源码发现 XML 编译器隐式 `mj_step` 而在执行前被阻断，**从未激活**。保留其原合同和 BLOCKED 记录；不要运行该模型构造入口，也不要用 no-DOF 或新 data.time=0 来豁免内部积分。早先 review 末尾的可激活建议已由后续 review 明确撤回。

替代 kernel02 合同 SHA256 `0869d5836add5dc12a295d5ac4618bff984244c9703516202f9485a29fdf4768` 使用只读 typed primitive carrier 初始化描述符，调用固定3.12.0库的非公共原生 CCD，加源码等价本地扰动 wrapper。最终 C SHA `27bca448180d267c2954f127279c70939f85420d5d1b5da4fe991d63da8cf888`，编译二进制 SHA `86b831e7aedb51e3cca05acdd4211528edcc73cc0b529fb09a848968aefb2aad`，140项输入在一次运行前冻结。

主代理实际执行16次 CCD 尝试/16次返回（含条件 primary-only），额度全部用尽；descriptor init8、size1、显式纯数学171、version/versionString各1。模型编译/分配、控制步、原生积分、训练均0。没有借静态研究复跑旧完整机器人重建。常规 GitHub 两版本全仓 CI 另记，不宣称它也绝无 MuJoCo 调用，不混入本地预算。

## 确认的物理证据与仍未知事项

3485/3486/3487 三组保存姿态的最终接触各3条，位置、距离、完整frame全部与旧记录逐位相同。坏候选确定来自 query7：3486 第一切向轴 −0.001 rad 扰动；原生记录 GJK 计数字段6、EPA 循环索引字段4、epa_status0（并非宣称 EPA 总共4次迭代）。normal 来自原生 witness 相减，不是记录符号转换或距离覆写造成。状态0不能独立证明收敛；这次不是35迭代上限耗尽。

primary-only只返回正常主接触，其 position/dist/normal/frame 与旧 primary 逐位相同。该静态事实不等于关闭 multiccd 后完整机器人的耦合动力学仍稳定。内部 EPA 选面、退出分支及 affine 权重失效机制尚未确定，引擎尚未修复，台阶资格与 RL 门仍关闭。

主代理纯输出复核见 `W/native_ccd_reconciliation_02.json`，最终预算见 `W/native_ccd_budget_closure_02.json`。所有尝试事件在调用前fsync；启动计数与最终计数分别保留。不要把初始budget的 attempted 状态当成尚未运行。

## 下一工作边界

下一份[源码修复合同](../results/d1_driving_stability_development/contact_geometry_forensics_20260923_01/next_epa_source_fix_plan_01/next_contract.md)已冻结，W路径为 `next_epa_source_fix_plan_01/next_contract.md`，SHA256 `8fd28ec3e2eec24db0c0b758dad0cc5b97bdf963a73d5c4c2c8a156dae47ee46`，**尚未执行**。它允许先16次隔离源码核插桩复现，机制审阅通过后才开放单个补丁的24次固定回归，总计最多40次 source-local CCD，0动态/0 installed mjc_ccd 调用，不复用旧16次额度。先读取实际结果与独立复审；只做隔离局部源码取证和有明确上限的最小修复研究，不修改已安装引擎或原机器人。不能过滤零力接触、翻转法向、取绝对值、放宽0.0021或调整margin来刷通过。不追加第三次平地跳跃训练。新的物理 readiness 或 RL 须等实际数值修复证据后另立明确预算，不能复用已耗尽额度。

历史记录的控制/observer窗口累计仍为317824控制/1589120原生积分（不含更早三个65k与24场G1）；**不再把它表述为覆盖编译器内部调用的全引擎总计数**。新发现的隐藏编译step未在旧观测账本完整计数，本轮没有追算确切历史总数，也不改写旧记录。原两场2400/12000是保存窗口内计数，不能证明窗口外编译调用为零。详见公开包 `provenance/historical_counter_scope_addendum_01.md`。本轮两阶段确无模型编译/积分，结论范围明确。旧两次131072训练及物理、已完成2400/12000 readiness、旧四姿态静态重建、本次两遍完整离线诊断与16次CCD均不得重复。77冻结输入、前轮215输入、本轮193物理输入及所有旧记录哈希保持不变。
