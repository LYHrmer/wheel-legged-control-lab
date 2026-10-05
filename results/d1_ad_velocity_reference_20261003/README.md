# A/D 横向速度研究子集：C34 与 C35

本目录含两个连续的否决结果。两者都没有合格提速、没有 RL 收益、没有 GUI 交付。

## C35：固定两支撑连续横移试验被第一例否决

唯一一次授权的物理运行在第一个用例 `pair_stand_zero` 内被硬接触门禁终止。200 步 B22 准备和 4 步侧移控制之后，第 5 步侧移控制触发 `C35 new all4 actual support/load bound`：对角摆动轮 3 的独立重建法向载荷为**精确 0 N**、实际几何间隙 +0.87 mm，发生在必须至少持续 0.06 s 的 `transfer` 相位历时 0.05 s 时。**没有完成任何双腿交换，没有测到任何横移速度**，后四个用例（双向连续、双向取消）从未执行。

根因是两份冻结文档互斥：`clarifications35_01.json` 要求 `transfer` 期间四轮载荷都严格为正且无宽限，`clarifications35_03.json` 又要求同一相位把摆动轮分配权重在 0.06 s 内归零。不是调参或种子问题。

- [完整结果与根因定位](../../docs/ad_lateral_pair_feasibility_20261004.md) · [本轮交接](../../docs/codex_handoff_20261004.md)
- [独立报告](continuation35/independent_read35_01.json) · [root 裁定](continuation35/root_adjudication35_01.json) · [闭合状态](continuation35/continuation_state35.json)
- [保留的 native 接触失败记录](continuation35/development_01/episode_0/native_contact_failure_0000.json) · [唯一仿真回执](continuation35/development_01/worker_receipt.json)
- [实际 Astra 源码 GO](continuation35/source_go35.json) · [合同](continuation35/astra_plan/contract35.json) · [四份 clarification](continuation35/astra_plan/)
- [独立验收回执](continuation35/readback_execution35_01/receipt.json) · [公开文件与完整本地清单](publication_manifest35.json)

C35 实际 204 个控制步（200 B22 准备 + 4 侧移）、1,021 个正常 native 子步及 2 个构造子步，上限分别为 9,000 / 45,000 / 2；201 次 B22 预测（含一次 32 行探针）、232 行。没有新策略、价值网络、优化器步骤或训练；1 次授权物理尝试已用尽，未自动重试。本轮没有新的 Astra 终审，裁定由 root 依据保存证据写出并已显式标注。

## C34：速度补偿未通过提速质量门

唯一新增案例为左向 30 mm、β=0.5：周期 **8.83 s**，原 C33 固定参考为 9.37 s。独立读回确认原任务和 native 安全通过，但目标误差 11.45679 mm、位移保持率 0.959851 未达预先冻结的质量门，后续开发点立即停止。没有合格提速、40 mm 结果或 RL 收益。

- [完整结果与计步](../../docs/ad_lateral_tracking_20261003.md)
- [独立报告](continuation34/independent_read34_02.json) · [实际 Astra 终审](continuation34/final_review34.json) · [闭合状态](continuation34/continuation_state34.json)
- [唯一仿真回执](continuation34/development_01/worker_receipt.json) · [停止和选择记录](continuation34/development_01/campaign_selection34.json)
- [首次读回失败](continuation34/readback_execution34_01/stdout.log) · [修复后的读回回执](continuation34/readback_execution34_02/receipt.json)
- [公开文件与完整本地清单](publication_manifest34.json) · [设计及打包回执复制身份](supplemental_manifest34.json)

C34 实际 1,483 个控制步、7,415 个正常 native 子步及 2 个构造子步；600 次 B22 控制预测和一次 32 行探针。没有新侧移策略、价值网络、优化器或训练。

## 公共边界

两个子集的公开文件均逐字节复制，原始轨迹、native、状态和模型载荷留在本地；**任一子集都不能独立重放完整验证，也不是可移植运行包**。完整清单列出大小、SHA-256 与每条排除理由。

下一步必须先由实际 gpt-6-astra ultra 作出两项裁决，而不是直接改语义重跑。[步态方案重审](../../docs/ad_lateral_gait_review_20261004.md)（零物理步、零模型调用；[时序与机构脚本](review_20261004/gait_review_20261004.py)、[姿态静力脚本](review_20261004/posture_statics_20261004.py)及两份输出随文保存）把问题拆成三个独立约束：

1. **时序**：理想上界 0.02624 m/s，渐近线 `v_peak/3.75 = 0.042667 m/s`，0.030–0.040 m/s 发展目标结构上不可达，唯一一阶杠杆是脚端速度剖面。
2. **静力 x 向**：整机 COM 比四轮接触矩形中心前移 21.7 mm，两条对角线都不过 COM，所需力矩超接触斑容量 1.34–1.38 倍；把标称 thigh 从 0.800 改到 **≈0.731 rad** 即精确对齐，机身高度与站宽不变，**零时间代价**。
3. **静力 y 向**：密封 `landing_xy35` 的 0.975 s 落点前瞻只在 28.2 mm 步长时让漂移对称，60 mm 下退化为整程单向漂移，**把速度限到 0.02365 m/s**；前瞻改为 `1.5·T_pair − 0.18` 后余量 23%，同样零时间代价。

排除性结论：纵向加速度不能替代姿态修正（30 mm 纵向上限只允许所需力矩的 10.2%）；冻结合同没有为两支撑相位定义任何稳定性判据。URDF 质量树与正运动学已对保存前状态校验（质量 0 误差、COM 0.002 mm、轮心 ≤0.18 mm）。

因此需要：(1) 修语义同时纳入上述两个标量修正、补稳定性判据、目标下调到 0.020–0.026 m/s；(2) 对 >25–31 mm 的侧向重定位决定是否改用已合格的 `W＋Q/W＋E` S 形换道。禁止扫描 transfer 参数、在旧 GO 下重试或追加训练。原始设计见 [Astra C35 主任务设计](design35/continuous_velocity_main_task35.md)和[连续横向速度主方案](../../docs/main_plan_20261003_ad.md)；设计目录前两份是架构分析历史。B22 GUI 的 A/D 仍未启用，R 仍是仿真复位，物理自救未实现。
