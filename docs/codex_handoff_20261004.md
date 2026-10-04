# 2026-10-04 交接：C35 固定两支撑横移试验被第一例否决

用户完整目标仍为 **稳定直行 → 可靠越障 → 提高速度**；当前重点是 A/D 横移太慢。本轮按既有授权执行了 C35 唯一一次有界物理验证和同一 GO 下的独立验收，结果为**明确的否决**，并已定位到一个设计层面的互斥约束。整体目标未完成，本轮没有取得任何可验证的 RL 收益。

仓库：`/home/lyh/wheel-legged-control-lab`（下文 R），分支 `main`。工作证据目录：`/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260924_full_drive01`（下文 W）。最新闭合状态为 `W/continuation35/continuation_state35.json`。

## 本轮实际做了什么

1. 全量核对 `source_go35.json` 的 **5,136** 项源码身份：0 缺失、0 不符；确认无运行中 Python 实验进程、无 `development_01` 预约。
2. 用既有 `request35_01.json` 通过 `root35/host35.py` 执行**唯一一次**物理运行。第一个用例 `pair_stand_zero` 内即被硬门禁终止，按合同立即停止，未自动重试，剩余预算未转用。
3. 按同一个 GO 用 `launch_saved_reader35.py` 完成保存数据独立验收，退出码 0，零物理步、零模型调用，前后源码身份无差异。
4. 依据保存证据定位根因，写出 `W/continuation35/root_adjudication35_01.json` 与 `continuation_state35.json`，并发布公开子集。

## 唯一的物理结果

| 项目 | 实测 | 说明 |
|---|---|---|
| 尝试用例 / 闭合用例 | 1 / 0 | 后四个用例（双向连续、双向取消）从未执行 |
| 完成控制步 | 204（200 B22 准备 + 4 侧移） | 上限 9,000；另有 1 次未完成危险控制尝试单独记账 |
| 正常 / 构造 native 子步 | 1,021 / 2 | 上限 45,000 / 2 |
| 完成的双腿交换 | 0 | 最低要求为原地 2 次、连续 4 次 |
| 实测横移速度 | **无** | 从未进入横向移动相位，不能给出任何速度数字 |
| 终止门禁 | `C35 new all4 actual support/load bound` | native 1020，仿真 2.042 s |

失败瞬间：对角摆动组 `[0, 3]` 在 `transfer` 相位历时 0.05 s 时，轮 3 的独立重建法向载荷为**精确 0 N**，实际几何间隙 +0.87 mm（真实离地）；分配器该步命令的摆动组法向力仅 0.576 N 与 0.0075 N，权重 `[0.03549, 1, 1, 0.03549]`。四轮法向和 466.18 N 高于 0.5mg = 236.16 N（质量 48.146865 kg），门禁的**和**条件通过，失败的是**每轮法向载荷 > 1e-8 N**。同子步非轮接触 0，姿态/高度/腿速原生门禁通过。

## 根因（已定位，不是调参问题）

`clarifications35_01.json` 把 `transfer` 划入四轮支撑区间，要求每个 native 子步四轮载荷都严格为正、无失接触宽限；`clarifications35_03.json` 又要求同一 `transfer` 相位把摆动轮分配权重按 `1 − smoothstep(elapsed/0.06)` 在 0.06 s 内归零，且 transfer 至少 0.06 s 才能进入允许离地的 lift。摆动组法向载荷的零点因此必然落在 transfer 内部。入场静态载荷不对称（轮 0 147.8 N 对轮 3 120.8 N），较轻的摆动轮更早到零，于是在 0.05 s 失败。**该门禁在当前冻结设计下无法满足。**

这一结论不能推出：可达横移速度、该机器人物理上做不到两支撑连续横移，或配对 IK、耦合惯量、落点投影、停止逻辑、成本核算有错（它们从未被执行到 transfer 之后）。也不构成任何 RL 结论。

## 下一步唯一任务

**先读 [横移步态方案重审](ad_lateral_gait_review_20261004.md)。** 那份零物理步的分析表明，仅修好上面这个互斥还不够：按冻结的参考限幅，对角双腿步态的理想上界只有 0.02624 m/s（渐近线 v_peak/3.75 = 0.042667 m/s，发展目标 0.030–0.040 m/s 结构上不可达）；更关键的是整机 COM 比四轮接触矩形中心前移 21.66 mm，使两条对角线都不过 COM，两支撑所需绕轴力矩 8.07 N·m 超出接触斑容量 5.87 N·m 达 1.38 倍，**该步态在本机质量分布下静力上不成立**，而冻结合同既没有按步机身平移也没有两支撑稳定性判据。补上平移后速度回落到 0.02154 m/s（激进重叠）或 0.01815 m/s（不重叠）。

因此下一步需要的是**两项决策**，不是直接改语义重跑：

1. 修改冻结的相位/接触语义（例如为完全卸载阶段单列只要求进入支撑组载荷为正的区间），**同时**补上按步机身平移与两支撑动态稳定性判据，并把目标下调到与机构相符的 0.020–0.026 m/s。
2. 对中远程侧向重定位（>31 mm，即用户实际抱怨的场景），决定是否改用已合格的 `W＋Q/W＋E` S 形换道——0.150 m 侧移仅需 2.598 s（0.3 m/s）或 1.293 s（1.2 m/s），比任何可达步进快 2.2–4.4 倍，代价是 0.76–1.54 m 纵向净空与瞬时航向变化，且属于任务重定义。

两项都需要实际 gpt-6-astra ultra 裁决、重新冻结源码并签发新 GO。

**禁止**：扫描 transfer 时长或权重（重审第 2 节已证明参数扫描在结构上拿不到发展目标）；在现有 GO 下重试；临场修改冻结源码后沿用旧 GO；追加任何训练或参数扫描。固定可行性未通过，因此**不建立同权限强固定基线，也不制定 RL 合同**。

## 能力边界（与本轮无关的既有状态）

| 能力 | 证据 | 不能推出 |
|---|---|---|
| B22 固定直行约 1.60145 m/s | 仿真证据，与横移分开报告 | 不是横移速度、不是实机 |
| A/D 连续横移 | **无任何实测速度** | 不具备连续侧移资格 |
| B22 GUI 的 A/D | 未启用 | 无键盘横移交付 |
| R 键 | 仍是仿真复位 | 物理翻倒自救未实现 |
| 可靠越障、提速 | 未完成 | — |

77 份冻结输入、B22 checkpoint、旧实验记录和用户的 `results/my_course_drive_01/` 本轮均未改动，旧训练与旧物理批次未重跑。

## 独立验收与纯检查

独立验收耗时 18.70 s（上限 1,200 s），报告确认记录可审计、属于部分记录、保存的 native 证明**未通过**，未完成要求为 `all4_stop`、`400_B22_retention`、`complete_native_case`；全局账本 `passed: true`，确认 204 控制步、1,021 正常 native、4 个返回侧移控制、5 次侧移 compute。验收不独立重算 Jacobian、偏置力与 `fullM` 耦合惯量，这三项由密封源码、API 作用域计数与缓存身份约束，不是第二套刚体动力学实现。

纯检查累计 2.4507 s：验收器 8 项 pytest、新增 2 项账本 pytest、root/控制器 8 项 unittest、schema unittest 与关键 ruff 通过。旧 `pure_reader35_01` 实际执行零项 unittest，仅导入与 lint 有效，原始回执保留，不计入测试覆盖。

本轮**没有**新的 Astra 终审。`source_go35.json`（SHA256 `36bed6660fafb382e1de8eaabc401efa52ed17659780e9a42060cdcce32aa0ee`）由实际 gpt-6-astra ultra 审签；裁定由 root 依据保存证据写出，记录中显式标注 `astra_final_adjudication_obtained: false`。

## 证据入口

- [C35 完整结果与定位](ad_lateral_pair_feasibility_20261004.md)
- [横移步态方案重审](ad_lateral_gait_review_20261004.md)（零物理步；速度上界、静力可行性与替代机构）
- [A/D 连续速度主方案](main_plan_20261003_ad.md)、[C34 速度补偿结果](ad_lateral_tracking_20261003.md)
- 公开子集：[results/d1_ad_velocity_reference_20261003](../results/d1_ad_velocity_reference_20261003/README.md)
- 本地权威记录：`W/continuation35/root_adjudication35_01.json`、`continuation_state35.json`、`independent_read35_01.json`、`development_01/episode_0/native_contact_failure_0000.json`

## 新窗口启动顺序

```bash
rtk git status --short
rtk git log -1 --format='%H %s'
rtk gh api repos/LYHrmer/wheel-legged-control-lab/commits/main --jq .sha
rtk gh api 'repos/LYHrmer/wheel-legged-control-lab/actions/runs?per_page=3' --jq '.workflow_runs[] | {id,status,conclusion,head_sha}'
rtk pgrep -af python
```

1. 先读本文、`W/continuation35/continuation_state35.json` 和 `root_adjudication35_01.json`，再读 [C35 结果文档](ad_lateral_pair_feasibility_20261004.md)和[步态方案重审](ad_lateral_gait_review_20261004.md)。需要原始设计时读 `W/continuation35/astra_plan/` 下的合同、规格、接口与四份 clarification。
2. 核对工作树、GitHub main、CI、运行进程与 77 份冻结哈希。C35 源码仍冻结；`development_01` 已用尽，不得重试。
3. 把定位结论和重审结论一并交给实际 gpt-6-astra ultra，取得上节两项决策与新 GO，再执行新的有界物理验证。
4. 固定可行性通过后才建立同权限强固定基线，并依据真实结果制定有界 RL 合同。

每条 shell 命令加 `rtk`，原始输出用 `rtk proxy`。GitHub 上传沿用已验证的 SSH443：

```bash
rtk proxy git -c 'core.sshCommand=ssh -p 443 -o HostKeyAlias=github.com -o BatchMode=yes -o StrictHostKeyChecking=yes' push git@ssh.github.com:LYHrmer/wheel-legged-control-lab.git HEAD:refs/heads/main
```

Python 环境沿用 `PYTHONPATH=.local-deps:src:.` 与各 BLAS/OMP 线程数 1；物理运行的完整环境由 `source_go35.json` 的 `runtime_environment` 规定，不要手工拼装。
