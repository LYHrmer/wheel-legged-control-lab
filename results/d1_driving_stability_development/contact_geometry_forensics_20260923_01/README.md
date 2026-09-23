# 真实 15mm 台阶：记录忠实性与接触几何的独立诊断

本包延续“稳定直行 → 可靠越障 → 提高速度”。已完成纯离线诊断，原台阶结果保持未资格化；没有新 RL 训练或重跑旧物理批次。

## 已完成的离线合同

实际 GPT-6-astra ultra 负责方案、指挥与独立复审；实际 GPT-6-sol 接替不可用的 Claude 完成有界实现与测试代码；主代理执行测试、完整扫描、身份核验与计数。12 项真实夹具案例在第一轮全部通过，两遍完整存档扫描的三份结果逐字节一致，完整扫描预算 2/2 已耗尽。本阶段没有 MuJoCo 导入尝试、构造、静态调用或动态积分。

| 核验项 | 平面 | 真实 15mm 单台阶 |
| --- | ---: | ---: |
| 原生记录 / 控制端点 / 控制区间 | 6000 / 1201 / 1200 | 6000 / 1201 / 1200 |
| 接触候选记录 | 45430 | 46387 |
| 原始记录链路忠实 | 是 | 是 |
| 所有候选几何合格 | 是 | 否，1 条异常 |
| 箱体候选 / 正载荷候选 | 0 / 0 | 11053 / 11032 |
| 原冻结 task_passed | true | false |

台阶唯一几何异常仍是 native 3486 / contact 5，后左轮 `RL_foot#geom1` 对箱体。原始 frame、位置、间距、active 身份与六维零载荷逐值保存；box→wheel 法向 z≈−0.999991，箱顶支持锥残差 1.0，原阈值仍为 0.0021。其记录忠实不意味着几何合格。11032 条正载荷箱体候选均符合原几何规则，只是单独观察，不能替换所有候选门。

纯解析圆柱支持点位于箱顶上方约 0.676532493 mm，XY 位于箱体 footprint 内；与已保存的引擎距离相差约 4.83e−10 m。没有再次调用引擎求距离。原台阶评分仍是 `record_valid=false`、`task_passed=false`，所有补充报告的 `original_score_overridden`、`qualification_granted`、`rl_gate_open` 均为 false。

平面诊断的 `positive_load_candidate_geometry_valid=false` 表示没有箱体候选，此字段只针对箱体；不改变平面的原成功评分。

## 可复核文件

- `full_diagnostic_pass_01/` 与 `full_diagnostic_pass_02/`：完整结果和解析支持点；receipt 只在 pass 计数上不同。
- `provenance/diagnostic_source_freeze_01.json`：四份新代码/真实夹具冻结身份。
- `provenance/zero_call_contract_closure_01.json`：独立审阅、两遍比较和零调用阶段闭合。
- `provenance/pure_tests_round_01.xml`：12 项纯案例结果。
- `provenance/upstream_source_audit_01.md` 与 `sources/`：官方 MuJoCo 3.12.0 源码与来源散列。
- `contracts/zero_call_diagnostic_contract.md`：已完成的合同原文。

原始两场物理存档继续保存在相邻 `single_step_readiness_01/physical/`，本包不复制或改写原分数。真实故障夹具为仓库 `tests/fixtures/d1_single_step_native_normal_anomaly.json`。77 冻结文件、前轮 215 项输入、本轮物理 193 项输入均保持原哈希。

## 已阻断的微型模型方案

`next_engine_kernel_plan_01/next_contract.md` 从未激活。源码复审发现 XML 编译器内部用临时 data 执行 `mj_step`，即使模型没有自由度，也不满足零动态预算。`kernel_preparation_01/` 仅是无引擎调用的输入准备包；对应 C 草稿入口已封闭。不得运行该 XML 编译路径，也不得把新 data 的 time=0 当成内部未积分的证据。

## 合同 02 的实际静态结果

主代理在 Astra 最终 GO 后编译并冻结 140 项依赖，唯一运行 `scripts/d1_contact_native_ccd_probe.c`。使用固定 MuJoCo 3.12.0 的非公共 `mjc_initCCDObj` / `mjc_ccd` 与源码等价的本地扰动 wrapper；这不是直接调用 `mjc_Convex`，也不是全机器人重放。没有编译或分配仿真模型。

16 次 CCD 尝试全部返回，预算已用尽，无 warning、无重试。3485、3486、3487 的三组最终接触（各 3 条）均与冻结记录的位置、距离和完整 frame 逐位一致。唯一坏候选定位到 **query 7，3486 的 axis0_negative（第一切向轴 −0.001 rad）**。原生记录的 GJK 计数字段为 6、EPA 循环索引字段为 4，`epa_status=0`；这不是迭代上限被耗尽，也不能把状态 0 解释为已证明收敛。

坏法向由原生返回的 `x1−x2` 直接归一化产生。该查询的原距离 `margin+status.dist=0.0006766142700192765 m` 后被 wrapper 替换为主接触距离 `0.000676533074470901 m`；距离覆写没有制造或修正法向。解析检查显示坏 witness 位于膨胀圆柱内部约 0.425368 mm、膨胀箱体内部约 0.003043 mm。具体 EPA 选面、停止条件和 affine 权重分支仍待源级取证，不能据此宣称数值根因已修复。

满足冻结条件后执行了第四组 primary-only，只有 1 条正常主接触。主代理直接对原 fixture 核对其位置、距离、normal 和 frame，均与旧主接触逐位一致。C 中第四组 `reference_equal=false` 仅因该组未启用整组 3 候选比较；独立补充比较见 `provenance/native_ccd_reconciliation_02.json`。这不证明对完整机器人关闭 multiccd 的动态行为。

| 本阶段调用 | 实际数 |
| --- | ---: |
| native CCD 尝试 / 返回 | 16 / 16 |
| descriptor 初始化 | 8 |
| CCD 工作区尺寸查询 | 1 |
| 显式纯数学 helper | 171 |
| version / versionString | 各 1 |
| 模型编译 / 模型或 data 分配 | 0 |
| 控制步 / 原生积分步 / RL 训练 | 0 / 0 / 0 |

`native_ccd_run_02/ccd_events.ndjson` 保存所有尝试、返回、witness/simplex、候选接纳与完整比较。`provenance/native_ccd_budget_closure_02.json` 是最终计数；原始启动 budget 保留为启动时记录，不覆盖。绝不能重新启动这个程序来增加 query 17。

本包没有声称修复引擎或通过越障。可靠越障仍未验证，提高速度、物理自救和新 RL GUI 仍未完成；R 仍是仿真复位。下一步按[已冻结的源码修复合同](next_epa_source_fix_plan_01/next_contract.md)开展局部源码定位与最小修复研究，SHA256 `8fd28ec3e2eec24db0c0b758dad0cc5b97bdf963a73d5c4c2c8a156dae47ee46`。该阶段**尚未执行**：A插桩16次，机制明确后单个B修补候选24次，总计≤40次 source-local CCD；installed mjc_ccd、模型初始化、动态积分和RL均为0预算。旧16/16额度不复用，RL保持关闭。

## 历史计数范围补充

新源码审阅发现编译期隐式step。旧317824/1589120及两场2400/12000继续按原记录窗口引用，不声称覆盖所有编译内部调用；旧记录不回写、不回放追数。详见[计数补充](provenance/historical_counter_scope_addendum_01.md)。本轮纯诊断与无模型CCD阶段均未编译模型，零积分结论不受影响。
