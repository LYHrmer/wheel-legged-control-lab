# B22 半增益完整任务套件35：中止记录（2026-10-18）

35在 RAM 保留线触发后以 `INCOMPLETE` 结束。原现场为 `results/b22_gain_suite35_20261018_01`，失败原文为 `ValueError: C35 RAM admission/runtime reserve exhausted`。这不是完整套件通过，也不代表半增益无效；当前证据不足以给整套任务无退化或RL贡献结论。

## 已知计数与未知部分

整轮 `actual_counts=null`，活跃阶段 `active_stage_actual_counts=null`，两者继续保留。独立重读11个闭合阶段，得到已验证小计 **18,400 controls / 92,000 normal native / 22 compiler native**，与原父报告 `known_actual_counts` 精确相同。

| 已闭合阶段 | 会话数 | controls | normal native | compiler native |
|---|---:|---:|---:|---:|
| true15mm step pair（zero/B） | 1 | 2,400 | 12,000 | 2 |
| flat0.6、flat1.6、yaw1.2、bumps0.4、rough0.35，各zero/B | 10 | 16,000 | 80,000 | 20 |
| 合计 | 11 | 18,400 | 92,000 | 22 |

step pair单会话含两个arm，因此这11个worker会话对应12个逻辑arm记录。11份独立readback均为 `COMPLETE`，数值reader本身没有模型或物理调用。重新核对其283项原始工件引用，所有SHA相符；连同11份readback、父报告、release，共590个闭合归档文件已逐项哈希保存到机器报告。

中止时活跃的是 `ramp_0p45_complete_zero`。存在session、构造证明、进度文件与未闭合的 `.partial` native块，但没有最终worker receipt，也没有该stage的独立readback。进度文件最后显示570 controls，不能据此补造最终native/compile或整轮精确计数。本记录只绑定这些文件的现有SHA，不解包、修复或接续未闭合native块。

`ramp_0p45_complete_B` 与 mirrored-yaw zero/B 未采集。旧partial ramp不进入可复用闭合集合，也不按0成本处理。

## 停止与清理证据

父资源日志952行，最后样本为 elapsed 942.591043s、有效可用内存 **3,770,355,712 B（约3.511 GiB）**，确实低于冻结的4 GiB保留线。前两样本分别为5,049,921,536与5,006,032,896 B。父报告耗时944.678388s，`minimum_effective_bytes`与最后最低样本一致。

这些数据支持资源停止原因；不能识别具体内存占用者，也不能给未采样瞬时内存或进程峰值RSS。关闭时自己的subreaper清理回执 `remaining={}`，清理耗时0.024246s。`retry_permitted=false`、`automatic_model_replacement=false` 保持原值，没有修改旧35源码、工件或补写成功回执。

## 资格边界与后续分账

原完整套件的 `finite_suite_task_qualification_passed`、`RL_contribution_passed`、`no_task_regression` 均为null；gain0.5候选不能靠闭合前缀升级完整资格，旧C22/C24阈值保持不变。

可复用的是11个有最终原始归档和独立readback的完整会话。新37只在独立新输出、来源与预算下采集缺失的4会话，最多6,800 controls / 34,000 normal native /8 compiler native；它既不续写旧35，也不把旧partial ramp当作重跑后的0成本。闭合证据与新37的任务覆盖合计上限25,200 /126,000 /30，仅是验证场景的合计，**不是历史两次尝试的已知总开销**。旧35额外partial开销仍未知；本记录不代记37实际结果。

主要绑定：

| 工件 | SHA256 |
|---|---|
| 35父报告 | `1f2464480c13c57ee7a237f822bd35169adede9daeea77abd91017749cca71a3` |
| 35 release | `c0d09c7e9a969a28987b128d9f393d5a3be09f656d3c28fbcb470db6e3f2e568` |
| step pair readback | `cf5c0188bb6e06dbd096eef6035da1848be6562d8e28f0b8d64f96a85144ced9` |
| 活跃ramp session | `b9d4f92a848ba57cf5a2508d800220678d4493f631a70141054415d412ce6e35` |

完整原始SHA、11阶段计数、资源样本与partial文件身份见 [机器报告](../reports/b22_gain_suite35_interruption_20261018.json)。本次仅读取既有JSON与文件字节，没有加载模型、执行新推理、物理、GUI或测试；未重新扫描原v27的4072项依赖，仅保留原release中的来源闭包。
