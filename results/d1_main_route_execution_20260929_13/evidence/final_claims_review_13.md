# 阶段 13 最终声明只读审阅

**结论：GO。** 下列执行、读回、性能和来源声明与保存证据一致，无阻断性声明问题。此结论仅审阅已完成的 seed 88813、平地 0.6 m/s、600-control 固定开发脚本配对，不授权再次运行或后续预算。

本次只读取指定汇总、独立读回、最终文档及小型收据，并查看既有 reader 中对应比较字段；未运行物理、机器人模型、测试、reader 或 GB 级依赖哈希，只新建本文件。公开包装配后的文件闭包由 root 的 `verify_package.py` 检查另行负责，本审阅不冒充该检查。

## 逐项判定

| 声明 | 判定 | 已核对的证据与边界 |
|---|---|---|
| 控制与原生调用计数 | GO | 两臂的 worker/case 收据均为 600 attempted/completed controls、3000 attempted/returned/checked normal native、2 compiler native。合计 1200 controls、6000 normal native、4 compiler native，与 `execution_summary_13.json` 一致。C 末状态 phase/target 为零、violations 为零；没有新增旧任务重跑或训练声明。 |
| 策略入口计数 | GO | 每臂 strict load 1、32 行 probe 1 批、正式 predict 600。入口账为每臂 predict attempted/returned 601，合计 1202，包括两次批量 probe；不能把 32 行 probe 写成 32 次入口。learn/save 为零，未执行 optimizer。strict-load 收据内的 1024 optimizer steps 是原训练预算元数据，不是本轮执行次数。 |
| 24 项绿色纯测试 | GO | `archive_green_receipt_01.json` 的 13 项、`gui_bridge_green_receipt_01.json` 的 5 项、`reader_green_receipt_01.json` 的 5 项及 `subreaper_green_receipt_02.json` 的 1 项合计 24，均记录 0 physics controls、0 robot model calls。文档明确保留旧归档红色复现和首次 OS fixture 超时，没有把所有尝试宣称为全绿。 |
| 阶段时限与退出清理 | GO | 两臂 launcher 收据均 exit 0、failure null、完整后检完成、source mismatches 空、worker 已退出且无 orphan；outer host 均确认 readiness/subreaper、exit 0、survivors 空。GUI X11 收据为 passed、child return 0、server stopped、auth removed。实际阶段耗时见下表，均满足冻结上限。 |
| 数值等价 | GO | `headless_readback_13_01.json` 与 `pair_readback_13_01.json` 的 execution/numeric integrity 均通过，pair numeric equivalence 为 true，reasons 为空。reader 对 601 个状态、600 步观测/动作/命令/控制器力矩及执行器链、3000 行原生数值与对应 contact/cache 行作了明确比较。两臂状态载荷 SHA 相同；这里的等价不表示所有归档文件字节都相同，含各自时间信息的 `controls.jsonl.gz` SHA 不同不构成数值配对失败。 |
| GUI 性能与截图 | GO | 独立读回报告 active RTF 1.1904158097、实际 active draw FPS 12.4993660、63 次 active draw、poll p95 88.956706 ms、snapshot-age p95 140.198038 ms，分别满足 0.8、8 FPS、250 ms、250 ms 门。三张初始/运动中/最终 PNG 已由 reader 检查摘要、像素非空与对比度、对应 control index。它们是私有软件 Xvfb 的结果，不是硬件输入延迟测量。 |
| 短脚本物理观察 | GO | 保存收据给出最大绝对 roll 0.4291487111°、pitch 1.1179022590°，汇总给出最小端点 clearance 0.4548674724 m；非轮接触为零。两臂 warnings/thread violations/copy rejection 为零。compact guard 明确 `force_sampling_performed=false`、完整接触力资格为 false；不得将这些观察升级为完整接触力资格。 |
| 冻结源数量表述 | GO | `frozen_source_check_13.json` 支持旧冻结清单的 77 项全部匹配。77 指该旧清单，区别于本次每臂 readiness 的 4567 项来源闭包，也不是公开包文件总数。两个 launcher 的完整后检均无来源变化。 |
| Claude 来源措辞 | GO | `claude_session_metadata_13.json` 的成功 identity session 与编码 session 均保存 assistant 元数据标识 `anthropic/claude-opus-5`；成功 identity 项的 `ready_exact=true`。这支持“API 返回的模型标识”措辞，不是仅由 `opus` 别名推断身份。`claude_provenance_13.json` 区分首次 turn 上限、恢复后 402 日支出上限及未执行/未验收草稿；实际采用的是 Sol 的 reader，与绿色测试及本次读回一致。早期 `claude_identity_01/receipt.json` 为另一次超时失败，不能作为成功探测依据；新的去敏元数据已明确区分，两份记录可并存。无需公开私有原始对话。 |
| 资格与后续工作边界 | GO | 两份说明均将通过范围限定为此 600-control 开发脚本，明确 `qualified_for_default_GUI=false`、RL gain 未证明、compact force 未资格、未重做原 1600-control 任务；未声明自由键盘、硬件输入延迟、1.6 m/s GUI、多地形、A/D 侧移或 Space 跳跃验收。R 为仿真复位，不能称物理自救。下一项 value/梯度诊断仅为待另立合同的提案，未执行或预留。 |

## 实际阶段时间

| 阶段 | headless | GUI | 冻结上限 |
|---|---:|---:|---:|
| 纯预检 | 47.13852 s | 127.67400 s | 240 s |
| 执行与封存 | 12.23500 s | 11.48692 s | 95 s soft / 115 s close / 120 s hard |
| 全 SHA 后检 | 22.55875 s | 57.91589 s | 180 s |
| outer 完整 wall | 82.10385 s | 197.46489 s | 600 s |

headless active wall/RTF 为 4.319944145 s / 1.3889068466，GUI 为 5.040255756 s / 1.1904157825。worker 与 reader 的 GUI RTF 末位微差来自各自保存的高分辨率时钟边界，不影响门限或文档的五位小数。完整 wall 包括来源检查及清理，不能用 active RTF 表示启动速度。

## 本次审阅绑定

下列小型文件在审阅时的实际 SHA-256 已核对。`C` 为本文件所在 continuation13，`R` 为 `/home/lyh/wheel-legged-control-lab`。

| 文件 | SHA-256 |
|---|---|
| C/execution_summary_13.json | `232c82d4d37fea995a482d1745bb918de7587de01189c6279dbc4053740bfe88` |
| C/pair_readback_13_01.json | `b67c179138cf287439263cbd33c253d5c0f8e940c02f79241020fb13165420c4` |
| C/headless_readback_13_01.json | `89a6f9b682792a32d475469e73cef44a83de8291c577c2873c692258d62d7d8e` |
| C/claude_provenance_13.json | `68c70d10b8c398678fa97834fd34faece93b745511ea04b78dbbaefd75b5ceb2` |
| C/claude_session_metadata_13.json | `5f3afcc1bbe1066fd6e26b0e35d03a512ed20e3f29acf862b73358930083c3fd` |
| R/docs/main_route_execution_20260929_13.md | `634efffadcd8c968d0edd955b810efd2c27f3dd39616e33f4aa0bc16ba0ca40e` |
| R/results/d1_main_route_execution_20260929_13/README.md | `3e393058061c85ec97e84914329b68e4f615a1324ef385cd932e082cc94ce30e` |

最终判定分别为：执行计数 GO；绿色测试计数 GO；phase/cleanup GO；numeric equivalence GO；GUI 性能 GO；范围界限 GO；Claude 来源措辞 GO。没有剩余阻断性声明问题。
