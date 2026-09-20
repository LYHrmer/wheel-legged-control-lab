# flat-plane 固定集合续跑补充合同：2026-09-20

制定者：gpt-6-astra / ultra。只读审定；本合同编写与复核未调用物理积分、训练或模型。原 `flat_plane_contract_20260920.md` 和 `flat_plane_01` 全部文件保持不变。本补充只修正跨案例初始观测的 signed-zero 比较，并规定复用已完成两场后执行剩余四场的证据与预算边界。

**决定：条件成立时允许一次有界续跑。** 两场已完成停车案例有效保留；新目录 `flat_plane_02` 只执行原列表中的后四场，新增最多4000控制转移、20000物理子步。累计仍不超过原5600控制转移、28000物理子步；不重跑已完成两场、不重跑hfield、不追加案例、不改控制律或门槛。原两场停车门槛失败继续作为失败报告。

## 1. 已独立复算的事实

从 `flat_plane_01` 直接读取NPZ、JSON和gzip日志，核验完整目录清单，不构造或运行仿真。

| 证据文件（相对本目录） | SHA256 |
|---|---|
| `flat_plane_contract_20260920.md` | `3b1738d9a6651cbb8dc9fdfe7ee39330a7bbde3008b3c0fc8294362dbe601d7e` |
| `flat_plane_01/protocol.json` | `7ae147c3f4f21c76f5ff2152384340289fcf546167f489cfe2b2604fa97a10af` |
| `flat_plane_01/failure_manifest.json` | `b5553c1130b099af46cf22190d921f9364232994d609a4849d650a05516f872d` |
| `flat_plane_01/flat_forward_stop/complete_manifest.json` | `06b9b31ff080ea47162e40029dfce49fc3c323ceb9d5c1e2a007742e4b77fc69` |
| `flat_plane_01/flat_reverse_stop/complete_manifest.json` | `53f611ee864aaa860ac29e16084143adfdd7b14279b75baa6c818584ea48bb02` |
| `flat_plane_runner_attempt01.py`（第一次runner源码快照） | `08470a354bb750179fae1082a92d34f99bb186eda299a95566ea052976841824` |
| `flat_plane_partial_audit.json`（root纯文件审计） | `03fec6b75399c5ad503f0c1fda296a2f3aab968bcd186462afca7c82fc837ae6` |

根 `failure_manifest.json` 的30条记录和两场 `complete_manifest.json` 各12条记录，字节数和SHA全部吻合。两场各有800条原评分trace、800条plane诊断、4000条native记录，所有native调用均返回；各有801项qpos/qvel/observation，形状分别为(801,23)/(801,22)/(801,85)，类型float64/float64/float32。requested/applied action均为(800,8)。各场receipt均记录800完整控制区间、4000物理子步、0部分子步、无执行错误、无归档错误、无非有限值；累计1600/8000。各场实际累计时间7.999999999999341秒是浮点累计结果，不能据此补时长。

唯一根层失败是 `flat_reverse_stop_initial` 的 `observations=false`；qpos、qvel均true。直接比较初态：

- qpos与qvel原始字节完全相同。
- 两份85维float32观测全部有限，逐元素数值精确相同；raw bytes不同的原因仅为索引38、56、58的+0/-0，三项两侧均严格等于零。拷贝后仅将exact-zero写成同dtype的+0，再比较原始字节，结果完全相同。
- 原G1 `command_at_tick` 用 `forward_target_mps * clipped_fraction`；反向target为负、初期fraction为+0，故合法地产生-0。正/反停车的观测索引38在后续初始零命令阶段也可保留该符号，不能因此改写原命令或整段轨迹。原G1没有要求不同命令案例跨场初态哈希一致。
- 两场plane法向水平分量最大值均为0，非轮有效接触峰值0；各有4个无轮几何接触子步，不能把这些解释为正载荷接触。

原停车结果不变：正向late speed=0.08435577965664166m/s、late path=0.06299257316388528m、raw velocity RMSE=0.0590425138262486m/s；反向分别为0.08423582071177545m/s、0.0628194790371013m、0.05877599252704661m/s。两场均失败原 `late_stop_speed`、`late_stop_planar_path`、`velocity_rmse` 三项。碰撞表示正确没有解决当前停车控制问题。

## 2. 唯一比较规则修正

仅对“不同case reset后的initial observation”新增等价判定；不修改env、编码器、原始observation、NPZ、原initial_state_observation_sha256、raw命令、动作或评分。

1. 首先检查双方shape、dtype完全一致，85维float32且全部有限；不接受广播、dtype转换或NaN相等。
2. 同时记录 `raw_bitwise_equal`、逐项 `numeric_different_indices`、`signed_zero_difference_indices`。正负零例外须同时满足两侧值严格等于0且signbit不同。
3. 仅在独立拷贝上用精确 `value == 0` 掩码将零赋值为同dtype的+0，然后沿用原逐位比较。任意非零位模式保持原样；不得使用allclose、epsilon、round、abs整列或近零阈值。任何非零差异仍失败。
4. qpos、qvel以及其余状态、动作、PI和扭矩保持原始逐位规则。原比较失败保留；新报告注明 `comparison_rule=initial_observation_exact_zero_canonicalization_v1`，并保留raw失败和规范化后结论。
5. **现有同命令前缀检查不放宽。** 正/反停车不在同命令整段配对中；左右转向tick200以前yaw由原G1分支统一产生+0，两项forward impulse与forward stop的共同命令target同为+.25，故没有证据要求放宽这些前缀。若剩余场出现真实逐位差异，按原合同停止，不临场把规范化扩散到全部观测或全部数组。

最低非物理测试：+0/-0初态例子通过并记录索引；1个float32非零最小位变化失败；近零非零值失败；shape/dtype/NaN差异失败；qpos/qvel的正负零字节差异仍失败；原prefix检查仍能拒绝任意观测字节差异和不足长度。测试不得调用mj_step。

## 3. 已完成前缀的复用资格

新增明确 `--completed-prefix` 参数只接受本次 `flat_plane_01` 的两个完整首案例；这是跨案例实验集合续接，不是从第800步物理状态继续积分。剩余每个case仍按原G1独立reset、seed55101和原完整命令执行。

全部资格检查必须在任何新物理之前完成：

1. 逐项验证以上固定protocol/failure manifest/两个complete manifest的SHA和其全部文件bytes/SHA。只读原目录；不得修补、删除或覆盖原失败收据。拒绝缺文件、不完整/损坏gzip、额外部分物理案例或不符账本的来源。
2. 来源必须正好是原列表首两场，各800完整控制转移和4000物理子步，T+1状态及T诊断长度正确；native全部returned，receipt无error、archival_error、nonfinite或partial；plane语义有效。**不要求性能gates通过**，失败结果必须保留。
3. 来源根failure与实际初态复算必须符合上节唯一signed-zero原因。不能把任意批次失败、source changed、数值异常或缺失评分的案例当成可复用完整前缀。
4. 原G1 rev2协议SHA、全部case字典、顺序、seed、原gates以及77份冻结输入不变。来源与新运行的plant/control、d1_flat_plane_env、native/archive/endpoint诊断及其依赖源码SHA必须一致。仅允许本runner的初态比较、只读复用、记账和归档逻辑变更；记录runner旧新SHA并审核差异，不能用“runner例外”放行其中PlaneDiagnosticEnv、step、实际执行或评分行为的变化。
5. 原nonphysical preflight及其input_sha256仍全部有效；本次runner规则修正另有非物理测试/审查记录。若plant或物理输入变更导致原前检失效，本续跑合同不适用。
6. 复用完整原summary/plane_summary、states和plane诊断中的requested/applied torque与PI数组，保持原dtype、shape及原始字节。后续forward impulse与forward stop的共同前缀使用来源forward_stop的原数组，不能只加载summary后跳过跨目录配对。初态PI仍为四个0，原metadata/task/collision身份不变。

## 4. 剩余固定集合与账本

| 案例 | 来源 | 本次新增控制步上限 | 本次新增物理子步上限 |
|---|---|---:|---:|
| flat_forward_stop | 只读复用_01 | 0 | 0 |
| flat_reverse_stop | 只读复用_01 | 0 | 0 |
| stationary_turn_left_hold | _02原case独立reset | 800 | 4000 |
| stationary_turn_right_hold | _02原case独立reset | 800 | 4000 |
| forward_positive_yaw_impulse | _02原case独立reset | 1200 | 6000 |
| forward_negative_yaw_impulse | _02原case独立reset | 1200 | 6000 |

本次新增最大4000/20000，来源已经消耗1600/8000，全实验累计最大5600/28000。不能将复用两场作为本次“执行了1600步”，也不能只报告4000步而隐去原消耗；summary必须同时列出 `reused_completed_*`、`new_executed_*` 和 `combined_*`，案例条目标注实际来源目录/manifest SHA及是否本次执行。发生早停或异常时按真实native/clock收据记账，不把上限当实际数，不补齐、不重试。

原G1 raw评分、停令tick400、所有窗口和门槛完全保持；所有case零残差、无release shaping、无stop damper、内部yaw limit=.6。已知性能失败可以继续原固定后四场；源/日志/模型不变量、配对或数值失败停止剩余物理。

## 5. 新协议、来源链与最终归档

1. 用exclusive mkdir创建全新_02，在新物理前写新protocol：保留原六场整体定义与5600/28000总上限，同时明确source两场和本次只执行后四场、本次4000/20000上限、原合同与本补充合同SHA、来源protocol/failure/complete manifests SHA、runner旧新SHA与当前全部输入SHA。
2. 在新目录单独写复用审核收据和修正后的initial比较结果，原_01 failure仍如实保留。来源文件通过路径与内容SHA引用；可把完整原case目录逐文件原样复制到新目录，供统一后处理，但必须明确标为 `carried/reused_without_physics`、核验复制文件SHA并记录来源manifest。不能重新生成评分/轨迹、修改来源或把复制品伪装成本次新跑。
3. 原前缀边界不变：left/right比较state0..200、obs/action/torque/PI0..199；forward stop与两impulse比较state0..400、obs/action/torque/PI0..399；两impulse比较state/obs0..600、action/torque/PI0..599。所有最低长度检查保留，跨来源配对不得跳过。
4. 六场合并报告分别给plant_semantics_valid、execution_evidence_valid、各原gates结果。_02的汇总可以认定按明确补充规则修正了_01跨场初态检查，但不能抹去历史raw-bitwise失败或宣称六场均在_02执行。
5. 结束时再次检查所有来源文件和来源manifest本身SHA、全部冻结输入以及本次代码/协议输入SHA；即使失败，也写本次及累计消耗账本、失败阶段和partial状态。所有新文件关闭后写最终manifest；manifest只描述本目录文件，外部来源链由独立SHA引用完整保留。

这次修改只纠正证据比较规则及实验续接。停车、转向、跳台阶与提速的后续资格仍服从原合同；不得用本次续跑顺便启动停车阻尼候选或扩大物理集合。
