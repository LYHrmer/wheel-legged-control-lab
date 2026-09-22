# 单台块首次物理前：评分与记录链代码审查

实际 gpt-6-astra ultra，2026-09-22。仅源码只读审查与SHA读取；没有实例化环境、调用MuJoCo、运行测试或物理。本文件exclusive-create。已知root正在修复runner的`last_transition.torque_nm`为`requested_torque_nm`，本文不把这项已在处理的缺陷重复当新发现。StreamingNativeObserver当前版本已让prefix后检实际可达。

审查快照SHA256：

- scripts/d1_single_step_scoring.py：aff743d777c5cdf0e859a12c41d9f12d05cd8b292e52a6045f4857cc825ff277
- scripts/probe_d1_single_step.py：250d5a59eb9945901d633552a4d55463a2a20a254625c44ec39e5861bdffb5b8
- scripts/d1_single_step_records.py：8722ff45b60607b1ccb5897523861187045681a3a85cad6fbe4527b8abb7ee40
- scripts/d1_single_step_env.py：8aa0d400ed391f90bbd038a044f7fb91329cb2c1e5ccc08763a2348df0b64318

结论：**当前快照不应开始唯一两场物理。** 原native计步、窗口及原命令来源整体方向正确，但scorer尚不能独立证明它宣称的记录真实性；异常归档也存在可丢终端收据的路径。下面是具体实现缺陷，不建议更换物理或控制假设。

## 阻断项1：载荷和box资格未由原接触一致性确认

位置：scoring.py:260–291、328–339、456–474、500–503。

- `normal_load_n`不必等于`local_force_torque[0]`，`active`不必等于`efc_address>=0`。
- `is_box`仅由`box_feature is not None`推断，未验证terrain geom ID/name/body与compiled box一致；plane contact加一个feature即可被当box。
- `frame_orthonormal`、`plane_normal_valid`、`box_feature.geometric_support_valid`是直接信任的布尔，未从frame/normal/位置重新审计。
- `wheel_positive_normal_load_n`及`wheel_box_positive_normal_load_n`未与逐contact按wheel求和核对，`nonwheel_terrain_contacts`、`geometric_box_contact`也未核对。最终0.95占比完全使用可独立伪造的汇总数组。
- `array_positive and detail`不验证是同一轮或同一载荷；某一轮汇总正数与另一个轮的detail足以取得资格。某个native行有box正载荷但无detail仅append reason，**record_valid与task_passed不因此变False**；其它行只要存在一次detail便可能仍通过。

最小修复：让payload带root从compiled preflight冻结的geom/body/wheel/terrain身份映射。逐contact验证两个geom与body/name绑定、terrain唯一归属、robot/wheel归属、active/efc一致、normal_load与local[0]一致。重算法向方向和frame正交、normal/world force，并以相同正载荷定义重算四轮全部载荷、box载荷、nonwheel和geometric_box。随后要求所有保存汇总与重算一致；不一致属于record invalid，不是仅warning。box qualification由身份正确的active wheel-box且原local normal force>0判定。False条件出现任何box身份/feature/positive box load须record invalid。

供root纯测试的反例：空contact+四轮汇总全1；local[0]=0但normal_load=1；plane geom加top feature；wheel0 detail但wheel1 box汇总；非轮contact存在而nonwheel汇总0；frame全0但orthonormal=True。当前代码均存在未被相应真实性检查拦截的路径。这里是静态逻辑反例，未运行测试。

## 阻断项2：全部collision geom通过缺少集合完整性

位置：scoring.py:172–193、516–532。

`_parse_bounds`允许任意非空identity列表、重复identity或同一wheel重复，而最终只要求集合wheel={0,1,2,3}。删除所有非轮collision geom，仅留四轮越线，即可被当作“全部机器人collision geom已通过”。也可以将margin改0或交换wheel标签，scorer不会核对compiled binding。

最小修复：用上述preflight身份manifest的完整robot collision集合（当前原模型42个）作为expected set，不只硬编码42；每endpoint要求identity唯一且exact set相同，geom_id/body/type/wheel/margin与绑定一致，visual不得混入。前后端点不得自行改变集合。不要只以首endpoint集合为真值，因为首endpoint也可能已漏项。保留现有严格min_x > far_edge + margin。

## 阻断项3：endpoint与实际native状态脱节

位置：scoring.py:196–214、385–399、413–419、487–513；runner.py:141–144/147–155。

native内部qpos/qvel链被检查，但端点pos/rpy从未与该链对应；也没有审计`states.npz`中的qpos/qvel。修改endpoint1101..1200的z、vx或COMvz即可让末段指标“通过”，即使native states对应完全不同状态。初始x0/y0/yaw0也直接由endpoint字段决定，影响远侧线和全程误差参考。`com_vz_mps`甚至可以与同一行`com_velocity_mps[2]`冲突。

最小修复：endpoint0应与native0.qpos_before/qvel_before及reset数组一致；endpointk应与native[5*k-1].qpos_returned/qvel_returned一致。纯scorer可重算qpos[:3]与quaternion rpy；COMvz必须等于已存COM vector第2项并绑定scratch reconstruction输入state hash；body vx需与相同状态/既有truth receipt一致。root可在零积分kinematics离线审计中验证COM/完整bounds来源，scorer接收该明确已核验身份，不能用一个无来源通用True替代所有检验。状态流与controller torque、native实际ctrl的对应关系也应核验（当前名义ideal actuator每个interval的5个ctrl都应与其已保护request一致，或使用实际actuator trace解释）。

## 阻断项4：记录有效性与partial计数规则不完整

位置：scoring.py:362–410、294–300；runner.py:108–109、123–140。

A. scorer先将record_valid=True，再对native数量不等于5*T仅添加failure reason。因此任意多余native（包括>6000）或缺少native仍可能record_valid=True。完整1200+6000的task门槛会拦部分task success，但无法满足独立“记录有效”声明，runner也以record_valid决定是否消费第二场。

B. 另一方面任何`payload.error`都会在读archive前立即record invalid，任何native returned=False也立即拒绝；这无法区分“有限且如实保存的物理/控制中断”与“记录破坏/非有限/未知力”。普通物理termination且T<1200、N=5*T当前可以记录有效，这是正确的；partial suffix应有显式状态而不是随意放行或一概等同archive损坏。

C. runner在env.step返回后立刻`completed +=1`，随后任何controller字段/序列化/endpoint失败会造成已消耗5个native却trace/endpoints尚未完整写入。应记录physical_interval_completed和published_trace_completed各自计数；不可重标为已发布完整T，也不可用上一个controller/endpoint补齐。

最小规则：无错误的completed-control archive严格N=5*T、endpoints=T+1、trace=T、N<=6000；提前正常terminated可record_valid=True/task=False。错误archive需要receipt中的attempted/returned/advanced以及末尾partial interval声明：已发布完整prefix严格配对；随后最多一个partial control尾部，native失败attempt只允许末尾一次，保留其实际dt及post-error状态，不允许后续重试。记录真实性、partial完整性、任务complete分字段。缺接触数据、非有限、未知外力、无法解释dt/identity等仍record invalid。若本次选择只把成功published prefix用于评估，也必须另存partial archive状态，不能称其没有消耗native。

## 阻断项5：异常归档可能丢失最终状态及终端收据

位置：runner.py:70–90、127–140；records.py:120–130、143–152。

- env/scratch/observer构造发生在run_case的try之前，构造异常没有case receipt。
- finally依次env.close、np.savez、write command_records，再写receipt；任意一个secondary异常会跳过mandatory receipt并掩盖primary错误。
- `states.npz`仅包含snapshot已发布端点。失败在1..4 native或env.step内部完成全部5native但未返回时，最后integrator状态不会进入此文件。NativeObserver通常会保留最后native状态，但没有统一final_state receipt，ctrl/warmstart/act与partial状态也需清楚归档。
- observer native writer失败可能替换原native/contact-reader错误；__exit__顺序关闭stream/events，前者抛错可能跳过events关闭。父NativeStepObserver仅把delegate异常写entry.error；contact_reader异常时该entry可能returned=True/error=None且无contacts，必须用独立recording_error说明。

最小修复：用外层owned-output标记+初始化为None的env/observer覆盖构造到关闭；except保存primary_error，finally先复制有限/非有限均保真的最终integrator qpos/qvel/ctrl/time/warmstart/act及所有预算计数。各归档/close分别try并累计archival_errors，用最外层finally尽力写case/batch terminal receipt；原错误不覆盖。不中断已有native日志保存，不填数据、不重试物理。写盘系统性失败仍如实判记录无效；目标是任何可写的路径尽量保留收据，而非声称能克服磁盘不可写。

## 阻断项6：source/phase/finite审计存在未覆盖字段

位置：runner.py:147–155，scoring.py:260–290、294–348、359–366。

`load_case`无条件设置source_identity_valid=True，未验证protocol、合同、manifest或episode source身份。若作为offline已审计入口使用会把未验证资料冒充有效。运行时main:190的输入manifest比较有价值，但应把该结果与最终archive manifest/episode schema/preflight hash一起供load_case验证；load_case无验证时应返回unknown/False，不要填True。

Native与contact关键相位字段未审：`contact_sample_tag`与`contacts.sampling`被忽略；distance/includemargin/friction、terrain->robot normal/world force、warmstart/act可能非有限仍不影响record_valid。仅验证部分字段有限不能声称所有已记录关键状态有限。至少对schema要求的所有numeric物理字段及dtype/shape作finite验证，验证native采样标签exact matches；returned=True但error!=None或contacts缺失须有明确不一致/记录错误分类。未知geom/terrain不得仅靠summary布尔排除。

## 已确认正确或不应随意改动的部分

1. **raw .2精确比较不是当前runner的浮点bug。** `d1_single_step_env.py:35/53`使用Python float0.2，`:150`直接放入command_records，runner:104只复制dict；JSON roundtrip保存同一个double。float32仅用于action/observation，不是这个字段。因此scoring.py:233 exact equality应保留。若纯fixture从float32 observation反解命令产生0.20000000298，应修fixture数据源，而非增加raw schedule宽容度。world height与ground/clearance同理沿相同Python float算术；`_prepare`已经要求重建world height exact .455。
2. **最终窗口准确。** endpoints[1101:1201]是恰好100个端点；native[5500:6000]是最后500个native。每轮`mean(load>0)`每native只计一次，多contact不重复增加occupancy。需修的是汇总载荷真实性，不是窗口/0.95阈值。
3. **原生接触相位路径正确。** 原NativeStepObserver在delegate返回后立刻复制qpos/qvel，再读取仍保留的solver contact cache；不调用forward。之后原D1Plant才同步独立measurement data。raw row同存returned state与native solver cache时要保留两种phase标签，不应声称二者同相。
4. **global及case预算是实际delegate前检查。** PhysicsCallLedger只允许绑定model/data；split/bulk被拒；StreamingNativeObserver在6000后拒绝，root loop最多1200控制一次、累计2400。真实调用在construction前有外层ledger，静态forward单列；本次预检报告0physics不受影响。
5. **prefix后检已修复可达。** records.py:147调用super后无try内return；:153–162比较首次box几何接触之前的returned state。接触所在native的pre-state仍被比较，post差异允许。继续保留原接触定义；不要等到positive force才停止prefix比较。
6. root已识别runner:116读取不存在torque_nm；应用修复后必须把新source hash纳入preflight，否则manifest会正确拒绝物理启动。

## root应添加的最小纯检查（本代理未执行）

- 上述原始contact/aggregates真假交叉反例，去掉/重命名一个collision bound、改margin、伪造ep1110.z或COMvz。
- N=5*T-1、5*T+1、6001的无partial声明archive必须invalid；普通提前terminated的完整prefix仍valid但task=False；partial尾部有独立receipt和state。
- 严格raw Python .2和JSON .2通过；float32反解raw或.200001失败。k199/200/999/1000、ep1100/1101、native5499/5500的对比确保原边界不变。
- fake env.step推进0/1/4/5个假native后失败、snapshot/serializer/close失败，各自仍尽力写terminal receipts并保留final state、原错误和archival errors；不能运行真实delegate。
- source_identity和phase错误、NaN normal/warmstart/local wrench即invalid。载荷正占比严格采用原始active contact normal scalar，不用world Fz。

以上修复均属归档/评分真实性；不添加新的物理场次、阈值、控制增益或训练。修复后root完成纯测试和source manifest闭环才进入既有唯一2400/12000合同。
