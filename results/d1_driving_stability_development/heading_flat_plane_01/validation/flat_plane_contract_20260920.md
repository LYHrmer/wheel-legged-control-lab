# 独立 flat-plane plant 验证合同：2026-09-20

制定者：gpt-6-astra / ultra。状态：已审定的待实现、待执行合同；本文件未运行新物理、训练或控制实现。实际 Opus 负责新增有界模块，root 独立集成、审查和记录。只改变平坦地面的碰撞表示，保持零残差和原控制律。固定最多 **6场、5600控制步、28000物理子步**；不重跑24场G1、不训练、不追加重复物理。

## 决定与证据边界

继续暂停固定b腿阻尼的8k资格比较。先用原零残差控制器在显式版本化的原生 plane 上重建平地基线。

旧 `D1LocomotionTerrainConfig(layout="flat")` 仍编译121×241的全零heightfield；保存的转向和停车姿态在该hfield上出现近水平接触法向，而相同姿态、相同z=0的原生plane法向全部竖直。转向日志中部分异常法向有实际载荷；停车没有历史逐接触力日志，不能将几何差异直接解释为全部停车回摆的原因，更不能据此宣称plane已经让停车通过。

本合同引用诊断，不重复根因报告：

| 文件（相对此工作目录） | SHA256 |
|---|---|
| `turn_contact_diagnosis_01/collision_report.json` | `e9a4f226521f988f1f339542c02042b52e77aabf7cd6924e97abdd6456444c8f` |
| `turn_contact_diagnosis_01/stop_collision_report.json` | `cede4bee5e29bdbdea909db79e75fcc1ace64e499825dba26a64927238299db4` |
| `stop_kinematic_diagnosis_01/report.json` | `4bd0303665b5ee2c41d818b8b0237bd5e99732d348fd576a20e2c185a9190016` |

旧hfield、release和turn-limit结果保留为描述性对照。新plane轨迹不应与旧hfield逐位相等，不能以这种差异判实验失败，也不能更改旧sidecar或schema冒充同一个plant。

## 唯一实验变量和最窄实现接口

建议新增 `scripts/d1_flat_plane_heading_env.py`，包含以下两个窄接口；名字可按仓库风格调整，行为和身份不可省略。

**`D1FlatPlanePlant(D1Plant)`**：

1. 只接受 `arena="flat"`、无training terrain、合法的flat配置；非flat layout或任何非零坡度、起伏、台阶、phase参数必须在构造前拒绝。不要以“采样高度恰好全零”放行非flat布局。仍校验有限实数，拒绝bool/nonfinite坐标。
2. 调用冻结 `D1Plant.__init__`，固定传入 `arena="flat", locomotion_terrain=None, training_terrain=None, sampling_mode="synchronized"`。冻结 `build_d1_model` 在这种参数下已经构造 `floor` 原生plane，不必复制URDF/model builder，也不必改动已编译model。
3. 只覆写 `locomotion_ground_reference(x,y)`：在原有限查询域 `|x|<=6, |y|<=3` 内返回 `TrainingGroundReference(height_m=0, pitch_rad=0, roll_rad=0)`；边界外抛错，不把无限plane等同于无限任务地图。调用前验证floor仍是静态worldbody的z=0、单位姿态plane，或在构造及每次reset前后验证该不变量。
4. 原 `D1Plant.step/reset/set_domain`、actuator channel和同步measurement data全部复用；不复制物理循环，不插入额外settle积分。`self.locomotion_terrain`应如实保持None或另用明确命名的请求配置字段，不能暗示model仍含hfield。

**`D1FlatPlaneHeadingEnv(D1HeadingTrackingEnv)`**：

1. 仅支持原 `wheel_leg` / `independent8` / oracle，原默认 `D1WheelLegControlConfig()`；前向/偏航/腿/姿态增益以及内部yaw限幅0.6保持原值。不得接入release governor、stop damper、1.0 yaw limit、动作滤波或任何PPO。
2. 在 `super().__init__` 后、首次 `reset` 前，用上面的plane plant替换尚未执行的 `self.plant`。必须确认旧plant时间0、loop和decision尚为None、env inactive；这是仅对wheel_leg成立的接缝，因为该原controller不持有env.plant。不扩展到LQR/MPC。构造期间暂时编译一个没有执行过的旧hfield plant可以接受，不计为物理步。
3. 后续原 `_configure_dynamics()` 自行在新plant上重建domain/channel/provider/loop；不复用指向旧plant的provider、loop或MjData。不改全局D1Plant/model factory，不做运行中geom_type/geom_dataid替换。
4. `step`只接受8维有限且逐元素严格为零的残差；在调用super前拒绝其他值。运行入口无checkpoint参数、无policy loader分支。prepare/reset/step原时序、85维float32编码和8维动作结构不改。
5. 原heading环境的外环kp=2、kd=.4、limit=1.0仍保持；这与轮腿controller内部effective-yaw限幅0.6是两个不同层次，不能趁plant改动混为一个旋钮。

## 身份与旧checkpoint拒绝

新增明确身份，例如：

```text
plant_schema   = d1-native-flat-plane-plant-v1
terrain_schema = d1-finite-query-flat-plane-z0-v1
task_schema    = d1-heading-flat-plane-zero-diagnostic-v1
```

原controller_schema、source_schema、observation_schema保持原编码/控制含义，不能为了碰撞变更假称更换控制律。新env顶层task_schema与 `heading_task_config` 必须同时记录新身份；后者新增 `plant`/`terrain_backend` 身份和 `zero_residual_only=true`。注意原 `_build_heading_task_config` 内使用常量HEADING_TASK_SCHEMA，单改class属性不够：新覆写返回的配置必须显式同步新task_schema。`load_heading_policy`须在反序列化前因config不兼容而拒绝旧85维checkpoint；原 `load_locomotion_policy` 的顶层task_schema检查也必须拒绝旧任务。用旧metadata加不存在的模型路径作非物理测试即可，不实际加载模型。

原 `self.terrain` 可保留合法flat请求选择器供冻结构造/父类流程使用，但episode_metadata不能只留下旧 `d1-fixed-2d-road-v1`：在新reset包装中把实际 `terrain` 改为新plane身份与z=0/查询域记录，把原选择器放在单独 `requested_terrain_config` 字段。同步内部 `_episode_metadata`、reset返回的info和 `episode_metadata` property，保留heading reference初值。protocol同时记实际geometry类型、MuJoCo版本、源码/model构建输入SHA和本合同SHA。

## 固定初态、接触与求解参数

- spawn仍为 `(-3.8,0,.455)`，base quaternion仍为 `(1,0,0,0)`，16关节仍为 `NOMINAL_JOINT_POSITION`，qvel=0、qacc_warmstart=0，控制积分/reset状态与G1相同，seed仍55101。
- plane实际表面固定world z=0；不抬高或降低plane，不以0.5–2mm“接触修正”改变base高度，不修改轮半径.087、轮几何、joint target或clearance=.455。
- 原机器人碰撞geom margin=.001保持；floor自己的margin/gap等值从冻结builder原样继承，不能把机器人margin错误套给floor。原floor/机器人friction三元组、condim、solref、solimp、solmix、priority、contype/conaffinity均核对不变。名义ground friction=.9，三元组为(.9,.005,.0001)；全部domain scale为1，无延迟/噪声随机化。
- timestep=.002，control_dt=.01，每区间5个native子步；原IMPLICITFAST、NEWTON、iterations=20、ls_iterations=5及其余solver/contact选项不变。
- base/关节/执行器质量、惯量、armature、被动阻尼、frictionloss、扭矩/速度/位置限幅与旧模型逐项核对。保留80Nm腿/12Nm轮额定扭矩。
- 保留原任务安全边界 `|x|<5.3, |y|<2.3` 及有限map查询/边界终止。plane可在几何上无限延伸，任务域不能因此放宽。
- 相同初态的接触数量、distance、力和observation允许随碰撞表示改变；不得为了匹配旧接触数、初始观测或预期成绩调整姿态/高度。原案例内settle阶段照常计入物理预算和评分，没有额外未记账的预热。

## 执行前的非物理检查

所有检查先完成、归档，再启动唯一物理批次。构造、reset及运动学/碰撞cache更新不计积分，但必须assert所有data.time=0且没有调用mj_step；不要把真实步进“smoke test”藏在单元测试里。

1. 核对77份冻结输入和原G1 rev2协议SHA `cf5dbd51042bfafa163116f4a7fb2a990726d47c5a6000558f28aa6866a664fc`。记录新增文件、依赖和诊断证据SHA。
2. 用冻结builder分别构造旧flat-hfield与原生plane，不执行物理。断言同nq/nv/nu、相同joint/actuator/body身份；除floor的geometry表示/相关asset及其派生几何缓存外，以上机器人、执行器、接触材质、求解参数等必须相等。不要要求nhfield/nmesh asset索引或floor几何缓存不变。
3. plane floor必须为worldbody静态plane、geom_dataid为无hfield、pos=(0,0,0)、quat单位；场景只有预期地面，不叠加旧hfield或隐藏碰撞平面。复核z=0接地、有限查询域，含域边界和NaN/Inf/bool输入拒绝。
4. 对已保存的37个诊断姿态（9转向+28停车）可复用离线碰撞验证，没有积分：所有wheel-floor法向水平分量≤1e−12，|nz|与1误差≤1e−12。按geom顺序将法向统一为地面对轮方向；没有接触的帧不计作“法向正确”，另记数量。不推导未模拟的plane轨迹或力。
5. 核对新env构造/重复prepare/reset不会额外积分或重复消费命令；reset后的plant/provider/loop均指向新对象，reset返回85维有限float32、原8维action空间。查验实际metadata/task/config身份一致。
6. 非zero动作、非flat terrain、course/training terrain、不支持的baseline、旧checkpoint均在执行前失败。control law必须是未修改的原D1WheelLegController。
7. 审查runner部分异常归档：保留已完成T转移和T+1状态、最终物理qpos/qvel/time、当前未写入完整row的native子步/wrench/torque记录及异常；关闭文件后保存manifest，不能只写异常文本。失败区间已有物理时间算入全局预算，不reset重跑凑数。

## 唯一固定物理集合

只运行 **plane / zero_action** 一条件，沿用G1原case字典（命令、seed、duration、外力、gate_groups均按原协议读取）。建议顺序如下，便于先看基础启停，再看转向及原已通过的抗扰能力：

| 顺序 | 原case | 控制步 | 目的 |
|---|---|---:|---|
| 1 | flat_forward_stop | 800 | +.25m/s ramp/hold，tick400原始停令归零 |
| 2 | flat_reverse_stop | 800 | −.25m/s对应停车 |
| 3 | stationary_turn_left_hold | 800 | tick200..249原始yaw +.6rad/s，其后保持 |
| 4 | stationary_turn_right_hold | 800 | 同时序yaw −.6rad/s |
| 5 | forward_positive_yaw_impulse | 1200 | +.25m/s直行，tick600..619原生桥施加+.5Nm |
| 6 | forward_negative_yaw_impulse | 1200 | 同时序施加−.5Nm |

合计5600控制转移、28000个2ms物理子步，56秒仿真；模型选择和训练步数均0。左右转向前2秒零命令同时提供静止/初始落稳观察，不另加站立仿真。±冲击各20个控制区间，实际角冲量应为±.1N·m·s。不能通过被plant逐子步清除的xfrc直接写入冒充扰动，复用已验证的原 `plant.step(push_torque_world_nm=...)` 桥及native入口收据。

首次termination/truncation结束该case；不得补时长。性能门槛失败可以按固定列表继续后续case，不能临场改变命令、增益或运行更多case。日志/协议/plant不变量失败或数值异常则停止剩余批次并保留部分收据；任何后续修复重启必须显式计入已有预算和新增协议，不暗中再跑整套。当前合同不包含另一次hfield或plane重复。

## 评分和记录

原 `gates_and_metrics`、原门槛和窗口直接复用，不按新结果改宽。额外plane不变量检查与原性能门槛分别列出：

- 全部case：完整时长、无非轮地面接触/翻倒/边界，实际roll/pitch≤10°，height RMS≤.015m，heading peak≤5°。
- 停车：完整8秒；原raw停令从tick400为0；endpoint tick500..799最大|COM body vx|≤.03m/s，位置tick500..700累计平面路径≤.05m，heading RMS≤3°，全程raw vx RMS≤.05m/s。
- 转向：原目标heading最终增量±.3rad；endpoint tick450..799最大heading误差≤3°、最大|COM body vx|≤.03m/s，全程最大平面原点位移≤.1m；仍受全程heading peak≤5°约束。
- 冲击：沿用forward_and_impulse和impulse_cases全部门槛，vx RMS≤.05m/s，heading RMS≤3°，横向峰值≤.15m/末值≤.1m，endpoint tick820..1199恢复后最大heading误差≤3°。

raw误差始终为执行区间端点真实base inertial-COM body-x速度减该区间原raw命令。base_position沿原约定是可见body origin；不要将二者混用。停令时刻、最后prepared额外decision、执行T区间和T+1状态各自清楚标记。reward只按原定义记录，不替代raw门槛。

保存qpos/qvel/observation、raw与servo命令、动作、controller unlimited/safe torque、5子步实际actuator torque、PI前后、body/wheel速度、俯仰和原phase指标。停车另报最大原方向延伸、峰后回撤、总反向路程；转向另报wheel target/error与headingservo/内环限幅占用。

接触证据必须补足：在每个实际native子步返回时只读采集接触cache，记录接触源相位（不能误标为同步端点重算）、wheel-floor/非轮接触数量、最大法向水平分量、按geom顺序变换的normal/tangential合力和总wrench；`mj_contactForce`可用于读取既有求解结果，但不得为了日志在live data上额外调用mj_forward。同步端点的published contact flags/normal可以另记，明确其相位不同。使用只读观察器不得改变输入、积分次数或warm start。

plane运行时所有wheel-floor法向也应满足水平分量≤1e−12、|nz|≈1；纵横切向摩擦力可以非零，不能误判为法向异常。没有接触/轮暂时离地应如实记录，不通过缺失接触自动满足稳定性能。

## 不加预算的内部一致性检查

新plane自己的相同命令前缀提供配对依据，旧hfield不参与逐位检查：

1. 六case reset的qpos/qvel相同，原controller memory、reference初值相同；初始raw命令均0，初始observation也应逐位相同。
2. forward_stop与两项forward impulse在执行tick0..399命令和外力相同：比较qpos/qvel/position状态0..400（401项），动作/请求与实际扭矩/PI及观测0..399（400项）。stop的observation[400]已携带归零下一命令，不要求与impulse相等。
3. positive/negative impulse在执行tick0..599完全相同：比较状态0..600、观测0..600，以及动作/扭矩/PI0..599；tick600才开始相反外力，不能多比一个已受扰区间。
4. left/right turn在执行tick0..199相同：比较状态0..200及动作/扭矩/PI/观测0..199；observation[200]已经包含相反yaw命令，不纳入相等检查。

这些检查用于发现隐藏额外步、跨case状态残留、source重复消费或时序不一致，不要求改变输入后的左右轨迹镜像。若case早停，记录可比前缀长度并把完整配对检查标未完成，不能用共同短前缀冒称全通过。

## 通过判据与后续顺序

报告必须分开给出 `plant_semantics_valid`、`execution_evidence_valid`、六项各自 `original_task_gates_passed`。碰撞表示正确不等于控制通过；一次停车或转向失败也不自动证明plane接口错误。

1. **plant/证据不合格**：停止控制律候选资格比较，修复新适配/身份/记录问题，保留所有旧记录；不改控制器“补偿”错误地面。
2. **plant/证据合格，但停车或转向失败**：新的plane zero轨迹成为当前平地诊断基线，可继续独立停车或转向控制实验。先看新的失效签名，不能因为旧hfield失败就自动启动已存stop damper或1.0 yaw limit。固定b候选保留原值和来源；只有新plane同样存在可解释的纵向腿回摆、独立合同重新确认后才比较，不把plane与damper同时计作一次变量，也不事后重选b。
3. **六项原门槛全部通过**：这只支持该版本plane/zero的有限开发集合。继续同一backend的GUI渲染/输入/reset无额外步验证和用户人工试驾，并可准备独立低速起跳/落稳阶段。不能说完整驾驶目标或PPO泛化已完成；旧PPO仍不得绕过身份加载。
4. **障碍阶段**：保留原目标“稳定启停/转向→可靠越障→提速”。新plane模块只支持平地，不得把未来坡道、碎石、台阶heightfield全改成plane。障碍几何必须另立有真实高度/法向的版本（例如审计后的盒体台阶/斜面或独立修正heightfield），明确原5cm bevel与垂直台阶的差别。先验证实际四轮最小净空、同时离地时长和落地后稳定，不能用机身升高/动画完成替代；现有5.61mm四轮净空仍是未解决证据。
5. **提速阶段**：在选定backend下稳定启停/转向、低速障碍/落稳和人工驾驶验收通过后，再用独立协议逐档验证实际速度、制动距离、通过率及翻倒风险。现有Shift档位不等于已验证提速能力。物理自救另立任务，R仍只标simulator reset。

本合同只给下一次5600步plant验证授权边界，不预支后续控制、GUI或障碍实验预算。冻结77输入、旧结果/sidecar、旧release/tag均保持字节不变；新实现、新协议和新结果全部独立归档。
