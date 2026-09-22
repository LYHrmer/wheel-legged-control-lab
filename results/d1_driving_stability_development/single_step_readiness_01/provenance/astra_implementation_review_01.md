# 单 15 mm box 合同：实施前只读审查

审查者：实际 gpt-6-astra ultra 子代理；日期：2026-09-22。主代理负责接口集成、计步及物理；Claude 实际 provider 可实施有界模块。本报告不替代独立代码验收，不声称已经完成任何新几何前检或物理验收。

审查范围：冻结 next_contract.md、docs/codex_handoff_20260921.md，以及下文引用的直接相关源码。合同 SHA256 已只读复核为 `1ea04dc310486ed7025a73ed1e61a7977fdaafe0545f3de859905835d1b5c238`。仓库 HEAD/main、CI 与 frozen77 状态沿用主代理已完成的核验，不在这里重复宣称为本代理实测。未实例化环境/plant、未调用 MuJoCo、未训练或运行测试、未修改仓库/冻结文件/旧证据。本报告 exclusive-create。

结论：现有控制核心可复用；现有 flat 环境、flat 接触有效性判据及旧 native 归档均不能直接承担新合同。最危险的缺口是 world-z 转换、初始化缓存绑定和 native 接触时相。以下是保持原合同的实施要求，不增加 2400 control / 12000 native 上限，不改变通过门槛。

## 1. 必须先处理的具体问题

1. **world-z 0.455 与 clearance 0.455 不同。** `src/wheel_legged_control/d1/control_loop.py:259` 构造 world command 时执行 `ground.height_m + command.clearance_m`；`wheel_leg_controller.py:307` 再执行 world height 减 ground。直接保持 D1MotionCommand.clearance_m=0.455 会在 box 上静默增加 15 mm 世界高度前馈，违反合同。推荐在新环境的单个命令适配 seam 中保留独立 raw world-z=0.455 记录，用同一 oracle ground 生成 `clearance_m = 0.455 - h` 后进入原 loop；forward/yaw/time 必须仍是原 raw 值，raw callback 恰一次。另一可审查方案是新 loop 的最小 world-command construction override，其余逻辑逐项保持原样。两者只选一种。必须记录 raw world-z、实际 h、motion clearance、prepared world height、controller ground 参数，并验证 prepared world height 恒为 0.455。不要将经过高度语义适配的命令冒充未经处理的 raw 记录。

2. **box 应相对 episode 起点，不能相对构造临时状态。** `locomotion_terrain.py:28` 的 nominal spawn 是 (-3.8,0)，`locomotion_env.py:362` 在 reset 传入此 spawn；`model.py:609` 的构造 reset 默认却是 (0,0,0.455)。所以本合同 box 中心 (-3.10,0,0.0075)，前侧 -3.28，远侧 -2.92。应以同一冻结 spawn 常量构建两个条件，先编译 box，再 reset 到原 nominal episode 状态。不得 compile 后移动 box 以补偿错误起点。

3. **不能对旧 plant 只换 model。** `model.py:251` 直接调用内部 compile 的 build_d1_model，没有现成 factory 参数；`model.py:258` 的 measurement data、`model.py:276` 起的 ID/address/wheel map、`model.py:317` 起的 nominal arrays 均绑定旧模型。新模块可复制窄范围 frozen builder assembly，在 original `_add_box` 后 compile；新 D1Plant 子类构造器不调用旧 D1Plant.__init__，而针对新 model 新建完整 data/measurement_data、ID、地址、wheel map、terrain IDs、nominal arrays、actuator状态，然后复用原 reset/step。必须逐项初始化，不能靠旧对象遗留属性。不要全局 monkeypatch builder；不要编辑 frozen builder。复制构造代码需在源 manifest 和名义数组差异报告中显式列出来源。

4. **StopTurnEnv 的继承链携带 flat-only 语义。** `scripts/d1_stop_turn_env.py:26` -> TurnAuthorityEnv -> PlaneDiagnosticEnv -> D1FlatPlaneHeadingEnv；后者 `d1_flat_plane_env.py:380` reset 强制 `validate_flat_plane()`，其 `:169` 要求 terrain IDs 恰为 floor 一个，ground query `:258` 也会验证 flat。`PlaneDiagnosticEnv` 在 `probe_d1_heading_flat_plane.py:36` 已把 archive 绑定 plant。推荐新 env 直接继承 D1HeadingTrackingEnv，在第一次 reset 前替换未绑定 plant 和 controller，复用 StopTurnCompositionController 本体；不要用虚假的 validate_flat_plane 放行 box，不要在 archive 创建后换 plant。稳定 task config 在父构造期间会被虚调用，应只读取预先保存的标量身份，不能访问尚未建好的新 plant。

5. **旧接触数据不足。** `d1_native_contact_diagnostics.py:22` 起只留下 active wheel-terrain 接触，`:29` 非轮接触仅计数后丢弃，`:37` 丢弃 inactive contact，`:49` 未保存完整 geom pair/body/frame/local force。`d1_probe_archive.py:74` 正常 native 行只有 ctrl/wrench/time，qpos/qvel 仅在错误分支 `:89` 保存。合同需要每次实际调用前后状态及所有接触，故需新的完整 recorder。`probe_d1_heading_stop_turn.py:95` 和 `probe_d1_heading_flat_plane.py:114` 的“全部法向竖直”必须完全退出新 box 记录有效性判定；前立面水平法向是正确物理。

6. **native cache 与 post-step qpos 不同相。** `model.py:742` 原 mj_step 返回后保留的是本次 native 求解 cache，而 `data.qpos/time` 已推进；`d1_native_contact_diagnostics.py:11` 已明确这一点。原同步 endpoint 通过 `model.py:382` 复制到独立 data 再 forward。recorder 必须先保存原生求解接触和 pre/post state，再允许原同步 publication；不能为取得接触调用 forward、refresh_measurements 或 set_simulation_state。qpos_after 与 c.pos/c.frame 同行可以，但字段必须显式标注各自时相，不可说接触力是在 endpoint 新算的。

7. **zero integration 不等于 zero forward。** `model.py:639` reset forward，`:382` synchronized measurement forward，`:662` 附近 set_domain/setConst/refresh；`wheel_leg_controller.py:201` 首次 controller 构造会创建另一个 D1Plant，`:128` 的 IK 表构建在 `:149/:178/:197` 多次 forward。根代理应从构造前安装禁止 step/step1/step2 的前检 guard，并按 model/data 身份统计全部 forward，包含丢弃 plant、controller scratch 与 episode data。不可仅看最后 data.time=0 声称没有隐藏积分，也不可把静态 forward 力算作动态接触资格。

8. **不应把 active constraint 数当正载荷占比。** `model.py:559` 的 active_sample_fraction_by_wheel 来自 contact count>0；旧 native sampler 同样以 efc_address>=0 计数。这不证明法向载荷>0。新的每轮 0.95 门槛须由实际 native local normal force 严格正值形成 500 个布尔样本，每样本同一轮有多点也只计一次。不要用 abs(Fz)，水平前立面接触也可有正确正法向力。

## 2. 建议四个新模块及职责

- **几何与 plant**：`build_single_step_model(obstacle_enabled)`、完整新 plant 构造、compiled geometry binding、真实 ground query、只读 bounds/contact feature helper。仅有同版本 False/True 两种地形。固定 `_add_box` 原 helper (`terrain.py:83`) 参数，全尺寸 0.36×1.24×0.015，0.9 摩擦；不手动把 box margin 改成轮的 0.001，helper 的 compiled 默认也要保留与报告。
- **native recorder 与预算账本**：接受现成 plant/clock/call delegate，单入口独占实际 native 调用，完整记录正常和失败分支；不构造环境，不选命令，不主动积分。纯假 delegate 可测试计数/失败/归档。完整接触导出放在此模块或几何模块内，不能再堆旧 flat sampler wrapper。
- **环境适配**：直接承接 D1HeadingTrackingEnv，original oracle/default nominal/actuator设置，StopTurnCompositionController 实例，单次 raw bind，高度语义适配，新元数据。原 loop 和 inherited plant.step 保持一次消费；controller preview 不提交 PI/latch。环境本身不决定 batch。
- **纯 scoring + root runner**：scorer 只读完整 archive；runner 独占构造、前检、两场顺序、counter、source manifest 与最终收据。固定 seed77301、1200×2，k200..999 forward+.2，其余0，raw yaw0，每拍zero8。不要复用 G1 run_episode，因为它绑定旧 case/score/扰动和不完整旧 recorder。

若减少文件，纯 scoring 可与任务定义合并；不应为节省文件把物理、归档和评分隐式混在 controller 内。Claude 有界实施适合几何 binding/bounds 与纯 recorder/scorer；主代理最终审查模型构建复制、世界高度适配和真实 native seam。

## 3. 精确零积分前检（根代理实施，本审查未运行）

### 构造、身份与等价性

1. 在任何新 env/controller 构造、reset、测试 fixture 执行前，把 mj_step、mj_step1、mj_step2 及批量步进入口替换成立即失败的 guard；统计成功/被拒调用，记录 mj_forward/mj_setConst 调用数与 model/data 标签。别先实例化再安装。所有测试在新进程中只调用明确列举的前检，不运行可能包含物理的全测试集。
2. 同版本 obstacle False/True 编译，并只读构造旧原 native-plane 作基准（只构造/reset，严禁积分）。先核验数据不相互别名，measurement_data 与 integrator data 分离，fresh addresses 属于各自 model，所有 required mj_name2id 结果>=0、名称反查一致、wheel-body映射四项唯一。terrain set 分别恰为 floor、floor+单box；无 hfield、其他 world碰撞物、隐藏外力或 mocap。
3. 按稳定实体身份逐项比较 robot body mass/inertia/ipos/iquat、body/joint拓扑、joint axis/type/range/pos、qpos/dof地址语义、damping/armature/frictionloss、actuator target/gear/range/力和速度限制、所有 robot geom type/size/local pose/contype/conaffinity/condim/friction/margin/gap/solref/solimp、solver/integrator/timestep/gravity/options及依赖 mesh/material 资产。只允许额外 world box 与声明 identity/ID变化。新 False 与旧 flat 必须同 nominal物理；新 True删除 box后的结构与False相同。
4. 原 URDF `<collision>` 未手写名称（如 robot.urdf:86,325）。不能假设编译后几何均有可用名称；前检先枚举并记录。若原编译缺名，则在新两条件 compile 前给匿名 geoms 按 body名+原本地ordinal 分配唯一、稳定名字，另存与旧模型的结构键映射（body名、type、local pose、size、ordinal）。不要将运行期 geom ID 当实体名，不允许 missing lookup=-1 悄悄访问末项数组；命名差异单列，物理值必须不变。
5. 两条件原 nominal episode reset 后时间为0、dt=.002、control=.01、physics_steps=5；qpos/qvel/ctrl/warmstart/actuator/controller初值按相同语义比对。静态初始机器人所有 collision geom 与 box 分离；分别记录 constructor scratch、episode reset 与原 flat comparison，避免错误使用x0=0。

### 地形 query、真实 geometry、contact frame

6. 查询闭矩形 `[x0+.52,x0+.88] × [y0-.62,y0+.62]` 内高度 .015，外部0，包含四边/四角、中心、每边内外近邻。用实际 compiled box/plane 的下射线或独立 compiled几何距离结果对照，并将射线限定为 terrain，不能被机器人遮挡。边界按闭集固定；不能用近似 ramp 或给 pitch/roll虚构坡度。
7. 只用 scratch data 设置明确合成姿态，调用最小 kinematics/collision 所需入口；全部时间保持0。至少 plane轮接触、box顶面轮接触、前立面、前顶边棱、分离姿态、非轮box接触和非轮plane接触。任何 forward次数单列；静态正力也标注 static，不进入 readiness loaded计数。
8. frame检查：所有保存 frame行正交、单位，local wrench有限；原 frame第一行按 geom1->geom2解释。terrain=geom1时 n=frame[0]、F=frame.T@local_force；terrain=geom2时二者反号。正法向标量为 dot(F,n)，不应再把标量反号。对换 pair排序的纯合成输入应保持同一 terrain->robot世界力/法向。plane预期+z，box前面预期-x，顶面+z；边棱法向可为相邻面法向的凸锥组合，不要求竖直。
9. contact.pos可能是碰撞点对中间位置，不能以“恰好等于某表面坐标”作无容差硬判。使用 compiled box局部坐标、contact dist/margin与最近特征共同分类，保留 face/edge/vertex候选及残差；未知/不一致分类报告数据问题，不能静默归成top。记录全geom1/geom2 name+ID、body name+ID、dist/include margin、efc_address、dim、frame9、pos3、local wrench6、signed world wrench、normal scalar、wheel/terrain/feature分类。

### bounds、命令与归档纯测试

10. 只选择真实 robot collision geoms（检查 contype/conaffinity；纯visual剔除），四轮按foot body和实际 cylinder geometry绑定，不能因为foot body有decorative mesh而全收为碰撞。所有可碰撞形状必须被 bounds覆盖，不支持类型直接拒绝。
11. 本URDF机器人 collision 是 box/cylinder。对世界轴单位向量 e，box半投影 `sum_i abs(e·R_i)*s_i`；cylinder半投影 `half_length*abs(e·axis) + radius*sqrt(max(0,1-(e·axis)^2))`，min_x=center_x-半投影。这是当前旋转下真实形状支持界限。测试身份/90°/倾斜、多个geom同body、全机器人最小值由后轮/非轮分别决定、visual长突起排除、margin严格大于而不是>=。box距离另外用compiled shape query，不能用wheel-plane gap冒充。
12. 固定命令边界纯测试k=0,199,200,999,1000,1199；允许终端prepared k=1200是held零命令，仅为T+1状态，不追加第1201次compute。raw command回调每prepare一次，full run应1201次；compute1200次、PI更新1200次，stop latch初始0至999不活跃，1000..1199活跃200次，turn gate始终False。原 controller preview任意次数不推进状态。输入zero8 strict拒绝bool/nonfinite/非零/错误shape。
13. 高度语义单测 h=0/.015以及边界查询：raw world=.455，prepared world=.455，controller收到真实h，relative目标=.455-h；forward/yaw同raw。禁止由于terrain换高新加controller补偿。必须保留原计算链，而不是复制一个“相似”PD/PI。
14. 假 native delegate 覆盖返回1步、入口抛错、时间推进后抛错、observer/serializer抛错、partial 1..4 native、full5 native、第6001/12001次被拒、错误model/data、bulk nstep参数、其它模型的意外积分被拒；保存attempted/returned/advanced/control-started/control-completed计数，不能把返回数或round(time/dt)单独当全部计步证据。预算守门放delegate之前，1200/6000与累计2400/12000双重硬限。
15. 归档用临时新目录exclusive-create；完整态T+1、控制字段T、native5T；异常态保留已有状态与partial native，缺失controller行不可填充last record。NaN必须保留原浮点/位置，同时记录无效；写盘错误与物理错误分别归档，finally仍写终端计数与manifest，close不能覆盖已有终端。关键native行应增量持久化，别仅存内存到正常结束。
16. scorer伪造输入覆盖空/短/NaN记录、1100与1101端点差异、500th sample off-by-one、零力但active constraint、单轮多contact不能重复加占比、只base越线但后轮没过、非轮瞬时native接触、first contact前1步分岔。保证这些缺口不会误报通过；不运行任何物理。

## 4. native记录与前缀比较

每个实际调用必须在delegate前保存control tick/native index/time、qpos/qvel/ctrl、qacc_warmstart、所有xfrc_applied和qfrc_applied；delegate后无论成功/失败保存post状态、时间变化、返回标志及异常。成功返回后立即导出全部contact，不限wheel或terrain，以便发现未知接触。完整状态/接触可分NPZ与JSONL但索引应唯一可联结，采样时相与model manifest身份明确。额外建议保留integration state中影响重放的激活量/warmstart；不能仅凭qpos/qvel/ctrl宣称完整独立重放。

Root独占、顺序运行plane-only再box。先分别记录 first geometric box contact、first active constraint、first strictly-positive wheel-box native load；这些时刻不能混淆。跨条件共同前缀以首次box实际几何/solver相互作用之前为界，对每个完全在界前的native比较pre/post qpos/qvel、ctrl、raw命令、PI/stop状态，并在接触所在native入口比较共同pre-state。只排除已声明的模型ID/terrain身份差异；不能泛化数值容差掩盖前缀动力学差异。box接触后不同是预期，不比较它与plane相同。完整pre-contact controls不足1个也要如实报告可比较长度。出现此前不可解释分岔时停止新物理并诊断记录/plant，不调控制、不补跑。

计数不以data.time替代拦截器；二者相互核验。记录constructor/reset/静态forward次数单列，物理总数仅来自已授权native。不得通过旧G1 observer让外来model的step自由透传（`probe_d1_heading_g1.py:111` 的旧行为不适合新预算guard）。

## 5. 评分结构：三种有效性与任务结果分开

输出独立字段：record_valid、geometry_preflight_valid、wheel_box_contact_qualified、mechanical_crossing、final_stable、task_passed、failure_reasons。plane-only是对照，box qualification/crossing标记不适用，不能让无box事实把对照record_valid置False。box条件未接触为曝光不足、机械通过False；仍可记录有效。

- **记录有效**：所有实际已执行native有完整对应数据，索引/时间/计数自洽、数值有限、动作zero8和raw身份正确、original名义参数/额定保护/零外力、真实terrain身份、sources未变。原保护触发或提前terminated可以是有效失败记录，不能因物理未达1200而删掉失败；但partial无法通过完整任务。
- **接触资格**：至少1次native已求解的真实wheel-box contact严格正法向载荷，且geom/body/name/frame/feature解释一致。query高度、base位于box投影内、efc_address>=0、静态forward载荷均不代替此项。
- **机械通过**：完成同一次固定轨迹；至少满足上述资格；最终四轮与全部robot collision geom的保守/真实world min_x均严格大于`x0+.88 + original contact margin`。应使用每个geom实际compiled margin或事先声明且不弱于原轮0.001m的统一保守margin，不修改模型参数。最终越线证明仅针对采样时刻。任何非轮robot-box/plane实际碰撞在native或endpoint出现均失败；保留纯margin候选与dist<=0/active/positive分类，不能靠零载荷把真实非轮碰撞排除。保留原倾覆/位置/速度/力矩保护。
- **全程几何姿态门槛**：|heading|<=5°、|roll|和|pitch|<=10°、|y-y0|<=0.1m；使用实际orientation和位置，0..1200 endpoint以及可用native状态分别报告，捕捉native瞬时越界。不能把姿态相对伪坡面或航向内部伺服请求当实际值。
- **最后100控制interval**：准确选endpoint 1101..1200，对应控制k1100..1199。max|body vx|<=.03、max|whole-robot COM world vz|<=.03；RMSE sqrt(mean((base_world_z-.455)^2))<=.015。COM不能用base qvel[2]；可沿已验证 `d1_jump_ppo_env.py:343` 的质量加权位置和mj_jacSubtreeCom@qvel，只在同步endpoint data计算，并核实base subtree含全部robot质量。box是world geom，不应引入机器人质量。
- **最后500 native**：准确选索引5500..5999（0-based），每轮存在严格正normal load的样本数/500 >=.95；同轮多contact每样本只算一次，可来自plane或box。完整500缺少任何样本则不能通过，不补数据、不变分母、不缩短窗口。
- **最终task_passed**：仅在完整1200/6000、record_valid、接触资格、机械通过、全程门槛、final_stable均满足时True。有效zero失败仍是合法下一RL任务基线，不能调高度/速度/gain多试场次刷通过。不得添加四轮flight/20mm门槛，也不得自动训练。

原contract的“任何非轮接触”与原0.001m接触margin需在root preflight执行schema中写清采样定义，严禁运行后根据是否通过再选dist/force阈值。这里建议同时保存所有原始候选与分类，从而可审查且不丢掉碰撞失败证据。

## 6. 进入唯一两场前的必要交付

新源码+测试source manifest、合同hash匹配、frozen77只读复核、False/True/原flat名义差异清单、zero-integration guard与forward计数收据、合成几何/contact frame结果、query/ray一致性结果、pure计数/异常归档/scorer结果，以及主代理对world-z语义和缓存重绑定的签核。所有都通过才消费一次1200+1200；中断保留实际消耗和未执行部分，不能重试或续成额外场次。

未解决项属于具体实现验收，不需要重开用户例行许可；当前合同不包含下一RL训练预算。审查到此结束，仍为新阶段0 physics。
