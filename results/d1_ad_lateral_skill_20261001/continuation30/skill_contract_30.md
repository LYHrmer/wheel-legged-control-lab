# C30：事件级 3D 侧移技能，先可执行参考，再小规模 RL

状态：工程与资格合同草案，当前只授权新源码/纯测试准备；不含任何模型、物理或训练 GO。实际 Astra 主路线审阅者制定；root 独占执行与计步。C27–C29 失败、B22 与所有旧冻结源保持原样。C29 仅做其已保存前缀的诊断，不重跑、不再追 GUI 管线。C30 无渲染，不以 GUI RTF 阻塞技能数学，亦不声称交付了可用 A/D GUI。

## 本轮要交付什么

一个可被事件级学习器调用的侧移技能：四个真实腿规划边界各产生一次 3D 动作，内部仍是 100 Hz 控制、每拍 5 次正常 native 积分。第一场只执行预注册 `fixed_nonzero_left`，独立读回通过后才逐项放行其余资格场。不得一次自动扫完六场。不开阻尼训练，不复制 B22 网络/Adam，不把新技能写进旧 99D vy 槽。

分工：Sol 编写 `sol30` 中版本化 teacher 子类、参考时钟与宏事件接口及必要纯测试；root 接现有同步 env、headless worker、归档/计步与执行；另一 Astra 在 `read30` 复用现有 compiled FK 和数值读回，增加整轮净空/宏转移验证；主 Astra 负责合同与有限 GO。优先复用 C27 同一 env.step/provider/servo/B22 交接链，不新建 GUI、进程架构或通用证书框架。headless 只登记live、measurement、sideScratch三个独立MjData，同owner同model，仅放行live→measurement与measurement→sideScratch两条copy边；没有渲染槽、显示data或owner线程join。几何直接复用冻结`kinematics24.collision_bounds24`，运行读实际measurement的geom位置/姿态，reader用`reconstruct24`从保存qpos独立求解；不调用会增加逐体速度查询的`endpoint24`。

## 三维权限与零对照

动作在每腿原 `_prepare_leg` 规划完成后、修改参考之前锁存，随后不变：初始机身坐标 XY waypoint 修正各不超过 12 mm，重叠系数 λ 在 [0,1]。方向和脚最终横移 30 mm 均由任务固定，最终机身 recenter 仍回原目标，不开放步幅、力矩、抬高或直接跳相位。

精确零动作走原 teacher 函数与运算路径。不能用重写后的公式接近原结果，不能把原型对象身份测试当实际整周期同轨证明。第一资格场中的固定规则唯一：每腿 waypoint 沿 `body_from - original_body_to` 的水平单位方向回缩 8 mm（零跨度则 XY=0），记录变换到初始机身坐标的 proposed 值；λ=.5。不搜索其他向量。

非零 XY 候选最多做一次额外原 plan IK（outer=2、inner=6），检查原关节范围、IK误差≤8 mm、max_shift≤120 mm、预测三足静态余量≥原40 mm目标减去最大12 mm权限，即≥28 mm；原实际动态 margin>20 mm、速度和接触准备门仍独立保留。未满足则完整回退该腿 teacher XY并记录原因。不得任意降低原 fault 门。新 shift 时长调用原 `_move_time(new_span)`，保留原 ZMP 15 mm预算与 .30 s最小时长；不能把旧实际时长作为下限，也不以1.40 s截短原加速度约束。

λ 采用新的受事件触发参考时钟，不能直接把离线原型 `sample` 的身体/足部共同零时刻接进运行。上升仍35 mm/.26 s，水平仍原 quintic、原0.16 m/s尺度和原 `_swing_time`，下降仍原 .26 s至hover与原probe/dwell。水平提前启动必须同时满足：本拍实际整轮 collision min-z>12 mm；此前已完成的5个正常 native均无该轮有效地面接触；支撑/姿态/关节故障门未触发。记录实际触发拍，不预知未来接触。λ控制可用提前量，不能绕过门。

提前下降只能在上升到顶且参考垂直速度/加速度均归零后启动；其解析目标在水平尚未结束时保持整轮min-z≥17 mm，实际每个已记录native整轮min-z必须>12 mm。17/12 mm的5 mm差是本轮待验证余量，不是全域跟踪保证。几何由实际compiled wheel collision形状、姿态和支持函数导出；旧 `_geometry` 的 .087/.020圆柱近似是已有近似，不得说成没有几何，也不能不验证便当整轮证明。只支持实际封存模型的类型，未知类型拒绝；scratch接触不能作native载荷。

取消锁存后禁止开始下一腿。为保持参考位/速度连续，允许当前已开始的水平/垂直轨迹完成至当前腿安全落地，再进入原abort_hold；不得把有非零参考速度的重叠运动直接切到旧静止land_from。取消至安全交接候选硬帽300 controls，且总场2200帽不变。若无法实现连续取消，先报明确接口阻断，不执行非零场。

## 有限资格场与依赖

最多六个独占 cold worker，各≤2200 controls、≤11000正常native、+2 compiler；合计上限13200/66000/+12，不是预留自动执行。每场load B22≤1、torch_load≤3、probe32行≤1、B22 predict≤2201/actor rows≤2232，side期间B22 predict必须0；训练、优化、保存模型均0。侧步compute≤1500、prepare_leg≤4、start≤1；新额外IK上界见spec。已有API output记录，无额外mj_objectVelocity查询。FK支持函数为纯算术，另记次数，不计作正常积分。

直接复用冻结`continuation27/director27/host27.py`的无渲染分支，不复制host。C30新wall合同为soft240/close270/hard300秒，原host外层900秒、source preflight240秒/postcheck180秒保持有限。依据是C29实测入口/关闭和全机负载条件下原短wall窗提前结束，不能将此归因于某个其他进程；这仅给新headless场明确wall余量，不增加控制/native/model次数，不恢复C29或提供GUI性能资格，也不保证一定能在该期限完成。

headless固定200拍准备后一次意图，raw/servo vx/vy/yaw始终0；完成交接后固定400拍保留，最后100拍沿用原停止/载荷门。原本已具证据的W复驾不在本轮重复；须记录新的交接记忆清除、provider连续和B22真实恢复predict。这只资格侧移技能，不能替代新GUI多周期/键盘验收。

顺序：①fixed_nonzero_left；其完整安全/任务/非零权限实际生效通过后，②fixed_nonzero_right；然后③zero_left、④zero_right，使用同版本headless精确起拍/初态；⑤cancel_left、⑥cancel_right，在第一段水平与垂直参考速度均非零时的下一控制边界取消。每场硬停止后不补步，任一源码/归档/安全失败先停止后续。未出现真实重叠的取消场记覆盖不足，不伪造触发或追加场。

完整成功沿用旧30 mm任务门：方向正确、净移25–50 mm、终点误差<12 mm、yaw误差<.12 rad、四腿真实离地/落脚、finish margin≥5 mm、4 s保留≥.90、回退≤5 mm、原姿态/关节/接触/力矩故障门；末100 raw/servo0，mean|vx|/|vy|≤.04、mean|yawrate|≤.05、clearance std≤.02、各轮正native载荷比例≥.95。取消只验安全连续落脚/交接，不计位移或速度成功。实际边界和计数必须从原始记录复算。

第一场不要求与尚未运行的新zero比较；C28保存左移只作描述参考，绝不恢复旧正式资格。六场齐后才评估训练触发：左右固定候选均真实非零且各至少两腿有≥1控制拍重叠；全部安全/任务和两取消门通过；相对同版本zero，两方向周期时间均至少降低5%，终点误差不高于zero+2 mm，保留比例不低于max(.90,zero−.02)，完整归一化力矩平方均值不高于zero的1.20倍。未过则本权限不训练，不搜索一批新向量。5%是是否值得pilot的门，不是最终用户提速交付。

## 宏转移与后续 pilot 候选（不含训练 GO）

接口 `begin_leg(teacher_plan, observation, direction) -> latched_action`，内部每拍追加实际reward项、5T状态/力矩/净空与投影记录；下一腿边界结算一次 `(obs, proposed_action, reward_components, elapsed_controls, next_obs, terminated, truncated)`。latch在实际control i生效，各宏范围为[latch_i,latch_next)，第四腿动作到实际side_end+400，包含recenter及固定400拍保留，不能丢掉终端成本。200拍准备不计宏但计全场资源/力矩。方向做明确镜像归一化，观测使用独立schema，不能冒充B22普通99D。先锁字段、归一化与来源，再建模型；不为凑维数复制重复100Hz动作作训练样本。

本资格阶段不计算带权训练奖励，只保存以下可独立复算分量。设初始机身坐标的有符号侧移为s、纵移为x，L=.03m：φ(s)=min(s/L,1)，不设下截断，progress为逐拍势差之和，首项使用该宏第一拍pre endpoint；backtrack=Σmax(−Δs,0)/L；elapsed=n×.01s；x²、(s−L)²的时间积分同时保存原单位与除L²版本；相对初航向的wrap(yaw)²积分同时保存原单位与除.12²版本。平方积分使用每拍实际post endpoint。实际力矩项=.01×Σmean_(5native,16actuator)((τ/limit)²)，全场与各宏分列，不称能耗。终止reason为`next_leg / success_after_retention / safe_cancel / physical_failure / budget_truncated / software_failure`，预算截断/软件失败不得伪作成功。安全取消的最后宏同样延续至真实handoff+400再标safe_cancel，300拍取消门不含这400拍观察；若预算或故障使该窗不完整，保存真实截断而不补步。训练时具体权重、失败罚及bootstrap处理另行冻结。

建议唯一低容量pilot以256个真实宏转移、64个周期和140800 controls三者任一先到为硬停止；正常native≤704000，最多8个完整32-event rollout。失败/截断也消耗周期与控制额度；尾部不足32不追加物理。原32768controls约76–124个腿事件，不能作为充分学习结论。候选线性Gaussian actor与线性critic，零均值初始化；32-event rollout、batch8、epochs2、最多64 optimizer步骤。gamma=1有限周期回报，GAE按真实持续时间衰减，初值每秒.95；实际学习配置、reward权重、native/compiler/reset预算和host上限须在未来训练合同独立冻结。当前不得据本段启动训练。

候选奖励按100Hz真实时间积分：净侧移势差只奖励最终净移，固定30 mm目标，时间成本、纵漂/yaw/反移和真实归一化力矩平方成本分列；成功奖励须完成保留/安全门，失败不能因少耗时获利。投影 proposed/projected/consumed 和每腿有效权限覆盖全部保存；全部回退teacher即无有效动作空间，不训练。

pilot最终必须对新冻结保留集同时比较zero、固定工程动作和learned，报告30 mm/周期及保留净移/周期、两方向与取消。至少对zero实现可复核10%周期降低且超过固定动作5%才称初步学习收益；2倍或0.01 m/s是研究目标，不承诺、也不能用37 mm超调冒充。完整测试场景、评分门和预算在训练前封存，开发/最终不混用。单台确定性仿真结果不是多随机种子的统计泛化。

## source GO 前最小检查

只做必要纯测试：零分支原函数路径；新waypoint计时/额外IK次数；quintic端点和解析速度加速度；compiled collision支持函数与实际绑定；重叠时钟/门缺失拒绝及取消连续性；真实宏计数/变时长回报。实际worker入口通过同一origin函数在模型前预飞，最终保存段用保存fixture走真实代码，不再维护第二份模块清单。host使用既有有限独占管理；首物理GO只含第一场及精确冻结source/hash，不含pilot或其余自动执行。
