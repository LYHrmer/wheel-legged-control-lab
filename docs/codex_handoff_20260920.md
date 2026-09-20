# Codex 续接：平地停车/转向组合通过固定八场，继续跳跃资格

完整目标仍为 **稳定直行 → 可靠越障 → 提高速度**，没有完成。必须同时阅读 [2026-09-14 交接](codex_handoff_20260914.md) 和本文件；最终进程、HEAD、GitHub CI及下一步状态位于本机 `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/continuation_state.json`。旧交接及旧状态文件没有改写。

## 授权和不变约束

用户授权常规开发、验证、及时上传GitHub，要求 gpt-6-astra / ultra 规划，真实 Claude Opus 编写有界模块，主代理集成和物理验证。不要把“已通过单元测试/CI”当作驾驶能力通过。不要重跑三个65536步训练或完整24场G1，不修改 `results/d1_budget_study/protocol.json` 中77份冻结源，也不改旧记录、模型sidecar或旧manifest。通过新增模块和子类扩展。

每条shell命令加 `rtk`；原始输出或无专用过滤器用 `rtk proxy`。仓库没有 `.codegraph/`，不创建索引。常用Python环境：

```bash
rtk proxy env PYTHONPATH=.local-deps:src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ...
rtk git push git@github.com:LYHrmer/wheel-legged-control-lab.git HEAD:refs/heads/main
```

SSH上传已验证；以GitHub API/实际remote refs核对HEAD，直接URL push可能不更新本地origin/main。不要恢复已废弃的HTTPS分批push。

## 本轮按优先级完成的物理比较

所有下列评测均为zero8残差、原G1 raw命令、原评分。没有新的训练、参数扫描或默认控制器切换。

| 独立试验 | 新增控制拍 / 物理子步 | 结果 |
|---|---:|---|
| 固定 .5 m/s² release参考，4case×2条件 | 8000 / 40000 | 正倒停车均失败；不采用 |
| 内层yaw cap .6→1.0，左右turn×2 | 3200 / 16000 | 两方向均未过原5°峰值；不采用 |
| 原zero控制器，native-plane六case | 5600 / 28000 | 表示/证据有效；两impulse通过，两stop/两turn失败 |
| 固定停车腿阻尼，plane四candidate | 4000 / 20000 | 两stop全部原门槛通过，两impulse完整逐位noop |
| 固定系数1轮中心转向补偿，plane四candidate | 3200 / 16000 | 两turn改善但仍失败；两stop完整noop且保留原失败 |
| 固定纯turn腿纵向阻尼，plane四candidate | 3200 / 16000 | 两turn仅小幅改善仍失败；两stop完整noop |
| 恢复原内层yaw反馈，plane四candidate | 3200 / 16000 | 两turn全部原门槛通过；两stop完整noop且原失败保留 |
| 停车/转向组合，原六场加两交接 | 8400 / 42000 | 八场、全部原数值门槛及全程/前缀配对均通过 |
| 本轮独立实验累计 | **38800 / 194000** | 不包含只读分析；复用基线没有重跑 |

两次批次中断均完整保留，已完成物理没有重算：yaw-cap第一批完成左baseline800拍后写清单失败，续批复用它；plane第一批完成两stop1600拍后因跨方向初始观察中的±0符号比较失败，续批复用它们。不存在漏计或将复用计为新增。详见各公开包中的来源、failure、carry和补充合同。

## 原交接指定的release方案：已验证失败

[公开记录](../results/d1_driving_stability_development/heading_release_01/README.md)。release第400拍为±.245 m/s，第449拍首次为零；评分始终从原始用户tick400停令计算。基线与原G1保存态逐位一致，no-stop impulse全程不变。

前向raw速度RMS .05954→.06826 m/s，晚峰 .06087→.10602 m/s，晚路径 .05669→.06426 m；反向 .05992→.07023、.08635→.09319、.07617→.06760。虽减小早段腿/轮差，三项原停车门槛仍失败；不得改评分起点或把参考尾段视为用户仍在命令前进。

## 地面表示：新native-plane与旧heightfield不能混称

[接触诊断](../results/d1_driving_stability_development/heading_turn_contact_geometry_01/README.md)和[新plane基线](../results/d1_driving_stability_development/heading_flat_plane_01/README.md)。旧全零heightfield实际turn记录出现带载近水平法向；相同保存姿态换native plane后法向竖直。此为表示差异证据，不直接称MuJoCo引擎bug，也不是停车/转向失败的完整根因。

`scripts/d1_flat_plane_env.py` 为真实Opus实现、主代理收紧验证的独立plant/env：机器人/摩擦/solver/原控制器不改，plane物理无限但任务/查询区域有限。具有独立task/collision身份，拒绝旧heading及PPO checkpoint。60项非积分前检、188项模型不变量、37保存姿态及5600真实控制拍支持该平地表示。native observer读取真实step solver cache；同步endpoint的力和Jacobian另记相位，不跨相位相乘宣称功/能量。

原plane前检测试文件在归档后仅被Ruff调整一个from-import名称顺序；原测试原件保存在公开包source/tests且匹配原报告SHA。新的停车前检已核对除名称顺序外AST一致，其余输入/物理模块SHA相同。不要修改旧报告来伪装当前测试文件SHA一致。

原flat-plane四项失败没有被覆盖。旧障碍heightfield需要单独几何审计，不能把所有障碍替换成平地后宣称越障通过。

## 已通过的有限能力：固定停车腿阻尼

[代码、Opus原件、合同、四场完整轨迹和独立审计](../results/d1_driving_stability_development/heading_plane_stop_damping_01/README.md)。核心为 `scripts/d1_stop_leg_damping.py`，native-plane入口为 `scripts/probe_d1_heading_plane_stop_damping.py`。不要运行仍保留的旧hfield `probe_d1_heading_stop_damping.py` 8k入口；它从未作物理资格评测。

固定 `b=126.4374005337902 N·s/m` 来自名义模式推导；只在实际compute消费的forward从非零变exact-zero后，给四腿加入 `-b Jxᵀ(Jx qdot_leg)`，轮增量严格零。preview不触发latch；恢复非零命令立即关闭。delta加在未保护请求上再走原保护，原父compute/PI恰一次。没有release ramp、PI reset或新yaw cap。

| 原停车指标 | 前向candidate | 反向candidate | 原门槛 |
|---|---:|---:|---:|
| 全程raw velocity RMS m/s | .044766 | .044838 | ≤.05 |
| 晚段峰值速度 m/s | .005823 | .006336 | ≤.03 |
| 晚段累计平面路径 m | .002591 | .002736 | ≤.05 |

两stop其余原门槛也全部通过，无保护触发。两impulse全1200拍和1201状态/观察与对应plane-zero基线逐位相同，实际外扰±.1 N·m·s。root从4000保存状态独立重算阻尼，最大力矩等式误差1.33e−15 N·m，原始评分、native/PI/前缀、T/T+1和来源检查均通过。瞬时增量功不正不等同于积分区间被动性证明。

这仅支持同一plane/oracle/固定命令开发集合的停车；未验证与新转向、GUI、侧步、跳跃、障碍、噪声/延迟或实机组合。默认入口未替换。

## 转向：固定轮中心校正仍失败

[本轮转向候选](../results/d1_driving_stability_development/heading_plane_turn_center_01/README.md)。真实Opus模块 `scripts/d1_turn_center_compensation.py`，root薄env/runner `scripts/probe_d1_heading_turn_center.py`。只在raw forward恰0且raw yaw非0时，把水平heading方向轮中心腿速度 `+u/.087` 加入轮速未clip目标，系数1；原effective yaw cap .6，最终轮速clip±30，原PI/保护不变。

只用原raw纯转向50拍，关断不延长；callback一次原样返回，preview和compute同gate，compute快照不被下一拍prepare覆盖。转向严格前缀执行0..199、状态0..200；obs仅0..199，因为obs200已preview新目标。26纯测试和100原保存态单拍代数核验先于物理。单拍筛查不预测后续闭环；实际候选最大轮速目标约4.436 rad/s，高于原态影子最大2.524 rad/s。

左右heading峰值 .234134/.234213→**.179587/.179692 rad（约10.29°）**，仍大于原5°，其余原turn门槛通过；不采用。无轮速clip或扭矩保护。补偿后轮差几何等效yaw达约1.756 rad/s，不能把仍为.6的effective-body请求当轮差未变，更不能当实际bodyyaw。两noop stop全程逐位一致，原3项失败清单/指标保持；它们没有启用上一节停车阻尼，不能混淆两组结果。

旧plane轨迹的速度恒等式分解显示，轮中心平移对应约76.66%的wheel/body yaw gap，其余约20.89%是腿带动轮体转动，约2.44%为滑移。份额不是因果比例/改善预测；3–4.5秒的滑移和侧向接触反矩明显上升。失败后冻结系数，不追加参数扫描。独立保存态审计完成3,032,135项检查；另有root补充审计核对aggregate/per-case声明、位置逐位一致及cross-track评分。通过审计不改变原turn失败结论。

[失败诊断与直接body-rate替换资格核验](../results/d1_driving_stability_development/heading_turn_center_failure_plan_01/README.md)全部离线。root独立重现诊断报告SHA `761becefd00de15aceeb3b20d73addda3e0758d4d6bba3c29a4815b9708d6ba6`，以及200保存态代数报告SHA `dccd8764ace837b2a6ecd5ac5b602c9d77a8f217b8f77e4e57ee11a17639e55b`。候选pulse末heading确有改善，但腿中心速度增大；后半pulse接触净yaw矩更加反向。不能只依据wheel/body gap把剩余失败全部归给遗漏carrier转动。

把差动wheel误差直接换成body yaw-rate误差的代数恒等式闭合，影子力矩也没有超限；但这同时消除了body率不变时的差动轮速反馈阻尼。缺少loaded/weak-contact采样动力学证据，因此该直接替换为NO-GO，没有编写控制模块或进入新物理。未声称已证明数学不稳定。

## 后续固定纯turn腿阻尼：同样未通过

[公开包](../results/d1_driving_stability_development/heading_plane_turn_damping_01/README.md)。真实Opus模块 `d1_turn_leg_damping.py` 原件逐字节集成，只在raw纯turn的200..249拍加入同一body-x纵向阻尼，固定b，不加latch、不加center、不加post-stop。209姿态机械资格、26纯测试、104原保存态单拍计算先于物理；轮速PI请求差为0。原额定保护仍保留。

左右原始heading峰值为 **.216487/.216581 rad（约12.4°）**，仍失败；两stop全程noop且原三项失败保持。相较plane-zero，左pulse腿中心速度RMS从.05680降至.04441 m/s，body实际heading增量.06587→.08352 rad，效果不足以过门槛。native固定世界x/y分力yaw矩6.386/−5.500→8.511/−7.635 Nm，net均值.8863→.8758，末25拍反向net增强；不是旋转body轴分量，也不是因果/能量份额。无扭矩保护；每场最初4个无active-wheel-load子步属于与原基线逐位相同的settling前缀。

Root完成2,891,747项独立保存态/native/原评分审计，零新增物理，结果一致。此次只改torque，turn前缀必须包括 **obs200**，不能沿用center候选只到obs199的规则。累计物理预算已更新；不重跑入口。

## 已通过的有限转向能力：恢复原内层反馈

[公开包](../results/d1_driving_stability_development/heading_plane_turn_authority_01/README.md)。基线及turn-damper的raw .6 pulse中，原未截断请求始终约3.54–4.40 rps，而实际请求50/50拍均被截为.6，屏蔽了原body-rate反馈。新模块 `d1_turn_yaw_authority.py` 仅在raw纯turn恢复原 `servo + 4*(servo−body_rate)`；最终wheel±30、PI/antiwindup及原12Nm保护保留，没有新cap、动态headroom limiter、center或腿阻尼。

真实Opus编写核心，root仅修正文档/诊断边界/导出顺序并保留原件和diff；Astra ultra规划及独立审查。208保存态资格、29纯测试、104保存态控制调用（对账全0误差）先于物理。一次四场3200/16000完成，左右全部原门槛通过，heading峰值 **.063933/.063963 rad（3.663°/3.665°）**。两stop全程noop并保留原三项失败，不能混称为组合停车通过。

实际目标峰值11.85298rad/s，未clip；未保护请求峰值23.09777Nm，实际保护为12Nm，每turn有8个protected joint-intervals。native pulse净yaw矩均值+3.17557/−3.17612Nm。恢复反馈有有限效果，不证明接触/腿几何无关或鲁棒转向已完成。root独立重建全部3204保存态/PI/保护/native/原始评分，3,046,809项检查通过，零新增物理；base leg PD/support仍为归档值而非独立重实现。

turn前缀执行0..199、state0..200、obs0..199和native0..999逐位一致；obs200已改变preview，故不纳入。一次错误CLI路径在创建输出/构建环境/物理之前失败，零新增步；错误和正确原G1路径均已归档。此后唯一物理批次没有重跑。

## 停车/转向组合：固定八场已通过

[公开包](../results/d1_driving_stability_development/heading_plane_stop_turn_01/README.md)。实际Opus编写 `d1_stop_turn_composition.py`，原件逐字节集成；root实现薄env、runner和固定窗口评分器，Astra ultra制定并独立审查合同。单继承Authority，原PI一次；原Stop固定阻尼加在父未保护请求后，再走原保护。两个gate独立；持续零前进命令下，停车latch保持到后续转向，不为通过而禁用重叠。

39项纯检查先于物理。唯一八场完成 **8400/42000**：原两stop、两turn、两impulse全程与各自已通过的独立模块轨迹逐位一致，原G1全过；另两条正行→停→左转、倒行→停→右转各1400拍，也通过global和原数值stop/turn分段门槛，heading峰值 **3.068°/3.063°**。交接stop active1000拍、authority active50拍、两者重叠800..849拍，最大腿增量23.6273Nm；无wheel目标clip，每turn/handoff有8个protected joint-intervals，原12Nm保护保留。

交接只用前800拍评原stop，禁止后续静止稀释RMSE；turn使用物理600..1399拍及state600..1400，只平移评分索引，不重置原始航向参考。晚窗为endpoint1050..1399，平面位移原点S600。交接前缀保留state800、排除obs800和执行799的下一preview/horizon字段。旧stop日志未直接存全精度target，等价依据是同态/raw-servo/PI/request和不变inactive公式，不能声称缺失字段直接逐位比对；turn target/effective/error有直接逐位证据。

Root对8408保存态、PI、阻尼、保护、真实native/同步endpoint及原raw评分完成 **8,206,129** 项独立检查，零新积分/接触求解/控制调用；证据一致。其范围仍为plane/oracle/zero8的固定六场与两个已停稳后的转向历史。动态重新起步、同时停转、行进转向、四符号全组合、延迟/噪声和GUI/实机均未获资格。默认入口不变。

下一固定提案位于 `jump_obstacle_next_plan_01/next_contract.md`：新PD/PI链仅比较600拍恒高与600拍旧固定高度表，实际记录离地接触、collision轮底净空、COM和落稳，不移植LQR450N推力或调参。尚无新跳跃物理结果。

## 跳跃测量与新强化学习：实现中，尚未训练

用户进一步要求尽快回到强化学习主线，并让实际Claude承担更多编码、测试与检查。Astra ultra已冻结 `stability_20260920/rl_jump_training_plan_01`：新95维oracle任务、原8维物理残差、固定高度引导，640步流程检查后从零训练131072步，只评最终checkpoint的五个固定条件（zero/policy共最多6000步）。不加载或重跑旧三个65k，不重跑24场G1。1200/6000高度探针只要求记录有效；零动作未达到跳跃门槛不阻断RL，也不再转入连续经典调参。

[测量工具证据](../results/d1_driving_stability_development/jump_measurement_tools_01/README.md)已集成：`scripts/d1_jump_readiness.py`给出固定高度表、真实collision圆柱轮底几何及连续采样区间归并。实际Opus生成代码和测试；工具worker触及费用上限而未运行验证，root保留原件，修正错误的倾斜几何断言、转置fixture和临时路径依赖，并补全局部变换receipt。147项root非积分检查与Ruff通过，77冻结文件一致，新增物理步数为0。

readiness env/records/runner及RL task/env已有实际Opus初稿，尚未取得执行资格：发现runner无实际CLI执行分支、诊断字段/PI读取错误及RL父接口错误，已退回隔离Claude小任务修正并补测试。原稿、费用/工具回执和失败检查保留在工作目录。没有新跳跃、落地或强化学习结果，完整目标未完成。

## 仍必须完成的能力

1. 扩展固定组合域以外的可靠驾驶；已有八场通过，不代表行进转向、动态重启或人工试驾。
2. 新heading GUI仍无实现：9月14日Opus生成超时，没有代码。本轮没有重试GUI。旧课程GUI使用LQR/VMC，新heading任务为wheel_leg PD/PI，不可把两者的已知能力合并；也不能在heading环境外重复加航向P/D。
3. 人工试驾仍待真正用户验收，自动按键日志不等价。持续按键、A/D侧移、Q/E转向、Space、Shift等原需求继续有效。
4. 跳跃：旧记录机身抬升60.86 mm，而四轮同时最小轮底净空仅5.61 mm，且旧flat jump本身已是native plane，不能归因于heading旧hfield。旧LQR构造fresh-process另有511个辨识/settle控制步（2555子步），旧600拍只证明主trace；不要为检查旧GUI而实例化并漏计。新PD/PI没有旧vertical-force入口。需以真实轮底净空、持续离地及落地姿态/速度检验，随后逐级真实box台阶、坡道、碎石；动画完成和机身升高不算可靠越障。
5. 先稳定启停/转向/越障，再分档测实际速度、制动距离、通过率和翻倒/卡滞。现有Shift输入并不证明提速目标。
6. R仍是明确标注的simulator reset；物理翻倒自救没有实现。

物理探针都在不同新目录中执行一次，不能重跑入口覆盖已有目录。输出函数多用exclusive-create，额外日志关闭后的 `complete_manifest.json` 才是完整leaf清单。源文件或记录身份错误必须停止并保留实际已进入的native步，不reset重试或悄悄补跑。

## 续进：新高度探针完成，PPO 执行准备中

[完整新高度探针](../results/d1_driving_stability_development/jump_readiness_01/README.md)一次完成1200/6000，记录/共同前缀/保护/冻结77均通过。恒高与高度动作后段均落稳；高度动作只有12 ms卸载，四轮同时最小raw gap峰值1.4913875 mm（扣margin净空0.4913875 mm），没有达到20 ms和>21 mm原readiness门槛。前轮个别36 mm峰值不能替代同时净空。该失败是有效RL零动作对照，不阻断新训练，也不追加经典调参。

实际Opus初稿已由root接好原生plane、同步接触、wholeCOM、95观测和严格checkpoint接口。发现并修复请求前飞行追认及落地反弹增p两个reward漏洞，保留原稿、错误测试、修改diff和261项非积分测试XML。后三个大代码请求和一个小纯评分请求均超时且无代码；root直接完成必要集成，不能声称Claude完成了这些请求。

至此累计新物理 **40000控制/200000 native**；PPO640 smoke、131072正式及最多6000评估尚未运行。当前训练入口准备中；完整目标仍未完成，R仍仅复位。

## 新 PPO 流程检查完成

[640步新PPO smoke](../results/d1_driving_stability_development/jump_ppo_smoke_01/README.md)完整完成640/3200、5train/5epochs，第一场600步time-limit后自动reset并执行40步hold；root独立核对逐行episode/tick/native计数。第一场progress0/flight_seen false，没有跳跃能力结果。smoke权重已丢弃，正式从零seed62001，131072步，最终checkpoint才进入固定五条件zero/policy评估。源码/frozen77未变，累计40640/203200。训练/评估实现新增9+9项非积分测试通过。
