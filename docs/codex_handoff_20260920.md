# Codex 续接：平地停车通过开发门槛，转向仍未通过

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
| 本轮独立实验累计 | **24000 / 120000** | 不包含只读分析；复用基线没有重跑 |

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

把差动wheel误差直接换成body yaw-rate误差的代数恒等式闭合，影子力矩也没有超限；但这同时消除了body率不变时的差动轮速反馈阻尼。缺少loaded/weak-contact采样动力学证据，因此该直接替换为NO-GO，没有编写控制模块或进入新物理。未声称已证明数学不稳定。后续唯一待资格假设是单独的固定纵向腿阻尼用于turn：保持原轮速PI、目标和cap，先检查保存態的差动模式、力矩/保护、采样尺度及侧向接触约束。它不与失败的center补偿组合，尚未获得转向物理通过证据。

## 仍必须完成的能力

1. 可靠转向与停车/转向组合；所有未改动命令和评分继续保留。通过单模块不能自动组装后宣称同样通过。
2. 新heading GUI仍无实现：9月14日Opus生成超时，没有代码。本轮没有重试GUI。旧课程GUI使用LQR/VMC，新heading任务为wheel_leg PD/PI，不可把两者的已知能力合并；也不能在heading环境外重复加航向P/D。
3. 人工试驾仍待真正用户验收，自动按键日志不等价。持续按键、A/D侧移、Q/E转向、Space、Shift等原需求继续有效。
4. 跳跃：旧记录机身抬升60.86 mm，而四轮同时最小轮底净空仅5.61 mm。需以真实轮底净空、持续离地及落地姿态/速度检验，随后逐级台阶、坡道、碎石；动画完成和机身升高不算可靠越障。
5. 先稳定启停/转向/越障，再分档测实际速度、制动距离、通过率和翻倒/卡滞。现有Shift输入并不证明提速目标。
6. R仍是明确标注的simulator reset；物理翻倒自救没有实现。

物理探针都在不同新目录中执行一次，不能重跑入口覆盖已有目录。输出函数多用exclusive-create，额外日志关闭后的 `complete_manifest.json` 才是完整leaf清单。源文件或记录身份错误必须停止并保留实际已进入的native步，不reset重试或悄悄补跑。
