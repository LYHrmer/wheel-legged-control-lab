# Wheel-Legged Control Lab

[![tests](https://github.com/LYHrmer/wheel-legged-control-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/LYHrmer/wheel-legged-control-lab/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **求职向概览：[PORTFOLIO.md](PORTFOLIO.md)** — 30 秒定位、可核验结果、三个面试故事、简历条目与边界。

**30 秒版：** 16 自由度轮足机器人 + 同一层 VMC 基控，对照 LQR / 约束 MPC / 残差 PPO。残差 PPO 把高度 RMSE 从 99.3 mm 降到 22.4 mm（20 个配对种子、0 失败）；诊断驱动的偏航前馈把六任务跟踪 SSE 降低 28.2%；用 seed 复制证明单次训练结果的摆幅（7.4 点）大于臂间差异，因此不报告单点"最优"。

同一台轮足机器人，同一层 VMC 底层，三种外环策略的对照实验：**LQR、约束 MPC、有界残差 PPO**。下述成绩来自本仓库的MuJoCo仿真，未包含当前版本的实机验证。

**2026-10-18新增实测：** 完整任务、地形及键盘/失焦/停车验证通过；跳过无效推理的配对运行省去880/1600次控制预测，完整物理与控制状态精确相同。增益0.5相较增益1.0，在高速直行和转向两场降低了误差与力矩代价；相较纯控制器的六任务RL贡献门仍未通过。见[31–34实际结果](docs/local_optimization31_34_20261018.md)。

随后0.5增益整轮在内存保护线停止，11份完整读回可核验，整轮总消耗及部分坡道消耗仍未知，见[35中止记录](docs/b22_gain_suite35_interruption_20261018.md)。[37独立补验](docs/b22_gain_completion37_interruption_20261018.md)又收齐1场完整坡道，但下一场未达到6GiB启动线而停止，剩余3场未执行。已有五组配对中三组SSE超过纯控制器的1.02倍，因此即使补齐最后一组，固定0.5增益候选也无法通过原RL贡献门；见[必要条件分析](docs/b22_half_gain_bound_20261018.md)。两次中止的完整矩阵资格字段保持未知。

[36轻量NumPy后端实测](docs/b22_numpy36_results_20261018.md)通过192条固定输入，最大动作误差5.96e-8；一次冷启动对照的峰值RSS约670→37MiB、冷启动至预测完成16.64→3.96秒。[40保存轨迹复核](docs/b22_numpy40_results_20261018.md)核实原1600拍直行，[41独立补验](docs/b22_numpy41_results_20261018.md)又完成1600拍转向；两场原任务、安全和停车门及SSE/力矩代价≤1.001的后端兼容门均通过。41只新增1600控制拍，复用旧直行另列；本次不授予RL贡献或完整矩阵资格，默认后端未替换，冷启动数字不代表全控制回合的速度。

[38控制链启动核查](docs/b22_numpy38_interruption_20261018.md)因MuJoCo包来源清单缺失而中止。[39独立运行](docs/b22_numpy39_interruption_20261018.md)通过修复后的冷准入和1600拍直行，随后保存结果读取器因要求Torch库列表非空而中止，转向未启动。[40复核](docs/b22_numpy40.md)只修正这一前提并完成原直行归档核验，新增物理和NN执行均为零；39原中止状态与未知总数保留。[奖励敏感性分析](docs/b22_reward_signal_analysis_20261018.md)量化了平地跟踪奖励接近上限的现象及训练来源，作为后续等预算对照的依据。

[42有界训练对照](docs/b22_tracking_reward42.md)的[05完整结果](docs/b22_tracking_reward42_revision05_results_20261018.md)已完成：两组各32768拍训练，两个final与共享zero共18场任务、停车均通过。提高跟踪惩罚的deficit8组相对zero总SSE降低3.11%，相对本次control降低6.85%、力矩成本更低；已有4对满足原逐对门，但合池SSE比0.968869仍未达到0.85的RL贡献门。坡道获益明显，粗糙地面的速度误差与转向偏航误差仍需改善。整轮实际96136controls、52.8分钟，无OOM，默认模型尚未替换。[02](docs/b22_tracking_reward42_revision02_results_20261018.md)/[03](docs/b22_tracking_reward42_revision03_results_20261018.md)接口失败与[04](docs/b22_tracking_reward42_revision04_results_20261018.md)内存中止的历史记录分别保留。

[43 左右镜像候选的数值准备已完成](docs/b22_symmetry43_numeric_results_20261018.md)：绑定42 deficit8真实完整权重，384次SB3与384次NumPy推理最大误差2.98e-8，192次镜像组合也通过。观测映射通过原编码器对照，65组URDF运动学检查支持关节符号；该候选随后已完成[物理任务评测](docs/b22_symmetry43_physics_results_20261018.md)：原六任务与停车18/18通过，镜像平均的合池SSE比zero降低7.56%（比值0.924429；同权重不镜像为0.968869），逐对6/6不劣于1.02倍，力矩成本比1.0575；但仍未达到原RL贡献门0.85，且降幅集中在坡道任务，转向与粗糙地面没有改善。默认模型尚未替换。[保存数据诊断](docs/b22_symmetry43_yaw_diagnostic_20261018.md)显示：flat_1p2_yaw偏航SSE约95%是闭环偏航增益仅约0.63造成的稳态欠幅，三种策略都没改变它，且训练里|偏航指令|≥0.1的只占5.19%；这是相关性而非已证因果，修改基控制器与RL贡献须分开计。随后在基控制器上做了偏航指令前馈的[真实物理对照](docs/b22_yaw_feedforward44_results_20261018.md)：`3.9·指令+4(指令−实测)`使原六任务合池SSE比冻结基线降低28.2%（0.7178，6/6通过，转向任务SSE降至0.18倍，无转向任务逐位不变），预注册开发集与镜像转向案例同样降至约0.18倍；整体提高反馈增益则使合池变为2.05倍。这是基控制器改进而非RL贡献，单种子，预注册封存最终集（幅值0.28、速度1.1）一次性确认同样降至约0.17倍、无转向任务逐位不变；已作为新模块并入`src`（冻结控制器及其指纹未改动），未部署、未实机。在这个前馈基线上，已有42权重配镜像的合池比为0.8976；随后在前馈基线上做了[提高转向样本占比的两臂重训练对照](docs/b22_yawtrain44_results_20261018.md)（训练与独立读回有效）：转向课程臂raw合池比0.9024且逐对6/6、优于同基线原课程臂的0.9242，但**没有任何组合越过0.85门**，差距在单次训练的噪声量级内，不能排序。再训练粗糙地面强化臂（粗糙样本9.2%→29.8%）为明确负结果：raw合池比0.9974、逐对仅1/6、成本1.2091越门，rough_0p35反而变差，见[45结果](docs/b22_roughtrain45_results_20261018.md)。随后按[奖励形状设计](docs/b22_logsse46.md)训练了对数平方误差臂（β=8、cap=4，单变量对照44转向课程臂）：[46结果](docs/b22_logsse46_results_20261018.md)为目前最好——镜像合池0.8834、逐对6/6、成本1.026，rough_0p35由19.75降到17.35；**仍未过0.85门**，剩余差距集中在rough与ramp。[seed复制](docs/b22_seed47_results_20261020.md)显示单换seed即可让β=0镜像在0.9324与0.8586间摆动（7.4点），大于臂间差，按预注册分离规则**形状效应未获证实**，46的0.8834只是单次观察。[开源四足RL对照与诊断](docs/b22_open_source_review_20261020.md)（同机器人官方D1仓库等）显示：我们比开源少4500–60000倍样本，跟踪核函数反而比开源更尖，残差动作零饱和、仅用量程4–9%；[优化后的方案](docs/b22_plan48_20261020.md)据此改为先修评测方法、再用2×预算做一次决定性实验，不再新增1×预算下的奖励臂。2×预算实验（48）最终**归档**：13次硬编码失败加一次系统库漂移后不重绑来源清单、不产出结论，理由与记录见[48](docs/b22_budget48.md)。[当前结论与交付状态](docs/b22_final_status_20261021.md)汇总了已站得住的结论、撤回项与未解决问题；方法学结果已整理成可直接使用的[论文小节](docs/b22_paper_section_20261022.md)（含可写/不可写表述与表图规划）；47策略的[样本外检验](docs/b22_sealed47_results_20261021.md)显示原六任务优势未迁移（0.8586→0.9284），两种奖励形状在样本外打平。

```
D1 全机  23 nq / 22 nv / 16 执行器 · 物理 500 Hz · 决策 100 Hz（一个控制拍 5 × 2 ms）
  └── VMC 底层  D1VMCController
        ├── LQR 外环         D1LQRVMCController
        ├── 约束 MPC 外环     D1MPCVMCController
        └── + 16 维有界残差 PPO

6 状态平面教学模型  3 关节(base_x / body_pitch / leg_extension) · 2 执行器(wheel_force / leg_force)
  └── LQRController / LinearMPCController  + 2 维有界残差 PPO
     用来先把线性化、LQR、约束 MPC 和 PPO 的公式对到代码上
```

两层共用一套评测纪律：固定场景 + 配对随机种子 + 预注册门槛 + 失败原样保留。

## 两组对照，结论相反

### 平面 6 状态模型：残差 PPO 大幅改善高度调节，约束 MPC 相对 LQR 无差别

20 个**配对随机种子**，均值 ± 标准差，0 失败（[randomized_audit.md](results/benchmark/randomized_audit.md)）：

| 外环 | 平均回报 | 速度 RMSE (m/s) | 俯仰 RMSE (deg) | **高度 RMSE (mm)** |
|---|---:|---:|---:|---:|
| LQR | 0.783 ± 0.050 | 0.565 ± 0.163 | 2.471 ± 0.655 | 99.28 ± 27.20 |
| 约束 MPC | 0.783 ± 0.050 | 0.563 ± 0.162 | 2.404 ± 0.637 | 99.29 ± 27.19 |
| LQR + 残差 PPO | **0.881 ± 0.034** | **0.480 ± 0.130** | 2.557 ± 0.716 | **22.38 ± 7.69** |

- **约束 MPC 相对 LQR 统计上无差别**：平均回报到小数点后三位相同，速度 RMSE 相差 0.002 而标准差是 0.16。代价是每拍 0.42–0.64 ms 求解时间（LQR 为 0）。在这个任务和这个模型上，约束处理没有换来可测的跟踪收益。
- **残差 PPO 的收益几乎全在高度调节**：99.3 → 22.4 mm（4.4 倍），速度 RMSE 只改善 15%，俯仰 RMSE 还略微变差（2.471 → 2.557，仍在标准差内）。残差是 2 维有界量，叠在 LQR 输出上，**智能体从不直接给力**。

三个固定场景的逐项数值在 [metrics.md](results/benchmark/metrics.md)。其中模型失配 + 延迟场景最能说明问题：

| 外环 | 速度 RMSE | 高度 RMSE | 求解 P95 |
|---|---:|---:|---:|
| LQR | 0.543 | 125.23 mm | 0.000 ms |
| 约束 MPC | 0.538 | 125.23 mm | 0.637 ms |
| LQR + 残差 PPO | 0.429 | **33.76 mm** | 0.000 ms |

失配下两个经典控制器的高度误差同时劣化到 125 mm，残差策略守住 33.76 mm。

> **版本容差**：这些表产出于 2026-08，当时未记录 MuJoCo 版本。2026-10-11 在 **MuJoCo 3.12.0** 上以同一 seed 重算，俯仰 RMSE 六项逐位相同、`success`/努力量 RMS/最大俯仰逐位不变，但**速度 RMSE 偏移约 0.7–1.3%、高度 RMSE 偏移约 2.9%**。定性结论不变（MPC 与 LQR 仍统计无差别），但**比较控制器必须在同一 MuJoCo 版本下进行**。逐项差异与分析见[复现说明](results/benchmark/REPRODUCTION.md)。

### 全机 D1：同样的残差思路没能通过预注册贡献门

把残差从 2 维扩到 16 维、从平面模型搬到 23 nq 全机之后（[C22 报告](docs/main_route_execution_20260930_22.md)）：

| 判据 | 结果 |
|---|---|
| 保留集归一化平方跟踪误差，B（残差）/ zero（零残差） | pooled SSE **1.038768**，即比零残差差 3.9% |
| 力矩成本比 B / zero | **1.040694** |
| 保留集平均误差 | 比 zero **高 3.60%** |
| 六项里满足"误差不差于 zero 的 1.02 倍" | **1 / 6**（门槛要求 ≥4 项且 pooled SSE ≤ 0.85） |
| **`RL_contribution_passed`** | **false** |
| zero 失败而 B 成功的案例 | **0 个** |

课程对照本身是通过的：B 相对上一代课程 A 在保留集上平均归一化平方跟踪误差降低 13.50%、力矩成本降低 7.67%。**但"课程更好"不等于"RL 超过基控"**，这两个门是分开判的。

### 这个反差说明什么

同一个"有界残差叠在经典外环上"的结构，在 3 关节 2 执行器的平面模型上把高度 RMSE 降 4.4 倍；换到 16 执行器的全机上，16 维残差打不过零残差。这不是"RL 不行"，而是**残差维度与被控系统复杂度同时上升后，有界残差相对一个已经调好的基控失去了优势**。

这个负结果推动了独立项目：[ddt-d1-rl-lab](https://github.com/LYHrmer/ddt-d1-rl-lab) 改走**纯自训 RL、不叠基控**的路线。那边的早期五个版本未过平地低速门；截至2026-10-18，25版随机延迟训练已在原36例开发矩阵中通过33例，停车均通过，剩下60ms延迟下的三例转向误差超线，完整候选资格仍未通过。见[25版实际结果](https://github.com/LYHrmer/ddt-d1-rl-lab/blob/main/docs/low_memory_delay25_results_20261018.md)。两个仓库具有独立git历史和模型来源，成绩分别报告。

## 归属边界：1.60145 m/s 是基控成绩

B22 固定直行平地保持均速 **1.60145 m/s** 是**仿真**结果，来自 **C18 基控 + 16 维残差**的组合，而同一轮的 `RL_contribution_passed=false`。**不要把它读成"RL 跑到 1.6 m/s"。** 它也不是横移速度、不是实机结果。

同样需要分开看的还有：

| 结论 | 范围 | 不是什么 |
|---|---|---|
| 七项固定场景全部通过（C18） | 工程资格 | 不是 RL 收益 |
| B 相对 A 误差 −13.50%、成本 −7.67% | 课程对照 | 不是 RL 超过基控 |
| 真实 15 mm 台阶有界验收 | 真实碰撞几何，仍是 MuJoCo | 不是实机 |
| A/D 横移 0.07317 m/s | 运动学最小时间剖面推算 | **未测得任何实际横移速度** |
| R 键复位 | 仿真扶正 | 不是物理自救 |

整体目标"稳定直行 → 可靠越障 → 提高速度"**未完成**。

## 从哪开始读

想看**控制方法对照**：本页上面两张表 → [评测协议](docs/evaluation_protocol.md)（场景、成功定义、统计口径）→ `src/wheel_legged_control/d1/hierarchical.py`（`D1LQRVMCController` 与 `D1MPCVMCController` 共享的基类）。

想**按实验找证据**：[实验索引](results/README.md)把 121 个结果目录按模型分两层列出（平面教学模型 / 全机 D1），每条写明"可以检查什么"，并标出哪些门没过。

想**从头学一遍控制**：`docs/index.md` 里有一条 14 步递进路线，每步都有可运行代码和验收问题：

| 步 | 学到什么 |
|---:|---|
| 1 | 六状态模型、线性化、LQR 与约束 MPC（[learning_guide.md](docs/learning_guide.md)） |
| 2–4 | 手算 TD/GAE、在小 actor-critic 上做一次真实 SGD 更新、纸笔检查题 |
| 5–6 | 82 维观测、八维动作、奖励量纲；成功定义与配对统计 |
| 7 | 单转轴辨识台架，单轮 PI 与前馈对照 |
| 8 | IMU/编码器估计，测量包与融合器分开 |
| 9 | **约束接触力分配，整机任务空间 QP** |
| 10 | 四帧观测、两种延迟、固定预算与失败配对 |
| 11–13 | 合成辨识进入整机通道；仅用位置辨识的可辨识性极限；从带噪位置估速 |
| 14 | **漏建模摩擦如何改变延迟选择及前馈效果** |

想**直接跑**：见下一节。研究过程的逐轮记录（C14–C38 合同、断点、失败）在 [研究日志](docs/research_log.md)，[文档导航](docs/index.md) 是唯一的入口地图。

## 跑起来

本机先检查 B22 v27 环境身份（不加载模型、不打开窗口、不执行物理）：

```bash
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22_v27.py --check
```

2026-10-18 已用 v27 完成一次真实 GUI 与 `flat_1p6` 固定直行/停车演示：1,600 控制拍、8,000 正常物理步，持速均值 1.601452 m/s，末秒平均绝对速度 0.000207 m/s，三张真实截图与独立读回均通过。见[本次收尾结果与证据](docs/b22_closeout30_result_20261018.md)。这次限定演示不重新授予完整控制资格，也未复验键盘、坡道、台阶或实机。迁移白名单与来源登记仍见 [v27 说明](docs/b22_runtime_v27_20261016.md)；旧 v26 入口和清单保留原字节。

操作者需要运行时启动 B22（默认 1.6 m/s 固定直行脚本）：

```bash
rtk proxy /usr/bin/python3 -B /home/lyh/wheel-legged-control-lab/scripts/play_b22_v27.py
```

加 `--mode keyboard --seconds 60` 可用 W 前进、Q/E 转向、Space/X 停止、R 仿真复位、Esc 退出；键盘目标 1.2 m/s，**A/D 横移与跳跃未开放**。结果目录自动生成。固定坡道、粗糙地面和凸起等脚本见 [B22 启动与操作说明](docs/b22_gui.md)，资格、完整证据及失败记录见 [GUI 与 15 mm 台阶交付报告](docs/b22_gui_true15mm_20261001.md)。

重算平面模型的三方对照：

```bash
pip install -e '.[rl]'        # 完整复现三行；只装基础依赖时运行 LQR / MPC 两行
python -m wheel_legged_control.experiments --output results/my_benchmark --seed 11
```

残差权重 `results/residual_ppo/model.zip` 已在库中，所以装好 `[rl]` 后 fresh clone 可以复现全部三行；只装基础依赖时降级为 LQR / MPC 两行，并在 stderr 写明原因。每次运行都在输出目录写 `provenance.json`，记录 git 提交、各包版本与 checkpoint 的 SHA。（2026-10-16 之前这里写的是"`[rl]` 不是可选的"：权重存在时会无条件导入 `stable_baselines3`。用户解冻 `experiments.py` 后已修复，见[复现说明](results/benchmark/REPRODUCTION.md)。）读数前先看复现说明里的版本容差。

新实验可额外用[源码清单入口](docs/source_and_metrics_integrity_20261016.md)绑定尚未提交的源码内容，并在结束后核验；旧冻结 helper 的指纹不包含这部分内容。同次工程检查还修复了延迟日志缺拍可能被误认作完整时域的问题，历史 24 案例的 216 项摘要值复算保持不变。

旧连续传感器 PPO 任务的[完整时域配对入口 v28](docs/continuous_comparison_v28_20261016.md)只读已审计的小 CSV 与身份清单。失败、缺失或不等完整时长的配对明确记为 `INCOMPARABLE`；已有 33 个配对和逐训练种子留出均值逐值保持，未新增控制评测。

[控制器状态诊断 v29](docs/controller_state_v29_20261017.md)进一步流式读取 18 个完整留出配对：分别统计 PI 积分、Fx/Fz 和动作/力矩饱和，保留实际分母与缺失字段。该批三个训练 seed 的竖直动作全程触及裁剪端，但力矩未饱和；转向积分反号在零基线也存在。这是 continuous 历史日志的描述，不是 B22 新评测或因果结论。

已安装的 `d1-rl` 属于旧 85 维版本；当前入口见上。下文保留可运行的 82 维教学路线、旧键盘课程及其独立实验结果 —— 它们和当前 99D/16D 研究组合的入口、checkpoint、按键与资格范围不同，版本地图见[文档导航](docs/index.md)。原始轨迹与源码快照的下载方式见[复现说明](docs/reproducibility.md)。

项目与本末科技的官方代码无关。

## 82 维教学路线控制流

命令有三个量：前进速度、偏航角速度、机身离地净空。每拍先发布状态、预览命令与基线，编码
82 维观测，策略给出动作，控制器输出 16 轴力矩，再做五次物理积分。
[控制循环](src/wheel_legged_control/d1/control_loop.py)要求先 `prepare(command)` 再
`step(action)`，逐拍时序和坐标系细节在[命令条件轮足实验](docs/locomotion_lab.md)。两种控制
结构共用这套循环：

| 结构 | 动作 | 归一化 ±1 对应的尺度 |
|---|---|---|
| `wheel_leg` | 8 维：四腿轮心伸缩修正、四轮速度目标修正 | ±4 cm、±4 rad/s |
| `lqr` | 2 维：纵向、竖直机身力残差 | ±11.25 N、±20 N |

八维方案同时换掉了低层控制结构，它和两维方案的差别不能全部算到动作维数上。每种结构各留一份
自己的零残差对照。运行入口的状态来源两选一：`oracle` 读真值和碰撞几何高度；`sensor` 用 IMU/编码器融合
加理想接触开关拟合支撑面，需要已知的初始放置姿态，没有定位、在线 bias 估计或滑移模型。

## 装好并跑第一次

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[rl,dev]"
```

上面先安装 CPU 版 PyTorch。其他平台的安装方式见
[PyTorch 官方说明](https://pytorch.org/get-started/locally/)。
第一次跑零残差，12 s 平地，不开窗口：

```bash
python scripts/run_d1_locomotion.py --source oracle --seconds 12 \
  --output results/first_flat_zero
```

不给 `--policy` 就是零残差，不给 `--keyboard` 就不打开 viewer，无桌面环境也能跑完。输出
目录必须是新的，脚本不覆盖旧实验。跑完先看 `summary.json` 的终止原因，再看 `telemetry.csv`
里已执行的命令和误差、`states.npz` 的真实状态与逐子步力矩，`manifest.json` 记录每个产物的
SHA-256。

换 60 s 的开发道路：

```bash
python scripts/run_d1_locomotion.py --source oracle --seconds 60 \
  --terrain development --terrain-index 0 --output results/first_dev_road
```

地图固定 12×6 m，出生点在 `(-3.8, 0)` 的平地。回合跑满 60 s 不等于走过起伏路段，先在轨迹
里确认实际遇到了什么地形。

加载 checkpoint 要同时给 `--policy` 和它的 `--metadata` sidecar，脚本会校验 baseline、
各项 schema、history 长度、观测/动作空间、控制器增益和模型 SHA-256，任一项不一致直接拒绝；
训练与回放命令在[命令条件轮足实验](docs/locomotion_lab.md)。只加载自己生成或信任的 SB3
文件，它的序列化格式能执行 Python。

## 键盘接口

想试坡道、乱石、台阶和跳跃，使用项目已有的[地形课程](docs/interactive_course.md)：

```bash
python scripts/run_d1_course_drive.py --zone rough --seconds 600 \
  --output results/my_course_drive_01
```

该入口使用 `W/S` 前后移动，每按一次 `Shift` 循环切换 1/2/3 档（前进目标 0.30/0.40/0.50 m/s，
倒车 0.20/0.35/0.50 m/s），长按不会连续换挡，`R` 复位后回到 1 档。
`Q/E` 调整目标朝向、`Space` 请求受保护跳跃、`R` 扶正复位到
当前区域起点、`X` 取消运动、`T/G` 调整高度，`1..6` 重置地形区域。长按 `A/D` 在平地上通过
四腿依次抬落侧移，松键先落脚再交回轮式控制。默认快步一次约 3.7 cm，需要约 10.6 秒；
`--side-step-profile conservative` 可切换约 41 秒的保守步态。暂不支持坡道或碎石上的侧步。
试侧步时先按 `1` 回平地起点，停稳后长按 `A` 或 `D` 约 12 秒。
实测对照与 Claude Opus 的实现贡献见[侧步开发记录](results/d1_side_step_development/README.md)。
连续三步侧移约 10.8 cm 时仍有约 1.8 cm 纵向漂移，属于低速平地原型。
窗口自动测试及失败记录见[GUI 验证](results/d1_side_step_gui_validation/README.md)。
软件渲染较慢时可添加 `--render-quality low`；自动通过仍需人工确认操作体验。
三档平地实测约 0.31/0.41/0.50 m/s；松键清零目标后仍有制动距离。地形起步、制动及失败候选见
[换挡验证](results/d1_course_speed_gears/README.md)，尚不能保证任意台阶直接通过。
R 是仿真复位，不称物理自救。
它使用原有 LQR/MPC 课程控制器和 oracle 状态，不加载主线 PPO。
乱石为固定方块。各区重置会分段记录，不计作连续越障。

项目协作分工、Opus 的实际贡献和后续任务选择见[协作规则](docs/claude_collaboration.md)。

试已有 82 维强化学习教学路线时使用：

```bash
python scripts/run_d1_locomotion.py --keyboard --source sensor \
  --output results/my_keyboard_run
```

| 按键 | 按住时的作用 | 范围 |
|---|---|---|
| `W` / `S` | 持续前进 / 倒退 | `-0.20…0.30 m/s`，目标逐渐变化 |
| `A` / `D` | 持续左转 / 右转 | `±0.25 rad/s` |
| `R` / `F` | 连续升/降机身净空 | `0.43…0.48 m`，初值 0.455 m |
| `Space` | 前进与偏航命令归零 | 净空保持不变 |
| `Esc` | 结束回合 | 终止原因记为 `keyboard_escape` |
| `C` | 恢复相机角度与距离 | 鼠标左键拖动旋转，滚轮缩放 |

松开某个运动方向的按键后，该方向的命令归零；窗口失去焦点时清掉前进与转向命令。
按住 W 和 A 可边前进边转向，相反方向的键相互抵消。刹车仍然由控制器和物理决定，这些是软件命令限幅，
不是急停。82 维教学入口没有跳跃，也没有横移。D1 四轮没有转向机构，`A/D` 靠左右轮差速改航向。旧入口
`wheel-legged-d1-play` 的 `Space` 是跳跃，属于历史 42 维路径，两套按键不要混用。

键盘默认使用带缓坡、起伏与小台阶的开发道路，可用 `--terrain flat` 切回平地。
`Esc` 主动结束时当前 CLI 返回退出码 1，需结合
`keyboard_escape` 判断。[主线演示与手动试驾](docs/demo_showcase.md)给出完整命令及待填验收表。

## 留出道路结果

六个模型 = 两种结构 × 训练种子 24000/25000/26000，每个固定 32768 个样本，oracle 状态源、
单帧 82 维观测，训练用四张固定道路，开发和留出各两张，checkpoint 一律取预算末端，没有按
留出成绩挑过。下表对两张留出道路等权平均，PPO 再对三个训练种子等权平均。

| 控制方式 | 速度 RMSE m/s | 高度 RMSE mm | 机械活动量 W | 质量达标 |
|---|---:|---:|---:|---:|
| LQR 零残差 | 0.07969 | 15.30 | 47.44 | 1/2 道路 |
| LQR + PPO | 0.08482 | 15.99 | 44.63 | 2/6 种子×道路 |
| 轮腿零残差 | 0.04016 | 9.84 | 16.03 | 2/2 道路 |
| 轮腿 + PPO | 0.03328 | 11.76 | 20.50 | 6/6 种子×道路 |

轮腿 PPO 把留出速度 RMSE 从 0.04016 降到 0.03328 m/s，同时高度 RMSE 增加约 1.9 mm、机械活动
量增加约 4.5 W。LQR 结构在开发道路上达标 5/6，迁到留出只剩 2/6，这项退化保留在结果里。这批
证据支持继续研究八维动作接口，不支持“RL 已经全面超过传统控制”，也不能当成动作维数的消融。

固定 oracle 时，同一路面的两个评测 seed 产生了逐字节相同的 CSV 和 NPZ，它们只是重复标签，
不是独立复现，报告因此不给置信区间。完整指标、逐道路结果和两段 60 s 状态录像见
[实验报告](results/d1_v3_locomotion_report/README.md)，两段录像走的道路不同，不能当逐帧公平
对照。

本轮另做了[同条件 60 秒对照片](docs/demo_showcase.md)：零残差与预定八维 PPO 共用道路，
初态和全部 6000 拍命令逐项核对。该案例中 PPO 的净空误差减小，速度和偏航误差增大。

## 现在还不行的部分

新增[九模型延迟对照](results/d1_v3_delay_ablation/README.md)：20 ms 下单帧、四帧、
四帧加随机化分别完成 2/12、0/12、1/12；30 ms 下均为 0/12。
增加历史与这次随机化训练没有带来稳定改善，全部模型及失败记录保留。

- 整包测量延迟加到 20 ms 后，融合状态下的轮腿零残差在开发道路四个案例全部失稳，分别在
  2.05、1.20、0.91、4.41 s。把轮速增益改成 `0.55 / 1.5 / 4` 后，8 s 平地转向探针的累计
  转角达到指令积分的 70.8% 和 70.7%，但完整开发环境的四个案例仍在 3.1 s 内失稳；默认增益
  保持 `2.2 / 3.0 / 4.0`。记录在 [延迟诊断目录](results/d1_v3_delay_diagnosis/)。
- 曝光指标只检查机身/轮心投影处的地形高度和坡度，是几何曝光，不证明每个轮子保持接触，
  也不证明越过了整个障碍。
- 机械活动量是各子步 `sum(abs(力矩 × 关节速度))` 的均值，单位 W，不是电池功率。
- 没有 ROS2 链路、实机参数辨识和 sim-to-real。融合估计依赖已知初始位姿与理想接触开关。
- 轮腿控制按轮心 Jacobian 映射支撑力，没有滑移反馈。两维力残差的接触分配路径仍有静力近似。

## 文档

[文档导航](docs/index.md)是唯一的入口地图，分成学习顺序、当前 82 维主线、历史
42/44/45 维路线与 QP 原型、证据位置四节。常用页面：

- [命令条件轮足实验](docs/locomotion_lab.md)：主线的观测、奖励、八维动作含义，以及从真实
  PPO 更新记录复算 GAE 和概率比的步骤。
- [学习指南](docs/learning_guide.md)：LQR、MPC、GAE 与残差 RL 的基础练习和验收问题。
- [评测协议](docs/evaluation_protocol.md)：场景、成功定义和统计口径。
- [延迟与短历史](docs/delay_learning_lab.md)：四帧观测、两种延迟、随机化与失败配对。
- [执行器辨识移植](docs/actuator_transfer.md)：合成台架参数进入整机，补偿收益与代价。
- [仅位置编码器辨识](docs/encoder_identification.md)：PACE 启发的单轮练习，从位置轨迹拟合
  动力学，再检查未见激励预测与 PI 前馈；与整机主线分开评测。
- [编码器速度反馈](docs/encoder_feedback.md)：从带噪位置在线估速。tracking/reversal 的跟踪
  误差高于独立速度参考，stress 的误差下降伴随更多力矩限幅。
- [摩擦与延迟辨识](docs/friction_delay_identification.md)：遗漏摩擦项后，54 次拟合有 24 次
  延迟选错；低幅值激励在失配对象上的留出预测和控制更差。
- [同底座动作比较](docs/shared_action_learning_lab.md)：共享两维与独立八维动作共用轮腿控制器，
  区分策略概率、物理动作、参数量与配对评测。
- [连续训练的预算曲线](docs/budget_learning_lab.md)：每模型一次训练、四档存档，检查保存时机
  是否在 PPO 更新之后；[正式结果](docs/d1_budget_study.md)覆盖 200 例，198 完整、192 达标，
  增加预算未带来一致改善。
- [地形课程与受保护跳跃](docs/interactive_course.md)：已有地形接入新的键盘界面，支持目标朝向和仿真扶正复位。
- [ATEC 项目参考方案](docs/atec_reference.md)：视觉导航与课程训练可借鉴之处，及权重不能直接加载的原因。
- [ATEC 课程用于 D1 强化学习](docs/atec_rl_curriculum.md)：任务拆分、课程晋级条件及实施顺序。
- [PPO 减速诊断与共同轮速均值实验](docs/d1_wheel_common_mean_study.md)：闭环动作干预、Opus 策略模块及配对训练。
- [稳定直行、越障与速度优化](docs/driving_optimization.md)：偏航来源、真实航向外环和松键制动改进，记录尚未通过的能力。

## 模型来源与许可

D1 的 URDF 与 STL 来自 Apache-2.0 授权的
[`Rangens/WMP-D1-loco`](https://github.com/Rangens/WMP-D1-loco)，固定到提交
`540e98d0a0c2212bc74908b98088b870a79e2f53`，原许可证随资产保存在
`src/wheel_legged_control/d1/assets/LICENSE-WMP-D1-loco.txt`。结构和量级参考本末科技
[`D1 开发手册`](https://d1-development-manual-cn.readthedocs.io/zh-cn/latest/)与公开
[`DDTRobot`](https://github.com/DDTRobot) 仓库，没有复制许可证不明确的代码。本地
`d1h_wcd_description` 的检查记录在 [d1_model_card.md](docs/d1_model_card.md)。仓库自身代码
按 MIT 发布。

## 测试

```bash
pytest
ruff check src tests examples scripts
```

pytest 守的是接口和验收条件，例如同步采样的有限差分检查、观测与奖励的手算值、键盘命令逻辑。
键盘既有可注入时钟的单元测试，也已在真实 MuJoCo 窗口完成自动按键验证；命令变化与 CSV
逐项核对。[窗口证据与人工记录表](docs/demo_showcase.md#现有-gui-证据与待填记录)区分这两类
检查，人工试驾仍未验收。原始记录另有一份独立复算：

```bash
python scripts/audit_d1_locomotion.py --matrix-root results \
  --output results/my_phase4_arithmetic.json
```

它只用 NumPy 和标准库，不导入仿真器或训练框架，只写新文件；哈希一致只说明字节一致。
