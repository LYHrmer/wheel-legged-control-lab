# 下一终端冻结合同：原生候选异常的零积分诊断与报告修复

日期：2026-09-22。独立规划/复审：实际 GPT-6-astra ultra。状态：本文件冻结的是下一次工作的边界与验收条件；本文件的创建不代表下列实施已完成。

## 1. 唯一目标与当前结论

建立一个可复核、能够保留失败原因的纯离线诊断层，把“存档忠实性”“所有候选的几何资格”“正载荷候选几何观察”“冻结任务资格”分开表达。修复对象仅为诊断输出将已定位的几何拒绝折叠为 `unhandled_exception:ValueError` 的信息损失。不得把拒绝改成通过，不得在本合同中修改物理接触生成器。

已经确认：MuJoCo 3.12.0 在保存的 native 3486 before 姿态下，不经过控制器或约束求解，静态重建原样产生 contact 5 的异常向下法向。该候选原始局部六维载荷为精确零；其框架、身份、记录相位和符号转换与原生输出一致。圆柱支持函数另证轮子整体位于箱顶以上。具体数值算法内部失效分支尚未确认。

原 `single_15mm_box/score.json` 保持 `record_valid=false`、`task_passed=false`。平面场景原通过结果保留。原单箱 readiness 未获资格；RL、训练、残差控制、增益/命令调整全部关闭。

## 2. 固定输入与不可变证据

以文末 SHA256 锚点为输入。`single_step_readiness_01/manifest.json` 中的全部 24 个文件必须先逐项验 SHA256 与字节数；`protocol.json` 的全部 source hashes 必须保持匹配。保持 frozen77、原合同、原两场目录、原评分、原静态报告、原观察图和报告字节不变。

允许在新的工作输出目录独占创建补充材料；允许新增纯诊断脚本与纯测试文件。不得改写当前受冻结 source hashes 约束的 scorer、integrity、geometry、runner、recorder、plant、env 或控制器。文档只引用补充结论，不回写旧分数。

## 3. 明确预算与禁止入口

本合同预算：control attempts/completions = 0；native attempts/returns = 0；`mj_step`/`mj_step1`/`mj_step2`/`mj_forward`/`mj_inverse`/`mj_setConst` = 0。亦不新增 `mj_kinematics`、`mj_comPos`、`mj_collision`、`mj_geomDistance` 调用：已有四姿态静态复现足够，本轮直接读取其 JSON。

不创建环境、plant、MjModel/MjData，不调用 reset，不跑 rollout，不重放积分，不创建/修改 solver warmstart，不修改 margin、friction、nativeccd/multiccd、solver、timestep、碰撞几何、质量或驱动。原 2400 control / 12000 native 预算已耗尽，不存在重试额度。

允许 JSON/gzip/NumPy/标准库运算、文件散列、纯测试、官方固定版本源代码阅读、包元数据与已安装二进制的只读散列。诊断模块不得依赖导入 MuJoCo 或 env/plant/runner 来计算结果；绘图为可选纯存档可视化。不得下载构建或替换引擎，也不开展参数扫描。

## 4. 有限工作包

### A. 真实故障夹具与解析几何证据

从已冻结文件抽取一个带来源散列的最小 JSON 夹具：native 3486 contact 5、同样本 contact 4/6、3485/3487 对照候选、对应编译 geom 元数据，以及现有 static forensic 的四姿态摘要。抽取复制保留全部原值，不修正法向或载荷。

对圆柱使用保存的 center、rotation、radius、half-length 计算最低支持点：轴 `a=R[:,2]`，`p_min=c-h*sign(a_z)*a-r*(e_z-a_z*a)/sqrt(1-a_z^2)`。实际本例 `abs(a_z)<1`。记录最低点、其 XY 是否落在原箱 footprint 内、相对箱顶的 gap。与已保存 `mj_geomDistance` 结果只作对照，不重新计算引擎距离。

预期实值：3486 before 的最低点 z = 0.01567653249319506 m，箱顶 z = 0.015 m，gap = 0.0006765324931950617 m。法向 z 约 -0.99999135，必须仍判几何不一致。

### B. 新增纯诊断层；原总门保持关闭

建议新增 `scripts/diagnose_d1_single_step_archive.py`，可选择另一个同样明确的名称，但不得替换冻结评分入口。模块只读冻结存档，先核验 provenance，再计算所有记录的完整性和几何问题清单，避免第一个几何异常导致后续原始一致性审计停止。

输出必须至少含：

- `archive_identity_valid`、`source_identity_valid`、完整计数与连续性、`raw_record_links_valid`；各项注明核验范围。
- `all_candidate_geometry_valid=false` 及原生索引、contact 索引、geom/body 身份、原始 frame/normal/pos/dist/margin/local_force、候选面与残差，原因码例如 `box_normal_outside_support_cone`。
- `positive_load_candidate_geometry_valid=true` 只能作为单独观察项，必须来自所有 11032 条正载荷 box 候选的逐条原规则验证；不能充当所有候选门的替代。
- `original_frozen_score` 按原文件读取，`original_score_overridden=false`、`qualification_granted=false`、`rl_gate_open=false`。
- 若输出任务观测值，必须标 `unqualified_observations`，用原合同的精确窗口与阈值；不得产生新的“已通过 readiness”总分。

原始身份/链路检查应覆盖两场全部 native 行和控制 endpoint；geom/body/name、frame 有限/正交与正行列式、efc/active、局部载荷、geom-order 法向和世界力、每轮载荷与 box 载荷摘要、nonwheel/geometric 标记、qpos/qvel 连续性和控制/endpoint 交叉绑定。不得通过跳过异常 contact 获得 `raw_record_links_valid`；“原样记录异常”可以忠实，“异常几何”仍为失败。

保留 `0.0021` cone limit、原距离容差公式、原有非轮接触门、strict positive normal-load 定义和全时间/晚窗约束。禁止取法向绝对值、翻转法向、按零载荷删 contact、新增力 epsilon、放宽位置容差或使用轮中心越界替代全碰撞几何 clearance。

### C. 固定版本机制审计

只读 MuJoCo 3.12.0 官方源码，保存所读原文或 SHA256/URL/版本。范围限于凸体接触分发、`mjc_Convex` 的多接触扰动、`mjc_penetration` 的 witness→normal 转换以及所需直接调用的 GJK/EPA 支路。结合当前模型固定 flags/正 margin，形成函数调用关系及已证实/推测的边界。

已有官方源代码显示正 margin 会走重复窄相检测，多接触扰动用 1e-3 角度，额外 contact 可以继承主 contact 的 dist。它解释研究入口，不能单凭源码确认本例是哪一次扰动、哪一条迭代或退化分支产生符号错误。没有运行级内部证据时，明确写“内部数值根因未定位”，这仍是合格的有限结论；不得把“源码中可疑”写成已修复引擎 bug。

本轮不构建 instrumented 引擎，不切换引擎版本，不改 collision flags，不复跑已完成四姿态。

### D. 纯回归与独立复审

复用此前 7 项 synthetic 审计证据；新测试优先基于真实夹具。最多新增 12 个有区分力的纯案例：真实异常仍几何失败且原始链路忠实；把异常改为正载荷仍失败；篡改法向/身份/力摘要或加入 NaN 必须被识别；原真实 ±0.001 扰动 top 候选以及真实 front-top 边缘候选仍按原规则处理；截断/缺字段不获完整性资格；输出不得打开 qualification/RL 门。

真实 fixture 只从冻结证据产生；测试中的故意变异仅在内存/新测试夹具内进行，不反写来源。禁止只测试分类 label 而不保留真实 raw frame 和 geom-order 链。

完整两场离线诊断至多两次（一次开发核验、一次冻结后独立核验）；纯 fixture 测试最多两个修复/核验轮次。发现需要引擎实验或范围外改动即可停止实施，记录具体未解问题，不扩展实验。

## 5. 完成条件与交接

新增一个独占输出目录，包含：真实故障夹具及来源、完整离线诊断 JSON、解析支持点计算、固定版本源代码机制说明、纯测试结果、输入/输出/source manifest、独立复审和执行预算表。新的每项预算计数均为零；历史四次静态调用应以“已有证据”列出，不冒充本轮新执行。

完成指：失败能够准确定位，原始忠实性检查可继续覆盖全记录，异常几何仍被拒绝，未资格化观测可透明阅读，所有旧证据与阈值不变；不要求也不允许把本次 box 变成通过。

最终给出一个后续决策：保留阻塞并描述最小引擎复现/修复研究所需新合同；若没有证据支持具体修复，就保留内部根因未定。任何未来物理测试、候选过滤/重定向、引擎替换、重新 readiness 或 RL 都须独立新合同及明确新预算，不由本合同继承授权。

## 6. 冻结输入 SHA256

- `original_contract`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/next_terminal_handoff_plan_01/next_contract.md`
  SHA256 `1ea04dc310486ed7025a73ed1e61a7977fdaafe0545f3de859905835d1b5c238`
- `trial_manifest`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/single_step_readiness_01/manifest.json`
  SHA256 `80b7a829b6be1332bf050d68b2774903457961a5400c5efa8b59da094e366f46`
- `static_forensic`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/static_contact_forensic_01.json`
  SHA256 `0c46a59f8905e1310b7a73d4c6c749440264161ddb27a4d067ac0ce005d30bc2`
- `observations`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/observations_01/observations.json`
  SHA256 `b50121afd6dd1a519a02ab1a4b8adc4d495772da2cca172e671fe2af8e3232ea`
- `contact_inventory`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/contact_failure_inventory_01.json`
  SHA256 `48e1f39fbeb52727f647933cd6d6172049fe86124f4f47479c603f92b25c728b`
- `frozen_scorer`: `/home/lyh/wheel-legged-control-lab/scripts/d1_single_step_scoring.py`
  SHA256 `d9c13912cafaa92a52d53671b8016a85328116bc2cc4e6f267eb5007f33fefed`
- `frozen_integrity`: `/home/lyh/wheel-legged-control-lab/scripts/d1_single_step_integrity.py`
  SHA256 `fa1d1ca12de8bed2051b374720ab8df774a913308ef6d413a46c49411c5bda8e`
- `frozen_geometry`: `/home/lyh/wheel-legged-control-lab/scripts/d1_single_step_geometry.py`
  SHA256 `7c6a1c7450286286f3e732e214b6d0a08247eab2f073dcfae671453529de18ed`
- `static_audit_script`: `/home/lyh/wheel-legged-control-lab/scripts/audit_d1_single_step_contact.py`
  SHA256 `5b39d2f0deda605bbfeaddc5425f169be130b5bebfd94751ae8e12ce76ecbdc7`
- `observation_script`: `/home/lyh/wheel-legged-control-lab/scripts/summarize_d1_single_step_observations.py`
  SHA256 `98ab13e3fa4863882b36a2543474292f427fd5e53d2ad7b5cdb7e5d2e5cfe52e`
