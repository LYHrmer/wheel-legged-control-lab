# 原生 CCD primitive adapter 合同 02

2026-09-23；实际 GPT-6-astra ultra 规划/指挥/复审，实际 GPT-6-sol 实施，主代理唯一执行与预算持有人。此为已授权常规开发验证下的新有限阶段，不继承任何历史物理预算，不另造用户批准流程。

## 已完成与被阻塞的边界

`zero_call_contract_closure_01.json` 已确认：12 个真实 fixture 纯测试通过（本地 1 轮），完整两场离线核验 2/2，三个结果文件逐字节相同，receipt 仅 pass number 不同，来源冻结未变；新 MuJoCo/static/control/native 均为 0。本合同不得增加旧 full-scan 第三遍。常规发布 CI 的两个 Python jobs 另列发布验证，不能声称它们全部零引擎调用，也不伪装成新的本地零调用实验或 readiness。

**合同 01 从未激活并被撤回执行建议。** 官方 3.12.0 `user_model.cc:5614` 在编译验证时内部调用 `mj_step`，world-only/no-DOF 不豁免。原 01 正文保留，阻塞证据为 `astra_compiler_hidden_step_review_01.md`；先前 offline review 末尾允许激活 01 的建议由此明确撤销。禁止执行旧 XML/model 方案、用 sleep flags 绕过或将 compiler-private 时间推进漏计。

目标仍是稳定直行 → 可靠越障 → 提高速度。此次只隔离阻碍越障资格的原生接触候选异常，形成可供源级修复的最小诊断。原 box readiness 仍 invalid；零载荷候选不能被删除、翻转或取法向绝对值来过关。

## 确切入口与证据级别

只调用现有固定 MuJoCo 3.12.0 库导出的 **非公共内部接口** `mjc_initCCDObj`、`mjc_ccdSize`、`mjc_ccd`，配合严格复制的本地 perturbation wrapper。不得称其为直接 `mjc_Convex` callback、完整引擎复放或已修复实现。

使用完整官方 `engine_collision_convex.h` 与 `engine_collision_gjk.h`，以及安装包匹配的公开 headers。不得手写结构布局、ctypes 猜测、packing、定义 mjUSESINGLE。`<ccd/vec3.h>` 仅满足官方头中的声明依赖，记录其和 config/compiler include 依赖散列；不能进入 libccd 分支。运行前 sizeof(mjtNum)==8、header/native version 必须一致；所有内部接口符号必须解析到固定库，记录该事实。

`engine_collision_convex.c:726-783` 已逐字段审计：在 g>=0 且 cylinder/box 条件下，initializer 仅读取 `geom_type`、`geom_size`、`geom_xpos`、`geom_xmat` 四数组，复制到 descriptor 并设置原生 center/support pointers。允许零初始化的 typed `mjModel`/`mjData` **局部只读 carrier** 仅提供此四数组，传给该 initializer；这不是经分配/编译/初始化的有效仿真 model/data，绝不可传给其它 model API。descriptor 也整体零初始化；两个 callbacks 只读 descriptor 几何，box support 更新 vertindex。union/size[3] 在此路径不被读取。不保留 carrier 指针。

`mjc_ccd` 不接收 model/data。max_contacts=1 固定使 mesh/multicontact metadata 分支不可达；cylinder–box 不进入 sphere/capsule 专用分支。`mjc_ccdSize(0,0,35)` 计算工作区大小，一次普通对齐 malloc；不调用引擎 arena/stack。零初始化 status 可用于可靠序列化，但不能把未产生的 witness 宣称有效。任何 warning/error 留档并在当前调用返回后终止后续查询。

## 固定几何、算式与来源

输入严格来自固定 fixture 的 3485/3486/3487 before 世界轮姿态和保存的 box 编译几何；用 binary64 C hex literals 无损传递完整 3×3 矩阵，不做 quaternion 往返。新 descriptor IDs 为 wheel0/box1，原身份 wheel59/box1另存，g>=0 primitive 路径不依赖 ID 数值。

轮 size=[0.087,0.02,0]，箱 half-size=[0.18,0.62,0.0075]，box center=[-3.0999999999999996,0,0.0075]；轴对齐 box rotation=I。轮/箱实际 margins .001/0、gap=0，**pair-wide margin .001 必须完整传给两个 descriptor**，native support 内部各加其一半，不能先除二。固定 config：iterations35、tolerance1e-6、max_contacts1、dist_cutoff0、npolygonmax0、nmeshdegmax0。记录为原冻结 builder 未覆盖的 3.12.0 defaults，不声称重新读取了历史 model 内存。

wrapper 精确复制 `convex.c:87-132,822-853,881-961`：初始 query 返回 dist<0 才转换 witness；contact.dist=margin+status.dist、pos=(x1+x2)/2、normal=normalize(x1-x2)、tangent=0。初始 normal 的 frame 产生两个切向轴，顺序 axis0/axis1，各 -0.001,+0.001；双方绕初始 contact.pos 反向旋转，随后只恢复 pos/mat，保留原程序跨查询的 mutable support cache。不能每次重新初始化 descriptor 导致分支改变。

额外点 distinctness 使用 1e-3*min(geom_rbound)。若旧 fixture 未保存 rbound，必须先保存官方固定版本的 primitive rbound 计算源码并逐式采用（cylinder 与 box）；不能靠引擎编译补取。需同时保存原始 query status.dist 与接受后被 primary.dist 覆盖的 contact.dist。未经 distinctness 接受的 query 也完整记录，不消失。

本地 frame/helper 必须逐式对应原源码，原 kernel normal 独立保存；可以调用名单内纯数学公开 mju helpers，并记录各函数显式调用数。math 操作顺序/编译可能导致数值不同，必须在报告中保留限制，不能把本地 wrapper 当成已验证的原二进制 wrapper。

## 独立预算与顺序

主代理在执行前创建独占 `native_ccd_budget_02.json`，先登记尝试，再进入调用并 flush 输出。无重试，异常也扣额；禁止换目录/换进程刷新预算。

| 类别 | 上限 |
|---|---:|
| control / native integration / step / step1 / step2 | 0 |
| model compile/load/parse、engine model/data allocator/reset/setConst | 0 |
| forward/inverse/constraint solve/kinematics/comPos/collision/geomDistance | 0 |
| descriptor initCCDObj | 8（每 case 两个，最多四 case） |
| mjc_ccdSize | 1 |
| native mjc_ccd attempts | 16（前三case各最多5，条件第四case1） |
| explicit pure mju mathematical helpers | 512，总数及各函数记录 |
| native version/versionString | 各1 |
| libc workspace malloc/free | 各1，不是 engine allocator |

原生 CCD 内部纯数学迭代由35迭代参数约束，不冒充以上 explicit-helper 计数；所有出现的 warning 保留。禁止 model/env/plant/controller/rollout/import MuJoCo Python。不能调用 mjc_Convex、mjCOLLISIONFUNC 或编译整个引擎。仅新 harness 的 C 编译/链接、源审阅、文件散列和 JSON分析可以不耗查询预算；子代理均不得执行 harness。

固定顺序：3485 → 3486 → 3487，均 primary 后按原 wrapper 尝试最多四个扰动。若 primary 不恰好返回一个 witness，则按原 wrapper不做附加扰动；若输入不符、非有限有效字段、warnings、ABI错误、越界，立即停止而非完成剩余case。

**条件第四case：** 仅前三case有限且输入未变，邻居3485/3487各三个候选并均通过原 cone，3486三个候选的 primary valid、恰好一个下向几何失败并对应原 contact5，才对3486执行 primary-only 一次。对应关系需保存全部逐字段差异：先 bitwise，再诊断 correspondence（同序点位置/距离差不超过原参考 dist绝对值+pairmargin+1e-7，normal dot>0.99；这不是新物理通过门）。两邻居和3486的全部同序候选均需对应，而非只验证坏点。该第四case不修改任何 model flags，仅跳过本地附加查询；不能宣称证明了完整model关闭multiccd的动态行为。

## 强制保存与验收

每次 query 先写 attempted event，随后保存输入 descriptor 数值和变化、config、return distance、全部有效 status（nx/nsimplex/simplex/witness/dist/separated、GJK/EPA iterations/status）、raw normal/tangent、derived comparison frame、distinct结果、接受索引和覆写前后distance。EPA status0不等同收敛；迭代上限必须据实显示。每case的 final candidates 与保存old pair逐字段比较，保留“逐位相同 / 同类几何异常但数值不同 / 未复现”三级结论。

运行前后保存四数组输入校验、所有文件和库/header散列、实际加载路径、编译命令/compiler/version、stdout/stderr、所有计数和错误。每次尝试持久化到独占新输出，成功/失败均封存，不再次启动。符号名单和源代码调用图经Astra审阅，主代理再决定首次唯一执行。可以只对新输出做纯 JSON/NumPy复核，不能开启旧完整archive第三遍。

原箱体 cone limit .0021、位置公式 abs(dist)+max(inclusionmargin,geommargins)+1e-7、positive normal load定义、所有历史动态 gates 原样保留。本实验没有约束求解和力，不能生成或声称新的接触载荷。`qualification_granted=false`、`original_score_overridden=false`、`rl_gate_open=false` 必须显式。

## 收敛出口

A：重现同一异常且query日志定位到具体 axis/angle/EPA witness。交付可独立复现的固定native核夹具和最窄数值问题；下一动作是有限源级修复/回归，用有效支持几何纠正数值求解，不是过滤零力点。此合同停止，不在16次外追加probe。

B：同类异常但与旧pair不同。保留本地wrapper/编译算序限制，交付差异表；不能声称完全复现。只据已存query调查，禁止调参重跑。

C：未复现、健康对照失配或出现error。封存reproduction-gap并省略第四case；分析已存输入/ABI/算序问题，不切换flags/size/tolerance或重跑机器人。失败也完成这个有限研究的交付，不据此推翻旧异常。

无论出口，旧2400/12000预算仍耗尽。只有新的真实修复经过明确静态回归后，才另定新的越障readiness动态预算；提高速度仍排在可靠越障后。

## 固定锚点

- previous zero-call contract SHA256 `157f0a33b386cea0ef99556a0d693370126ae57479db0460e4b211dd7b27e7d9`
- blocked kernel01 contract SHA256 `0a442f05ba09693d9e97bdc171cee05fef9e1c835225a6bd414209c4c9e374b9`
- fixture `tests/fixtures/d1_single_step_native_normal_anomaly.json` SHA256 `91cb3c9d1137b11c39f071de01ddb9e2f9ba83ca90af8e04e3cc565800852a55`
- installed `/home/lyh/.local/lib/python3.10/site-packages/mujoco/libmujoco.so.3.12.0` SHA256 `bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8`
- official convex header SHA256 `3df5241f900dd9c01565c857191eeb740104b09f75a11af56500b73fbaefddfe`
- official gjk header SHA256 `756622585c9b5d1c5f824eb8e7d69c8c8393addd74733cfddbb01321ff3a219e`
- compiler block review SHA256 `dd3847da45da3ad31c2f0fad45aae074ea6ca869cb71437e1e53c932e5dff9e8`

其余源文件及准确URLs/bytes/SHA见 `sources/source_manifest*.json`，安装headers见 `sources/installed_package_readonly_fingerprint.json`。新harness/generator/header/descriptor JSON、rbound源码和本合同全部在首次调用前进入独占source freeze，不修改旧证据。
