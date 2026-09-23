# EPA 最小源码修复与固定回归合同 01

2026-09-23。实际 GPT-6-astra ultra 指挥/接口约束/验收；实际 GPT-6-sol 实现；主代理持有唯一执行账本。本文冻结下一阶段具体范围，不表示已经执行。用户已授权常规开发、验证和发布；完成准备与独立审阅后主代理可按本文推进，无需把内部合同变成额外用户确认。不得继承已耗尽的任何旧实验预算。

## 目标、当前证据与明确出口

顺序保持稳定直行 → 可靠越障 → 提高速度。本次唯一目标是把已最小复现的 EPA 错误收敛为**一个有机制证据的局部源码修补候选**及固定静态回归结果。不是继续扩大机器人实验，也不是仅再收集一个坏法向。

已闭合：纯存档两遍相同、12纯例一轮；原生 primitive 实验16/16 queries全部返回、前三组三候选逐位复现旧记录、第四primary-only保留正常主候选。失败具体为query7/native3486/axis0_negative，GJK计数6、EPA循环索引4、epa_status0、非上限耗尽；native x1-x2 已产生错误方向。epa_status0只指初始polytope成功，具体EPA退出/选面/affine分支尚未记录。

本阶段只建立两个有数值意义的版本：A＝固定源码+观察插桩；B＝A基础上一个机制驱动的最小修补。A若不能复现，停止数值修补，交付构建等价性缺口。A若明确机制但没有可论证修补，交付准确根因与最小失败夹具，不编造修复。B若固定回归失败，保留失败候选并停止，不再换补丁或调参。本阶段禁止安装/集成候选到真实引擎、任何readiness或RL。

## 输入和不可变文件

读取合同02最终日志中的16条 `ccd_attempt.descriptor_before` 和对应 `ccd_return`。保留binary64原值（hexliteral）、type/size/position/完整matrix/margin、geom身份、vertindex/meshindex、config及query顺序。仅以这些已经保存的两primitive查询输入进行源码核实验，不再重构全机器人、旧四pose、旧两场或wrapper姿态。

全部旧合同、原source audits、native_ccd_run_02、source freeze、fixture、frozen77、scorer/integrity/geometry/controller及原scores字节保持不变。源文件副本、源码patch、trace/回归结果与账本必须在新独占目录保存。旧fullarchive2/2和nativeCCD16/16都没有余额。

固定参数：cylinder/box，g>=0，pair-wide margin双方各.001（support内部各加一半），max_iterations35、tolerance1e-6、max_contacts1、dist_cutoff0、npolygonmax0/nmeshdegmax0。严禁改这些值、删候选、翻转/绝对值法向、按零力过滤或更换engine版本。

## 只构建隔离 GJK/EPA 源码核

使用已保存官方3.12.0 `engine_collision_gjk.c` 的新副本，不构建完整MuJoCo，不解析/编译XML，不构造可积分model/data。`mjc_ccd`、`mjc_ccdSize` 均重命名为清楚的local trace/fix符号，避免与安装库混用；nm/链接表要证明真正调用的是本地版本。不可借链接优化偷偷删除算法分支来简化构建。

允许下载并固定官方3.12.0所需的 `engine_util_blas.h`、`engine_util_errmem.h` 及其最少直接include依赖；保存URL/version/bytes/SHA。完整官方convex/gjk headers及安装公开headers继续固定。不得手写/猜结构ABI、引入fast-math、改变mjtNum精度或使用任意依赖最新版。

整份GJK源码引用但当前类型不执行的 `mjc_pointSupport`、`mjc_lineSupport` 不在安装库导出。允许从固定convex.c:201-203和219-232原样复制这两个小pure helper并重命名，以及固定inline数学依赖；禁止伪造stub、把不可达分支变成未解释行为或复制整个convex实现。记录这两个helper实际调用数应为0。

primitive原生center/support指针仍由已审四数组typed carrier调用 `mjc_initCCDObj` 取得，之后恢复日志中的geom/type/size/pose/margin/vertindex/meshindex数值；不能反序列化旧进程函数地址。carrier仅传该initializer。每query从自己的已保存descriptor输入开始，不依赖上一query的工作区或缓存。

已知其余外部引擎入口是纯数学 `mju_mulMatVec3`、`mju_normalize3` 和 `mju_warning`；sqrt/fabs为libm宏。新增所需入口须在执行前逐一源审，遇model/step/forward/solver依赖即阻止执行，不通过“静态模型”豁免。允许观察性日志回调，不调用其它引擎操作。编译选项固定并保存；插桩不得改变原来的分支、浮点表达式顺序、tolerance或状态更新。

## A：先捕获实际内部机制，16次

首次source-local基线按旧query1..16每个一次。重命名和观察插桩之外不改数学。保存：

1. 原始query输入、config、返回/status、原始simplex/witness、所有attempt/return序号。
2. 每个EPA迭代的k、当前和前一face身份、nmap/候选面索引与dist2、被选面的v/dist2/三个顶点身份；upper/lower更新前后、support点、dot、实际gap。
3. **每一个break/return的唯一原因标签**：前一面回退、gap停止（含负gap）、零下界、重复support、horizon不足、faces容量、attachFace退化、候选map耗尽、迭代上限等；不能事后从epa_status猜原因。
4. 最终被传给epaWitness的面和三组vert/vert1/vert2，triAffineCoord实际使用的投影轴、M_max/C31/C32/C33及**实际用于lincomb的lambda**。不能用事后重算lambda冒充当时值。记录sum(lambda)、最小lambda、重构v与x1-x2残差，保留有符号值。

可对这些保存的数字做标准库Decimal或NumPy高精度/稳定公式的纯离线对照，判断面选择/边界维护和仿射恢复谁先破坏了几何不变量；这种纯对照不调用核，不改变旧数据。

基线验收先报告bitwise比较。允许仅为构建等价性描述预先固定的roundoff界：每个浮点标量差≤256*DBL_EPSILON*max(1,abs(old),abs(new))；所有整数状态/计数、nx/nsimplex及simplex顺序必须相同，所有ret/witness/simplex坐标满足该界，且旧query7的方向/几何异常和其余15个健康分类保持。若只是roundoff equivalent，应明确声明并降低对原二进制内部逐位路径的归因；不能伪称bitwise。超出此界或故障未复现即停，不进入补丁阶段。该界仅绑定source-local复现，不是修改原始物理资格阈值。

A完成后由Astra只读核验trace并冻结一页机制memo：实际首个破坏的不变量、具体源码行、候选修补及其为什么正确。没有这一证据不能凭直觉修。Root仍统一持有下一阶段预算，不得实施者自行启动B。

## B：一个最小修补，24次固定回归

只有A通过并有明确机制后，允许一次局部数值算法修补，范围限被trace指明的EPA face/bounds维护、退化处理或witness affine恢复及其直接数学helper。不得借机重构全部collision、增加迭代数/放宽容差、改margin/flags、增加箱顶特判、按世界z定向法向或删零载荷contact。修补必须处理几何算法原因；新的trace字段不算修补候选。

B在第一次执行前冻结差异、源hash、输入和验收标准。按固定顺序：原16 queries，再8个预定rigid变换。8例来源为旧query [2,6,7,12]，每个应用以下两变换（按source序，再T0/T1顺序）：

- T0：同时将两primitive位置减去该query原box center；旋转不变。
- T1：Rz(+90°)的精确有符号坐标置换同时作用于两primitive位置和rotation矩阵，然后位置加[1,-2,0.5]；尺寸/margin/config不变。不做quaternion往返。

所有24输入在任何B调用前保存，不从运行结果自适应增选。方向/位置回到对应box局部坐标检查支持几何，不能把旋转后的箱仍当世界轴对齐。

验收必须全部满足：每query有限、nx1、负penetration返回，不能用丢接触来消除故障；错误query7恢复正确支持几何；16原queries按原wrapper的distinctness和主距离覆盖规则纯重组后，所有候选通过原cone .0021及原位置容差，健康场景不出现新几何失败。每个witness相对对应**半margin膨胀primitive**的解析表面signed distance绝对值≤现有ccd_tolerance1e-6；此标准允许已知健康axis1_positive约2.2e-7内插误差，并拒绝原坏点4.25e-4误差。

记录所有健康输出的逐位和数值差异；健康返回距离和witness每坐标变化≤1e-6。8个刚体变换在逆变换后相对其B原query距离/位置每坐标差≤1e-6、方向差范数≤.0021，且各自局部支持几何有效。实际bounds/终止原因必须与机制memo一致；不能仅凭epa_status0宣告收敛。任何非有限、warning、意外分支、无接触、新失败或超界即封存失败，不发起第二个修补。

以上是局部源码候选的静态回归验收，不替换原ready合同，也不授予机器人越障资格。

## 唯一新预算与禁止项

Root创建独占新预算并在每次调用前持久化attempt，错误也扣额；不得换进程/目录重试。最多两个算法可执行版本/最多两个执行进程：A一次、条件B一次。语法/链接修复只在调用前进行，不能借重编译重置已耗额度。

| 调用/工作类别 | 上限 |
|---|---:|
| control/native integration/step/step1/step2/forward/inverse/solver | 0 |
| model compile/load/parse/engine model-data allocation/reset/setConst | 0 |
| kinematics/comPos/collision/geomDistance/installed mjc_ccd | 0 |
| A source-local CCD attempts | 16 |
| B source-local CCD attempts | 24，A验收和机制memo后才开放 |
| source-local CCD总attempts | 40，无第二候选/重试 |
| native initCCDObj | 80，每个实际query至多两个 |
| renamed local ccdSize | 每binary一次，最多2 |
| native version/versionString | 各最多2，仅来源确认 |
| 普通对齐workspace malloc/free | 各最多2，无engine arena |
| 完整旧两场诊断/机器人重放/训练 | 0 |

原生primitive support和纯数学调用属于每个有界CCD算法的内部执行足迹，须列清入口并记录；不得宣称“所有MuJoCo调用为零”。这里零的是仿真初始化/积分/求解，40是隔离源核调用上限。静态构建和纯日志/散列/解析几何计算不产生CCD额度。

禁止覆盖安装库、修改默认环境、替换wheel、动态链接未固定engine、调整机器人模型/控制器/速度、全场景试验或RL。原frozen77/完整native证据/评分保持原样。

## 产物、停止和后续

保存源码来源清单与原样copy、两个小helper来源、观察插桩diff、实际编译命令/工具链/链接符号、A输入与完整trace、机制memo、唯一patch diff（若有）、B固定输入/逐query结果、解析几何与刚体不变性对照、全部attempt/return预算、独立结果审阅。严禁只留summary不留原数值。

成功即交付可复现的局部数值修补候选与固定静态回归通过；下一工作才是受控集成到独立引擎构建并验证接触生成一致性，再制定新的越障readiness动态预算。失败即交付最早不变量破坏/构建等价性缺口/失败候选，停止40以内实验，不追加参数扫描。两种结果都明确下一最小工程动作，不重新跑旧2400/12000批次寻找另一个异常。

qualification_granted=false、original_score_overridden=false、rl_gate_open=false始终显式。可靠越障尚未完成，提高速度和RL仍不启动。

## 固定锚点

- 当前日志 native_ccd_run_02/ccd_events.ndjson SHA256 `c69ee76a732f15a23390522782de7e35bb29a0fb7439eda154c5cd58a9197597`。
- executed harness SHA256 `27bca448180d267c2954f127279c70939f85420d5d1b5da4fe991d63da8cf888`。
- 原生阶段contract02 SHA256 `0869d5836add5dc12a295d5ac4618bff984244c9703516202f9485a29fdf4768`。
- 官方GJK源 SHA256 `0f3ff5b77893132dcc8412d4e96d80f9d625e99cb3009bb7a4505eba16320855`。
- 官方convex源 SHA256 `c616a3a43195cc2bb7ad52546c0a3443a1b514cf9dbcd79363247b1803a993d8`。
- gjk header SHA256 `756622585c9b5d1c5f824eb8e7d69c8c8393addd74733cfddbb01321ff3a219e`；convex header `3df5241f900dd9c01565c857191eeb740104b09f75a11af56500b73fbaefddfe`。
- 安装MuJoCo3.12.0库 SHA256 `bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8`，不得替换。
- 原真实fixture SHA256 `91cb3c9d1137b11c39f071de01ddb9e2f9ba83ca90af8e04e3cc565800852a55`。
- 根代理reconciliation SHA256 `e7a4f783528e930b1005e111a0cca228193771b4dd8b10e622ded2d61359d48a`。
- Astra独立结果检查 SHA256 `f2ba354bf9b189c540a782cc8cd75582e5165986607180ce49c776a598861009`；结果审阅 `414071a4961a386c06d7df4408845ebe6f2fd7d9c19df6a7598a185de3891f67`。

新header依赖/trace/patch/输入须先取得各自SHA并独立审阅，再由root激活账本；任何实际输入/源变更必须显式重新冻结，不能静默沿用旧GO。
