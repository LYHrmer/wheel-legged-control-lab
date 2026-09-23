# 最小圆柱—箱体静态 kernel 合同 01

规划/指挥：实际 GPT-6-astra ultra；实施建议：实际 GPT-6-sol；执行预算由主代理统一记录。
日期：2026-09-23。状态：已冻结工作范围，尚未执行。前一零调用合同完成并经独立复审后，主代理可记录阶段切换并在用户已授权的常规开发/验证范围内直接推进；本合同不是额外用户批准请求。

## 目标与解除阻点的方式

用户顺序固定为稳定直行 → 可靠越障 → 提高速度。当前越障验证的具体阻点是原生凸碰撞候选出现与真实箱顶不一致的下向法向；原 readiness 结果不得追认为通过。

本合同交付一个可独立运行、绑定现有 MuJoCo 3.12.0 二进制、仅包含圆柱与箱体的最小窄相复现程序。用少量新静态 kernel 调用，把失败从完整机器人场景收敛到两 primitive 的原生函数，并区分主接触和 multiccd 附加候选。得到的独立复现程序可直接用于定点引擎修复/上游回归；不再要求重复旧四姿态全模型重建来证明已知异常。

这一步不提升速度、不调整控制器、不训练、不修改碰撞判据，也不以诊断对照代替 readiness。数值内部根因不能在本合同内确定时，以严格最小复现与明确函数边界完成；不可宣称引擎已经修复。

## 激活前提

1. 前一合同的 12 项纯案例通过，完整两场离线核验最多两遍且均已归档；原始忠实性与几何资格明确分开，bad contact 仍被原规则拒绝；Astra 完成独立复审。
2. 所有旧证据及 frozen77、旧评分、原物理源文件散列保持不变。新 fixture、source audit 和本文锚点匹配。
3. 主代理建立独立 `kernel_budget_01.json`，将本合同与已耗尽的 2400 control / 12000 native 历史批次严格区分。本合同不继承任何动态余量。
4. 实施程序与纯输入生成器静态复审完成后才启动引擎。失败调用也消耗预算；不得换目录重置计数。

## 固定输入与二进制

使用已封存 fixture 的 static forensic 中 native 3485/3486/3487 **before** 三个世界几何姿态。只取轮圆柱中心、完整 3×3 rotation、原始尺寸；箱体位置/尺寸取对应编译 manifest，rotation 为原 axis-aligned identity。必须把 binary64 原值用无损方式传给 native harness，例如 Python 标准库生成 C hexadecimal floating literals；不得 matrix→quaternion→matrix 往返后把输入当成原值。

轮半径 0.087 m、半长 0.02 m；箱半尺寸 [0.18,0.62,0.0075] m。box center [-3.0999999999999996,0,0.0075] m。轮/箱 geom margins 分别 0.001/0 m，pair generator margin 参数 0.001 m，gap=0。主 case 保留 nativeccd/multiccd 默认启用、ccd_iterations=35、ccd_tolerance=1e-6；这些值来自固定 3.12.0 默认源码及原 builder 未覆盖对应字段，应在新 model 上明确核验，不能声称本轮重新读取了历史 model 的内存。

只链接现有安装库 `/home/lyh/.local/lib/python3.10/site-packages/mujoco/libmujoco.so.3.12.0`，SHA256 `bd3f702ace8a31e1046f746880387858a981d11b01772a55ebef48ffd55ea5b8`；使用同目录 include headers。元数据版本为 3.12.0，头版本宏 3012000，首次 harness 运行须读 native version 并一致。记录实际加载库绝对路径和 SHA256，防止无意链接系统其它库。不得安装/升级/替换包，不构建改版完整引擎。

公开 C 接口已只读确认：`mujoco.h:65` 声明 `mjCOLLISIONFUNC`，`mjdata.h:446` 声明 callback 签名。应通过有类型的 C/C++ harness 调用 `[mjGEOM_CYLINDER][mjGEOM_BOX]`，不使用猜测 ABI 的 ctypes 指针拼接，不改写该函数表。新 geom IDs 可为 0/1，但输出必须明确映射到旧 wheel59/box1身份，不能要求两模型 ID 相同。

## 允许的静态准备与调用预算

主代理只建立一个新的 two-primitive model，无其它机器人、关节、actuator、sensor、plugin 或 freejoint。允许一次模型编译/加载、一次 MjData 分配，随后直接写入独立 data 的两个 geom 世界 pos/mat 缓存供 pair kernel 使用；这是窄相隔离输入，不是可积分的机器人状态。尺寸、rbound、类型、margin 和 options 另行记录并校验。禁止运行普通 broadphase/整模型 collision 来代替该函数入口。

模型编译可能包含内部常量/几何准备；这些属于明确允许的静态编译工作，不得对外称本阶段“MuJoCo 总调用为零”。程序不包含任何 actuator lengthrange 求解/自动 settle 入口。不得有积分调用；若编译/初始化路径出现隐式积分、运行时意外入口或不能解释的额外求解，立即终止并保留计数/错误。

| 调用类别 | 最大次数 | 说明 |
| --- | ---: | --- |
| control intervals / mj_step / mj_step1 / mj_step2 | 0 | 不产生任何仿真时间推进 |
| 显式 mj_forward / mj_inverse / constraint solve | 0 | 不重算动态载荷 |
| 显式 mj_kinematics / mj_comPos / mj_collision / mj_geomDistance | 0 | 直接使用已经保存的世界几何输入 |
| two-primitive model compile/load | 1 | 仅本新隔离 model |
| mj_makeData | 1 | 无 reset/反复构造 |
| 主配置 pair kernel | 3 | 3485、3486、3487 before，每个恰好一次 |
| multiccd-disabled 对照 pair kernel | 1 | 只允许异常 3486 before，前置复现条件见下 |
| pair kernel attempts 合计 | 4 | 包含失败/异常，不能重试 |
| 内部 penetration evaluations 理论上限 | 16 | 默认各至多 1+4，对照至多1；不冒充 native integration |
| 显式 mju_makeFrame | 16 | 若为匹配 driver frame所需，最多每返回候选一次 |
| native version/versionString | 各1 | 仅来源确认 |
| 资源释放 | model/data各1 | 记录正常或异常清理结果 |

禁止 controller/env/plant/reset/rollout，禁止物理训练或动态反事实。准备输入、编译 harness 本身、散列和 JSON 比较不调用引擎，不消耗 pair budget。所有引擎 pair 调用均由主代理串行进行；子代理只写实现/检查。

## 四次调用的固定顺序与早停

先完成全部无引擎的文件/输入/源码检查。harness 启动后核验库版本、选项与 primitive 编译元数据。

1. 原配置：3485 before，记录全部返回候选。
2. 原配置：3486 before，记录全部返回候选。
3. 原配置：3487 before，记录全部返回候选。
4. **条件调用**：仅当前三次输入保持精确、输出有限，3486 的异常候选已在该新 kernel 中复现且良好对照没有新的不一致，才对 3486 关闭 multiccd，执行一次 primary-only 诊断；nativeccd 保持启用，其它字段完全相同。此 flag 变化只在本新 model 的对照中出现，调用后恢复并记录。若复现条件不满足，第四次不执行。

原默认输出必须和旧对应 wheel/box pair 的 position/dist/frame 对照。核验应先比较原始 binary64值；若微小差异只来自 driver 对 normal 的二次归一化，保留原 kernel normal，并另生成与 driver 相同的 frame 再比较，不能用宽容差掩盖未知差异。严谨区分“完全相同”“物理同类异常但数值不同”“未复现”。任何新的比较容差只能用于诊断差异表，不能更改冻结几何资格的 0.0021 与原位置公式。

内部 extra 的扰动 axis/angle 尚不可见；primary-only 对照用于核验首个候选是否稳定并隔离 extra 生成。不能仅因第四次去掉异常就称已修复，也不能把此 flag 状态移入真实机器人运行。

## 必须保存的产物

新独占输出目录保存：最小 C/C++ harness、无损输入生成器与生成的三姿态 fixture/header、两 primitive XML/模型描述、编译命令与编译器版本、加载库/头/输入/源码 SHA256、完整 stdout/stderr、逐调用 attempted/returned/error、模型/data静态准备调用计数、返回原始所有候选（顺序/pos/dist/normal/tangent）、完整 frame派生过程、support-cone与解析圆柱检查、与旧pair逐字段差异、前后输入与options校验、预算和结论 JSON。

不读取或发布不相关环境变量/凭据。不向上游发消息/提交 issue；若输出上游可复现材料，仅保存在本地供审阅。`qualification_granted=false`、`original_score_overridden=false`、`rl_gate_open=false` 必须保持显式。

## 成功与失败出口；不无限追加审计

**成功 A：默认 reduced kernel 原样复现坏 extra，primary-only 保留正常主接触。** 获得稳定、与控制器/整机无关的最小缺陷夹具，定位到 extra perturbed narrowphase。下一工程动作是用此夹具做源级数值修复/回归，目标是正确支持几何而非删掉零力候选。本合同停止，不能继续加碰撞调用；主代理可据结果另定一个有限源码修复预算，不必把常规开发授权重复问用户。

**成功 B：同类坏法向复现但与旧pair数值不完全一致。** 保存不同点与已证实的共同几何违例，不能伪称 bitwise reproduction。先检查已保存输入/编译元数据/工作区路径能解释的差异；不再调用引擎。本轮仍交付可运行缺陷夹具，但 source-level归因精度需如实降低。

**失败/复现缺口：主配置没有异常或对照本身不一致。** 立即以 reproduction-gap 完成报告，第四次不执行，不改flags、margin、tolerance、poses重试。指出缺失的 full-model context 或内部 workspace/二进制条件，不据此撤销旧异常。不要重新跑两场机器人试验寻找另一个异常。

**执行错误/超限：** 封存错误和实际次数，退出，不通过换进程/改输出路径补跑。

无论哪种出口，旧 box readiness 保持未资格化；这四次静态 kernel 调用不能证明改变接触生成后的整条轨迹仍稳定。只有未来实际数值修复有充分静态证据后，才讨论新的、明确预算的越障 readiness；提高速度排在可靠越障之后。

## 固定输入锚点

- `previous_zero_call_contract`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260922/next_terminal_contact_plan_01/next_contract.md`
  SHA256 `157f0a33b386cea0ef99556a0d693370126ae57479db0460e4b211dd7b27e7d9`
- `source_audit`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923/upstream_source_audit_01.md`
  SHA256 `ebeb3ddaae28c8a06ba61fbec03bdb6dca719c4b4aaf30e4f793a14ab83ceca6`
- `source_manifest`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923/sources/source_manifest.json`
  SHA256 `77a93d4a986307e22c56f69c82f7b569a065c24a707ab82c1224f8273aee8d0a`
- `source_manifest_addendum`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923/sources/source_manifest_addendum.json`
  SHA256 `e6e7fc0b6fbb137bb5b357099fd97f0461c32d6e174121aecc819316310ffa42`
- `installed_fingerprint`: `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260923/sources/installed_package_readonly_fingerprint.json`
  SHA256 `cff5e9088542920911982629ebb3c198d7b06f0586b099b35a9f882a32adcdb2`
- `fixture`: `/home/lyh/wheel-legged-control-lab/tests/fixtures/d1_single_step_native_normal_anomaly.json`
  SHA256 `91cb3c9d1137b11c39f071de01ddb9e2f9ba83ca90af8e04e3cc565800852a55`
