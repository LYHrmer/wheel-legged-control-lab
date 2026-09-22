# 最终执行前增量审查 02

实际 gpt-6-astra ultra，2026-09-22。仅只读最新源码与SHA；没有import项目、模型构造、测试、MuJoCo或物理执行。root报告的35项纯测试及0physics为root证据，本代理不重新声称实测。本文exclusive-create。

本次快照：

- integrity.py SHA256 `754adcc6e98e58e79052c463ed73050f11b39cee2e639f9fbae9b6c671556a88`
- scoring.py SHA256 `d9c13912cafaa92a52d53671b8016a85328116bc2cc4e6f267eb5007f33fefed`
- probe_d1_single_step.py SHA256 `b65d7df633a4b0a935ce07048576ec6c0d1f7a185fdd08fd1f6a1f37c06c38bc`
- records.py SHA256 `efdd47a55d10f87afeb05654ae060efa23810200e6df89edb78b07fc7cadc36b`

## 一个必须先修的真实执行阻断

**新增body_vx一致性公式用了错误物理参考点。**

`d1_single_step_integrity.py:33–37` 将 `forward_axis @ qvel[:3]` 与endpoint.body_vx_mps比较。前者是freejoint/visible base origin的body x速度；现有truth和controller接口使用的是 **base刚体惯性COM的速度**。

直接源码证据：

- `src/wheel_legged_control/d1/model.py:414` 的base_velocity明确返回base inertial-COM linear velocity，调用mj_objectVelocity(BODY)后转入visible base frame。
- 同文件`:439` 的base_origin_velocity专门减去 `cross(angular_world, rotation @ model.body_ipos[base_body_id])` 才得到visible origin速度，说明两者不能混同。
- `state_estimation.py:304` 使用plant.base_velocity，runner`:62` 又直接记录truth.base_linear_velocity_body[0]。

因此，只要compiled base body_ipos非零且发生相关角运动，原来正确的readiness记录会被新检查误判invalid，最坏会在已花掉plane-only6000 native后阻断box。这不是把1e-12放宽能解决的问题。

最小修复选择：

1. 几何manifest另存compiled base body_ipos和base identity；用freejoint的旋转/平移速度按原语义重建base inertial-COM速度。对于已确认的MuJoCo freejoint convention：`v_com_body = R.T @ v_origin_world + cross(omega_body, body_ipos)`，比较其x与body_vx。务必通过root零积分合成qvel（包含纯角运动）验证角速度坐标约定以及compiled非零offset。
2. 或让现有ScratchKinematics返回base body COM Jacobian @ qvel对应速度并连同state input hash归档，然后body_vx与同一state的该结果比较。它已有mj_jacBodyCom与scratch kinematics，不需要添加任何物理步，也不修改controller。不要仅去掉数据关联，也不要改endpoint字段为origin速度；合同保持原body vx定义。

root已在工具消息中收到此阻断。需要增加的纯/零积分反例是：visible origin平移0、base绕pitch轴有角速度、body_ipos有z偏置；COM body x必非零而qvel[:3]给0。无需真实轨迹即可暴露这个bug。

## 本轮已实质修复的部分

- `compiled_geometry_manifest`存在、导入路径与runner调用签名一致；包含用于contact和bounds的geom/body/terrain/wheel字段。没有发现新helper缺名或参数名错配。
- scorer导入并调用`validate_record_links`；严格N=5*T在判record_valid前执行。普通提前terminated且完整prefix仍可valid、task=False；异常partial当前保持保守invalid并另有terminal存档，不可能伪装完整通过。
- contact local force/normal-load、efc/active、geom/body identity、geom顺序normal、world force及四轮/box/nonwheel/geometric-box汇总均新增重算校验。原先“改汇总就通过”的主要洞已堵。
- bounds要求manifest中的完整collision identity集合与static geom/body/type/wheel/margin绑定，原先“删全部非轮geom”的洞已堵。
- endpoint位置/rpy与相应native返回关联正确；whole-robot COMvz与保存COM vector第2项一致。仅上述base COM/reference-point公式需修正。
- 每个native ctrl与该control的requested_torque_nm精确相等检查，在当前冻结名义actuator配置下成立：default delay0/tau0/gain1，`actuator_channel.py:209` instant分支直接取delayed，不存在渐近lag舍入。不应为此改原actuator。
- runner的transition属性已更正为`requested_torque_nm`，与D1ControlTransition定义一致。
- receipt与terminal_integrator现于env.close/最终arrays归档前写出；比前版显著降低cleanup掩盖已消费物理的风险。
- load_case已检查source与归档manifest哈希，停止无条件True。
- 原严格raw0.2、端点1101..1200、native5500..5999、prefix post-check及两层native预算未被这些修复改变。

## 不作为本轮新增阻断的保留事项

构造仍在case try之前、I/O连续失败不能保证每个文件均落盘、部分附加字段还未全面schema化、offline manifest不是数字签名。这些不构成当前真实且源已冻结路径上的新必现callable故障；outer batch ledger仍能记0physics构造失败，terminal receipt目前提前写出，评分遇missing/error保守拒绝。没有必要在唯一readiness前为此扩建框架或引入新物理。

COM/body-origin语义修复后，把实际新源码和新增测试纳入root最终preflight manifest；使用原已授权固定两场与不变门槛。除上述速度参考点问题，本次有限增量审查未发现另外一个确定会在当前入口触发的名称/接口执行阻断。该结论是静态审查结论，不替代root最终纯测试、零积分资格与来源核验。
