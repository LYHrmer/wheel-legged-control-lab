# 主代理执行登记：固定轮中心转向补偿

用户已授权常规开发、物理验证及GitHub上传。采用 gpt-6-astra / ultra 的 `plane_turn_diagnosis_01/next_contract.md`，SHA256 `f39dcfb0c51e72a61ca34a14b9d81ff5b1e60f4112cea6cc30b3111f056fce00`，固定两转向＋两停车隔离候选共3200控制拍/16000native子步。基线仅复用 `flat_plane_02`，不得改用停车阻尼通过轨迹或重跑基线。

主代理独立复算转向诊断报告逐字节一致（`25cfa109082efe578e3ff7ab8c3a9d10e3f667e9412f43b2f9656de558ad1450`）。实际 Opus provider回执为 claude-opus-5（`opus_turn_center_02`），核心原件及主代理集成差异保留。第一次请求在本地组装阶段失败，没有调用provider，记于 `opus_turn_center_01/creation_failure.json`。单拍前检第一次因本地NumPy tuple索引失败，未积分、未开始候选compute，原件和失败说明也保留。

集成去掉未用callback helper，raw回调由薄env包装原调用，原对象返回且计数。时间容差沿原state provider为绝对1e-10，仅校验时间，不用作控制gate。保留纯nominal preview、原compute一次及PI/保护，+u/.087系数1，水平heading轴，原effective yaw cap .6、最终轮速clip±30，原raw转向窗口与评分不变。另有Ruff导出列表排序，不改变控制等式。

26项新纯测试通过（全部禁止mj_step/mj_step1/mj_step2）。100个原保存态上的独立单拍目标/PI核对通过，最大目标差4.45e-16 rad/s，轮请求差1.78e-15 N m；每样本重置到原PI记忆，不构成轨迹或性能预测。独立Astra代码审查GO。

执行各case一次，任务门槛失败可继续固定列表，数值/来源/记录/配对失败立即停止余下批次，无reset重试/补时。两转向严格比较执行0..199与状态0..200；obs只比较0..199，因为obs200已preview新目标。两停车全800拍/801状态及完整原summary（除model标签）与plane-zero逐位/逐值相等，保留其原三失败项。绝不把noop停车误称阻尼通过结果。

分别报告plant、执行证据、配对、两turn原gates、两noop stop失败保持。不得将+u运动学校正称作阻尼、被动性或真实robot验证；即使通过也需另行检验与停车阻尼组合。当前先前独立实验累计20800控制拍/104000子步；四case完整则累计24000/120000。冻结77、旧实验、三个65k训练和完整24G1保持不动。完整驾驶—越障—提速目标仍未完成。
