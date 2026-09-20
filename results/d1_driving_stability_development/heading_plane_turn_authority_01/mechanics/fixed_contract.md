# 转向内层反馈恢复：固定零物理资格与唯一候选

本合同先于资格 helper 执行冻结。新增物理、控制器 compute、接触求解和训练均为 **0**。不改任何 frozen/default/controller 文件；只交付离线数值资格代码供 root 审阅执行。

已查明的事实：原 baseline 和独立 turn-damper 的左右四条轨迹，执行 200..249 的原内层公式 `r0 = r_servo + 4*(r_servo-r_body)` 均有 50/50 拍超出 ±0.6。baseline |r0| 约 4.20..4.40，damper 约 3.54..4.40，而实际 r_eff 始终 ±0.6。在这些保存状态处，clip 后 `∂r_eff/∂r_body = 0`；未截断内层本为 −4，连同未饱和原外层则为 −6。该事实证明原 body-rate 反馈通道在 pulse 内被屏蔽，**不证明侧向接触阻力无关，也不保证放开后通过任务**。

唯一待资格机制：仅 `raw forward == 0 and raw yaw != 0` 时令 `r_eff = r0`，然后按原 `(v_servo-r_eff*y_i)/0.087` 和最终 ±30 rad/s 得到 wheel target。原 wheel PI 2.2/3、积分 ±4、antiwindup、12 N·m wheel / 80 N·m leg 额定限矩、原位置和速度外向保护、原腿 PD/support、外层 ±1 均保持。inactive 返回原目标/结果；没有 center 补偿、没有 turn/stop damper、没有 latch、没有新的数值 cap、没有 gain/dt/命令时长选择。

固定资格集合：`flat_plane_02` 与 `plane_turn_damping_01` 的左右 turn，各取 `S[199]、S[200..249]、S[250]`，共 **208** 保存姿态。不添加候选轨迹、不积分、不沿用 shadow PI 到下一拍。damper 状态只作已见状态压力样本；其中原 `base_requested_torque_nm` 用于去掉阻尼，不能把它误当组合候选。

离线方法：

1. 原 G1 raw 协议与 77 冻结输入、四条完整 case manifest、当前源文件均 hash 验证。用保存 qpos/qvel 的非积分几何/物体速度查询复核真实横向轮坐标、body yaw 和原目标。
2. 每个状态同时报告 r0、原 ±0.6、servo/body/raw、完整反馈被遮蔽量及导数；只计算本次唯一 restored-formula target，不构造 0.8/1.0/2.0 等候选 cap 集合。
3. 从该条原记录的 I_before 独立做一次原 PI/antiwindup，再执行原 rated/outward 保护。保存所有 wheel target、请求/实际力矩、积分拒绝/clip、额定限矩占用、实际关节速度与离边界余量。预计首拍原请求会超过 12 N·m；**额定限矩占用是候选必须如实检验的行为，不以“无保护”作为新增门槛**。所有最终实际力矩必须有限并遵守原保护。
4. 原 antiwindup 且 |I_before|≤4<12 时，最终未保护 wheel 请求不超过 ±12 的精确误差区间是 `(-12-I_before)/2.2 <= e <= (12-I_before)/2.2`。同原 ±30 target 相交并由 `(v-r*y)/R` 反解四轮共同 r 区间，**仅作可用力矩范围诊断，不作为新 limiter 写入控制器**。逐端点复算原 PI 证实边界，不从区间选择下一 cap。
5. wheel target 对固定 body 状态的 wheel-spin 速度导数仍为 0，因此 PI 的原误差导数 −1、未饱和瞬时力矩导数 −(2.2+3×.01) 保留。额定饱和时实际力矩局部斜率可能暂时为 0；这与先前直接替换误差在未饱和区永久抵消差动 wheel damping 不同。完整 loaded 离散稳定性未知，不伪造证明。
6. 读取原 native solved contact cache，分别聚合 pulse 的 world-x/world-y 力对 yaw 的贡献、支持载荷与摩擦利用；不得与端点速度混算功。旧 1.0 cap 试验保留为失败证据，但其 plant 是旧 hfield，且仍保留固定 clip；不把它当成本 plane 下恢复原误差反馈已失败的证明。

受控 GO 的用途是验证一个尚未被证伪的反馈恢复机制。先核对数值有限、原保护生效、wheel damping 未被公式取消、没有隐藏第二机制；结合侧向几何和真实接触力风险作有限探针决定。若恢复 r0 只会造成持续轮滑/腿形变、机身响应不改善，则反馈恢复假设失败；不得继续递增 cap/gain 或延迟 raw 评分。

root 如决定执行，本机制只做一次最多 **3200 新控制步 / 16000 native 子步**：左右 turn 800×2，正反 stop noop 800×2，复用 `flat_plane_02` 四条原基线。新 plant/controller/task 身份拒绝旧 checkpoint；零残差。turn state[0..200]、执行[0..199] 与 native[0..999] 逐位相同，obs 只比[0..199]，因为 obs[200] 已预览新的 wheel target。stop 全 801 state/obs、800 执行、4000 native 与原 summary 除 model 逐位保持。raw pulse、raw reference、原全部 gates 不变。

若本唯一 authority/feedback 恢复试验仍未通过，停止 wheel-cap/PI-effort 家族的串行试探，按 native 纵/侧向抵消、轮滑与有限腿工作空间的失效签名转入新的几何/机动模式合同。停车已通过的独立机制不因此失效；可靠转向和停车组合门禁完成前不进入跳跃/台阶或提速。
