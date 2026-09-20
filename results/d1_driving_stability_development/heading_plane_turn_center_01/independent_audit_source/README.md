# 固定中心速度补偿：独立离线审计

`audit_candidate.py` 审查已封存的 `plane_turn_center_01` 四条候选，基线为 `flat_plane_02` 对应四条记录。脚本不运行环境、控制器或物理；`mj_step/mj_step1/mj_step2/mj_forward/mj_inverse/mj_collision` 均被 monkeypatch 为立即失败。只使用保存姿态的 kinematics/comPos/comVel/Jacobian 与物体速度查询，所有新建 `MjData.time` 保持 0。

本目录脚本交付时仅完成 AST/编译检查，**未执行保存态审计**。主代理读完后执行下面命令；`report.json` 存在即拒绝覆盖。失败同样保留独立报告和错误上下文，绝不修改候选/基线记录或重跑物理。

```bash
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps:/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/plane_turn_center_audit_01/audit_candidate.py --candidate /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/plane_turn_center_01 --baseline /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/flat_plane_02 --output /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/plane_turn_center_audit_01/report.json
```

核对范围：

- 原 revision 2 G1 协议 SHA、77 份冻结输入、候选 input hashes、完整候选根 manifest 和八条 case manifest；最后再查输入未变。
- 从 4 × 801 保存 qpos/qvel 独立重建水平 heading、轮中心腿相对速度 `u_i`、实际 body COM 速度/角速度、横向坐标；核对原外层 yaw、原内层 ±0.6、仅 raw forward=0 且 raw yaw≠0 的 `+u_i/0.087`，最后 ±30 clip。
- 独立重算 wheel PI、积分限幅/antiwindup；从存档 unlimited requests 重算全部 16 轴限矩、位置外向、速度外向保护。**没有独立重建腿 PD/support 请求**，报告如实保留此限制。
- 执行 tick 199/200/249/250 的 gate 快照与 callback `k+2` 计数。恰好 200..249 共 50 拍激活；停车始终 inactive。callback 最后一次准备未执行的 terminal decision，因此最终计数 801。
- 原 raw reference 与端点误差来自原 command schedule 和保存状态。原 scorer 只作为纯归约函数调用；保存态重算误差后再次计算门槛，不能用 servo reference 或移窗替代原评分。
- 转向 states[0:201]、obs[0:200]、执行区间 [0:200]、native [0:1000] 逐位配对。obs[200] 已包含候选下一拍 preview，故不要求相同。停车全部 801 状态/obs、800 区间、4000 native 记录及所有原 summary 字段（仅 model 除外）逐位相同；同 case 不豁免 ±0。
- 完整 3200 执行区间/16000 native 返回、5 子步 held torque、时间连续性、实际 wrench、实际接触法向和力/矩求和；execution_states/末态/receipt/partial 账本一致。没有新增物理步或基线重跑。

`passed: true` 只表示审计一致，**不表示原任务通过**。候选 original gates 独立保留在每个 case 中。端点接触速度从同期保存姿态重新计算；native 接触力是真实存档的 solved cache，脚本不重求历史力，不把两者相乘充作同步接触功。摩擦利用率只是存档力比值诊断，没有改成评分门槛，也不是稳定性证明。

交付校验和 `checksums.json` 仅覆盖审计 helper、本说明和静态检查收据；主代理后续生成的运行报告应另行归档校验和。
