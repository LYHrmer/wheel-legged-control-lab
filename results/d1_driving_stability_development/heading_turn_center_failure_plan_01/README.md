# Turn-center 失败诊断与零物理资格结论

状态：保存记录诊断已完成；直接 body-rate 误差替换的代数资格核验已完成；该替换 **NO-GO 进入控制实现或新物理实验**。这不是完整项目终止，也不推断任何尚未试验的方案必然不稳定。

本目录只新增离线证据。没有运行控制器、积分器、训练或新的接触力求解，没有改动旧轨迹、冻结输入、控制参数或评分。根代理已独立复算两份报告，输出逐字节一致。

## 实际改变与剩余失败

原始 plane-zero 与 center 候选均使用 raw G1 评分。左右转向峰值误差分别从约 0.234134/0.234213 rad 降至 0.179587/0.179692 rad，改善约 23.3%，仍未通过 5° 门槛。峰值发生在 endpoint 250；此时关断后的控制尚未影响物理状态，不能把这次门槛失败归因于关断冲击。两场 no-op 停车完整保持原 plane-zero 轨迹和原三项失败，不构成组合控制通过。

下表为左转完整 pulse 的同步 endpoint 201..250 均值；右转符号镜像且结论相同。角速度分解统一使用同一组实际接触点及法向载荷权重，不把轮中心与接触点拟合混在一起。

| 量 | plane-zero | center 候选 |
|---|---:|---:|
| 实际机身 yaw rate，rad/s | 0.132858 | 0.242393 |
| 接触处轮自转对应 yaw，rad/s | 0.453622 | 0.812196 |
| 完整腿运动造成的差值，rad/s | 0.312911 | 0.553583 |
| 其中同权重轮中心项，rad/s | 0.245896 | 0.433707 |
| 其中轮架角运动余项，rad/s | 0.067015 | 0.119876 |
| 滑移差值，rad/s | 0.007841 | 0.016219 |
| 轮中心相对速度 RMS，m/s | 0.057185 | 0.122930 |

轮自转 yaw 增量中，约 30.5% 对应机身 yaw 增量、67.1% 对应腿运动差值增量、2.3% 对应滑移差值增量。这是运动学恒等式的增量账目，不是因果比例或能量分配。前 100 ms 腿运动差值增量占比约 95.3%；中段机身响应随后增大，不能简化为“只有腿在动”。

候选在决策状态 pulse 200..249 的左转平均机身误差为 0.361838 rad/s；投影后的实际 wheel PI 误差已可见 0.221158 rad/s，二者差为 0.140680 rad/s。补齐 full-contact 运动学项有明确的几何理由，但它只解释部分缺失反馈，不能仅凭恒等式断言再次补偿即可过关。实际 wheel target 和 wheel torque 未触发原限制；本次也没有依据调 cap、PI 或系数。

native contact cache 的左转 yaw 力矩显示阶段变化：前 100 ms 净力矩从 3.114 增至 8.132 Nm，中段为 1.284→0.708 Nm，后半段为 −0.243→−0.941 Nm。整个 pulse 中纵向接触力力矩为 6.386→11.182 Nm，而横向接触力力矩为 −5.500→−9.813 Nm，抵消很强。因此纵向滚动补偿不是完整转向动力学逆解。载荷总量与汇总摩擦利用率不能排除个别低载荷接触达到摩擦边界。

## 可复算证据和相位

`helper.py` 在四条保存轨迹的 endpoint 200..800 共 2,404 个状态上，仅调用运动学 API：`mj_kinematics`、`mj_comPos`、`mj_comVel`、`mj_jacBody`、`mj_objectVelocity`、`mj_jac`。代码禁止 step、forward、inverse、collision 及各求解阶段；scratch time 保持 0，qpos/qvel/qacc_warmstart 不变。输入 SHA、冻结 77 项检查、实际 API 计数和所有分段结果在 `report.json`。

闭合最大误差：旧 body yaw rate 0；接触点速度分量 8.69e−17 m/s；yaw 拟合恒等式 5.97e−16 rad/s；轮中心日志 5.56e−17 m/s；native 力矩合计 7.11e−15 Nm。同步 endpoint Jacobian 只与该 endpoint 的位置、速度及归档载荷一起解释。native 力矩只用 native 自己的 contact/reference/world wrench；未把 native 力与 endpoint 速度相乘或据此声称机械功率。

`qualify_body_rate.py` 不导入 MuJoCo 或控制器，只在上述四条轨迹的 200 个 pre-step pulse 状态上逐点代数核验。原 PI 记忆来自每个保存状态；单拍影子请求不反馈至后续状态，不能视为新闭环轨迹。详见 `body_rate_qualification.json` 与 `next_contract.md`。

复算时输出必须是新的文件；以下命令中的 `/tmp/...` 如已存在，请换新路径。

```bash
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps:/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/turn_center_failure_plan_01/helper.py --repo /home/lyh/wheel-legged-control-lab --work /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920 --output /tmp/turn_center_failure_recheck_NEW.json
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/turn_center_failure_plan_01/qualify_body_rate.py --work /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920 --diagnosis /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/turn_center_failure_plan_01/report.json --output /tmp/body_rate_qualification_recheck_NEW.json
```

## 后续阶段边界

下一项工作只做保留差动轮速阻尼的机械与采样资格审查，由另一独立规划代理承接。当前目录不授权新控制律或物理预算。通过停车的独立 damper 不能与本次失败的 turn-center 自动合并。

停转组合仍需独立合同与 raw 评分；人工驾驶需确认同一已合格 backend 的输入、急停和状态恢复；跳跃需分别验证四轮净空、腾空及落稳；障碍需先审 native 接触几何，flat-plane 修复不能一概替换未来 heightfield 障碍；提速需在低速直行、停车、转向及恢复合格后分档验证；物理自救单独立项，仿真 reset 不等于自救。以上均未因本次诊断而完成。
