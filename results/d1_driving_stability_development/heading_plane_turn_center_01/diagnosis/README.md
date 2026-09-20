# Plane 转向离线诊断与一个运动学校正候选

2026-09-20，gpt-6-astra / ultra。推荐只测试一个固定机制：**原raw纯转向脉冲期间，对原轮速目标加入轮中心相对纵向速度÷轮半径，系数1**。不改yaw cap或增益，不叠加停车damper。该近似有明确证据和推导，但尚无新轨迹，不能保证原5° heading峰值门槛通过。

只分析`flat_plane_02`左右转向封存记录；新物理步0、controller.compute调用0、训练0。输入complete manifest及77冻结源hash均匹配，结束时再次核对未变化。新建model只用于保存姿态的几何/速度/Jacobian，不构造env、不给历史姿态求解接触力；`mj_step/mj_step1/mj_step2`被封锁，所有审计data.time=0、qpos/qvel不变。MuJoCo3.12.0。

## 失败发生在原脉冲末端

两场均完整800步，唯一失败是heading peak；晚段heading、平移速度、平面位移等其余原turn门槛均通过。

| 指标 | 左转 | 右转 |
|---|---:|---:|
| peak endpoint tick | 250 | 250 |
| 该端点raw参考rad | +.300000 | −.300000 |
| 实际heading rad | +.065866 | −.065787 |
| heading error rad | −.234134 | +.234213 |
| 最大实际轮力矩Nm | 3.34657 | 3.34220 |
| torque protection轴-拍数 | 0 | 0 |

执行tick200..249的raw yaw始终±.6；outer servo均值约±.9658，后25拍已在±1外环限幅；内层effective body-yaw request在全部50拍为±.6。不能仅因“clip占用”把13.4°误差直接归因于限幅：轮速差与机身yaw之间存在更大的机械运动差值，且wheel PI也没有完美跟上其目标。

## 相位与分解

执行区间t使用状态S[t]和当拍目标；trace[t]及diagnostics[t].endpoint_contacts对应S[t+1]。pulse均值使用**端点201..250**，不是把执行前wheel或leg速度配到执行后body yaw。state重建的body-z角速度与同端点trace至1e−12一致；当前heading水平轴为`ex=[cosψ,sinψ,0]`，lateral轴为`ey=[−sinψ,cosψ,0]`，与原nominal wheel几何一致。额外验证重建的body-x轮中心腿速度与下一行before日志一致，但候选u使用horizontal heading轴，不混用两者。

对同端点每个有载接触，以保存normal force归一为权重，令`yaw_fit(vx) = −cov(y,vx)/var(y)`。完整Jacobian点速度满足base+leg+wheel=slip，因此：

```text
contact wheel-spin yaw − free-base yaw fit
    = contact leg yaw fit − slip yaw fit
```

所有点速度逐项与保存的完整Jacobian值一致；每帧四轮都有正载荷，没有缺轮或退化lateral fit，也不使用底点代理。轮中心等权r*qdot拟合和载荷加权接触自转拟合分别列出；接触位置/轮轴方向造成差别，不能将二者当同一数值。

| pulse端点201..250均值rad/s | 左 | 右 |
|---|---:|---:|
| actual body yaw | +.132858 | −.132679 |
| 四轮中心r*qdot yaw-fit | +.469437 | −.469025 |
| 接触点wheel-spin yaw-fit | +.453622 | −.453195 |
| 接触点free-base yaw-fit | +.132870 | −.132691 |
| wheel-spin − base gap | +.320752 | −.320504 |
| 全接触点腿运动项 | +.312911 | −.312694 |
| 同点/同权重的轮中心腿平移项 | +.245896 | −.245730 |
| 腿带动轮体转动的剩余项 | +.067015 | −.066965 |
| slip yaw gap项 | +.007841 | −.007810 |

腿项约占gap的97.56%，但拟议轮中心补偿对应其中的76.66%，另外20.89%是中心到接触点的腿致轮体转动，2.44%为滑移。这里份额是**相同线性拟合的速度恒等式分账，不是独立因果比例、能量比例或改善预测**。late hold出现组件相消时这些比例可超过100%，report如实保留，不能截断成好看的百分数。

pulse完整平面切向速度的端点RMS为.004188/.004180m/s；中心共同纵向u均值的RMS为.002536/.002523m/s。左右腿主要呈相反的纵向相对运动：左转时左侧u<0、右侧u>0，右转镜像。接近脉冲后半段，原wheel目标约±1.5rad/s而机身yaw约±.15rad/s，轮PI仍有平均约.36rad/s的单轮误差；不是“actual wheel已逐位等于target”。

## 实际native载荷：驱动矩与侧向摩擦矩抵消

只读取已保存的native step solved cache；normal均为竖直、pulse无失去所有active wheel contact的子步，四轮端点有载。native force不乘另一个相位的endpoint velocity，不报告混相位的摩擦功或能量。

下表yaw moment拆分严格为world分量：`Mz_from_Fx=Σ−(p_y−reference_y)Fx`，`Mz_from_Fy=Σ(p_x−reference_x)Fy`。它们不是分别独立施加的控制量；normal沿z对Mz贡献为零。

| 左转实际执行窗口 | 2.00–2.50s | 2.25–2.50s | 2.50–3.00s | 3.00–4.50s |
|---|---:|---:|---:|---:|
| normal Fz平均N | 472.103 | 472.202 | 472.045 | 472.505 |
| Mz from world Fx平均Nm | +6.386 | +8.408 | +14.420 | +18.580 |
| Mz from world Fy平均Nm | −5.500 | −8.651 | −14.408 | −18.816 |
| 净Mz平均Nm | +.8863 | −.2429 | +.0121 | −.2359 |
| Σ|Ft|/(.9ΣFn)，时间加权 | 12.71% | 15.64% | 23.74% | 32.98% |
| 同窗口端点滑移RMS m/s | .00419 | .00523 | .00823 | .01711 |

右转对应符号反向、大小接近。脉冲最初10拍左转净Mz+3.1137Nm，后25拍已变负，表明侧向接触反矩逐渐抵消驱动矩。低早段滑移支持“腿变形/相对运动使wheel速度不等于body转动”，但**不能把全8秒都描述成无滑移**；3–4.5秒滑移和摩擦利用明显上升。全episode轮力矩远低于12Nm且实际五子步逐位等于请求，没有证明最大摩擦极限不足，也没有理由直接再提高cap。

## 为什么是加u，以及近似的边界

小姿态误差下，轮中心沿heading的速度为`v_body − yaw_rate*y_i + u_i`；无纵向滑移的近似滚动条件是它等于`R*omega_i`。原目标`(v_cmd−r_eff*y_i)/R`隐含u=0。把**+u_i/R**加入目标，使轮PI不再把腿相对运动吸收的轮速差误当成body跟随。

令`a_center=cov(y,u)/var(y)`，`r_w=−R cov(y,omega)/var(y)`。左转pulse平均a_center≈−.2553rad/s。原投影wheel error为`r_eff−r_w`；校正后变为`r_eff−r_w−a_center`。若近似`body_yaw≈r_w+a_full`成立，则新的误差约为`r_eff−body_yaw+(a_full−a_center)`，中心对应部分消掉，剩余contact/轮体转动项仍在。

不把contact剩余项通过经验系数吸收。当前state确实公开了contact-point Jacobian；本合同仍只选轮中心这个最小变更，不同时引入接触flag切换、接触点选取/分母反演或full angular补偿。若中心候选失败，原始剩余量可供新合同判断；这不是当前预算中的第二候选。

候选只在raw forward精确0且raw yaw非0的原50拍启用，不用servo yaw判门：保持段/直行受扰时servo可非0，仍不应误启用。pulse关闭瞬态完整计入原gates，不加latch或缓变。

## 单拍代数筛查及稳定性限制

每个原保存pulse状态单独用原PI memory计算一次候选目标和PI公式；不调用controller.compute，不把候选integral/状态传到下一样本。这是100个独立影子计算，不是candidate rollout。

| 影子量 | 左 | 右 |
|---|---:|---:|
| 最大|candidate wheel target| rad/s | 2.52409 | 2.52287 |
| 最大|目标增量| rad/s | 1.02553 | 1.02439 |
| 最大|单拍wheel request| Nm | 3.34747 | 3.34303 |
| ±30目标clip / ±12 torque超限 / PI clip或reject | 0 / 0 / 0 | 0 / 0 / 0 |

例如左转执行201，center补偿将约±1.499rad/s的目标变为约±2.47rad/s；等效wheel差动yaw目标约+.9887，而body effective request仍+.6。执行249等效wheel目标约+.8146。必须同时记录这两者，不能把运动学补偿隐去后声称完全相同轮速限制，也不能把较大的wheel目标当body跟随证据。

当前没有明显有限值/单拍保护阻碍，足以提出一个有限实验。但交叉速度反馈的新增轮力矩功没有固定符号：不限幅时增量近似`(2.2+3*.01)*u/R`。它不是腿阻尼，不保证sample/hold下能量下降；轮—腿惯性、摩擦、PI、10ms保持可能使腿/轮运动自我强化。影子τ不大不能证明闭环安全或稳定，名义运动学不能预测gates。候选若仅增大wheel spin/腿形变/滑移而body yaw、原heading峰值不改善，即不支持此机制；改善但仍超过5°也判失败。

## 固定下一合同

见`next_contract.md`：两turn各800＋两stop各800，共**3200新增控制步、16000native子步**，zero残差，复用`flat_plane_02`对应zero基线。两stop全程必须逐位noop，原三项停车失败也保持；不能换用已通过的stop-damper轨迹或要求这两noop case突然通过。两turn必须满足全部原turn gates，且plant/证据/前缀配对通过，才支持本候选在此有限开发集合有效。

未运行candidate、未调用Claude实现。真正Opus有界实现及root独立集成/物理安排另行进行。本任务不依赖停车damper结果，不把两项候选自动组合。

## 复算与校验

```bash
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps:/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/plane_turn_diagnosis_01/audit_turn.py --repo /home/lyh/wheel-legged-control-lab --work /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920 --output /tmp/plane_turn_diagnosis_recheck_new.json
```

输出必须是不存在的新路径。报告SHA256：`25cfa109082efe578e3ff7ab8c3a9d10e3f667e9412f43b2f9656de558ad1450`。合同SHA256：`f39dcfb0c51e72a61ca34a14b9d81ff5b1e60f4112cea6cc30b3111f056fce00`。完整目录成员由`checksums.json`保护，不包含该清单自身。
