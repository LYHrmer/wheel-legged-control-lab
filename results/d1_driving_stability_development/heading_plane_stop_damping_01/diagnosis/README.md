# Plane 停车离线诊断：固定腿阻尼有测试依据，尚未采用

2026-09-20，gpt-6-astra / ultra。两场 native-plane 停车的已存轨迹支持继续测试原固定 `b=126.4374005337902 N·s/m/leg` 候选：晚段车体仍回摆，轮速很小，接触滑移也很小；速度差主要对应腿相对运动和机身转动。没有证据支持在本轮改 wheel PI、放宽门槛或重新选择 b。此结论是独立有界实验的依据，不是修复结论。

本目录仅做离线诊断和下一合同规划，**新增积分0、controller.compute调用0、训练0**。未修改仓库或已有输入。`audit_stop.py` 将 `mj_step`、`mj_step1`、`mj_step2` 封锁；仅编译原 native-plane model，调用几何、速度、Jacobian、给定姿态惯量计算。所有审计 MjData.time 保持0，qpos/qvel保持原数组。执行环境为 MuJoCo 3.12.0。

## 原失败准确复算

输入是 `flat_plane_01/{flat_forward_stop,flat_reverse_stop}` 两个完整封存case，每场800转移、801状态、4000实际native子步。`complete_manifest.json` 的全部成员hash匹配；77份冻结输入匹配。batch的初态 `+0/-0` 比较故障没有使这两场失效，也没有改变以下评分。

| 原指标 | 正向 | 反向 | 原门槛 |
|---|---:|---:|---:|
| 全程raw COM body-vx RMSE m/s | .05904251 | .05877599 | ≤.05 |
| endpoint tick500..799最大绝对COM body-vx m/s | .08435578 | .08423582 | ≤.03 |
| position tick500..700平面路径 m | .06299257 | .06281948 | ≤.05 |

三项均失败，其余原停车门槛保持通过。上述值从原raw命令、端点速度和保存位置独立复算，与原summary差不超过1e−12。不能把7–8秒最终已接近停止替代原5–8秒速度门槛，或移动晚段起点。

## 相位、方向和恒等式

对状态索引k，`states[k]`、`trace[k-1]`端点、`plane_diagnostics[k-1].endpoint_contacts` 都是t=k×.01秒；k<800时还与`plane_diagnostics[k]`的before字段一致。重建的是 **base_link inertial-COM速度投影到当前base body-x**，不是可见body origin的差分，也不是总机器人COM。重建与trace的速度最大误差为0；wheel rolling及轮中心腿相对vx与同端点before日志吻合至1e−12。

world z向上，平地接触法向为+z；正向body-x沿车头方向，反向停车的速度保留负号。逐接触点，使用完整Jacobian分解：

```text
COM body-vx − r*qdot_wheel
  = contact tangent body-x + contact normal body-x
    − [omega_base × (point − base_COM)]_body-x
    − J_leg,body-x*qdot_leg
    − (J_wheel,body-x*qdot_wheel + r*qdot_wheel)
```

r=.087m。逐点闭合误差最大为正向1.21e−16、反向8.33e−17m/s。未将body速度和下一端点轮速混合，也未把轮中心Jacobian当接触点Jacobian。

主统计在每个端点用已记录的非负normal force作为接触权重；同一组权重也用于wheel rolling，保持恒等式。另保存“四轮等权、轮内normal-load加权”结果；本批诊断范围k400..800每个端点的四轮都有正normal load，没有缺轮样本或底点代理。本文百分数是组件对速度差的投影 `sum(component*mismatch)/sum(mismatch²)`，相加为1；**它们不是独立因果份额、能量份额或概率**。

## 腿运动与真实滑移

| tick500..799端点指标，m/s RMS | 正向 | 反向 |
|---|---:|---:|
| COM body-vx | .032327 | .032246 |
| 四轮平均r*qdot | .001951 | .001940 |
| 载荷加权COM−wheel速度差 | .032078 | .032114 |
| 腿运动分量 | .023890 | .024030 |
| base角运动分量 | .008261 | .008156 |
| 接触切向body-x滑移分量 | .000414 | .000410 |
| 完整平面切向滑移速率 | .000605 | .000603 |

晚段速度差投影：腿74.45%/74.80%，base角运动25.69%/25.32%，slip −.14%/−.12%；极小的负投影仅表示相位抵消。刚停后k400..449，腿分量RMS为.09696/.09478，角运动.03012/.03209，slip-x仅.00222/.00201m/s。几何轮轴项和normal投影更小。

两个明确端点帮助解释该结果：

- 正向k401：机身+.25871，轮滚动+.04170m/s，四腿轮中心相对vx平均−.16221m/s。轮已显著减速时，腿相对机身向后运动，机身尚未停止。
- 正向k531：机身+.08436，轮滚动+.000236m/s，四腿轮中心相对vx平均−.04902m/s。完整接触点恒等式中腿分量+.06330、角运动+.02088、slip-x−.000083m/s。反向同端点分别为−.08424、−.000247和腿中心+.04927，镜像趋势一致。

这里的候选阻尼作用于**轮中心**非轮三关节的相对vx；.04902的轮中心速度与.06330的接触点腿分量不是同一个量。它不能直接消除所有机身角运动，也不是位置锚定或body速度反馈。

## 支撑载荷有实际native证据

同步端点日志保存了`measurement_data`经原同步刷新后求得的contact force和Jacobian速度。它们可在同一个端点用于载荷加权分解，但不是刚刚执行的5个native子步平均力。

**另行读取实际native求解缓存**：每个2ms `mj_step`返回时已保存的力，不重算历史力，不在live data上forward。native记录本身没有同相位接触速度，所以本报告没有用其force乘另一个端点velocity冒充摩擦功。

| 实际native窗口 | 正向 | 反向 |
|---|---:|---:|
| 4–4.5s平均normal Fz N | 471.883 | 473.748 |
| 4–4.5s平均tangent Fx N | −33.091 | +33.057 |
| 5–8s平均normal Fz N | 472.311 | 472.324 |
| 5–8s tangent Fx RMS N | 6.577 | 6.561 |
| 5–8s累计Σ|Ft|/(.9ΣFn) | 1.963% | 2.053% |

机器人质量48.146865kg；法向载荷约等于其重力。在这些窗口，native水平法向最大0，无“所有轮都无active contact”的子步。端点四轮载荷持续、切向速度很小且native整体切向利用量很低，支持腿/俯仰回摆解释，反对“巨大的持续打滑解释了晚段.08m/s车体峰值”。窗口平均利用率不能排除某单个接触瞬间接近摩擦边界；10ms端点速度也不能排除2ms间隔内短促滑移。没有把这些边界提升为零滑移或摩擦根因排除证明。

## PI、回摆和候选代数筛查

停令执行tick400的实际四轮平均力矩为正向−5.91826、反向+5.91825Nm，五个子步均与请求逐位一致。正向比例项−6.31838Nm、更新后积分+.40012Nm；反向为+6.31840和−.40015Nm。它们立即沿制动方向作用，不是release首拍继续沿原方向驱动的模式。两场全部800拍都没有torque protection，实际16轴每个子步均等于请求。PI会参与后续耦合动力学，但没有积分饱和/长期主导wrong-direction torque的证据，不能据此加入积分清零。

两方向首次反向均在状态k435。正向首反向峰k465为−.14581，反向对应k464为+.14424m/s；随后同原方向峰均在k531，为+.08436/−.08424；下一反向峰k598约−.04537/+.04528，再下一峰k663约+.02063/−.02062。相邻同向峰约1.32–1.34秒，形成衰减回摆。原方向最大延伸.05115/.05060m，峰后最大回撤.06130/.06085m，释放后总反向路程.08207/.08160m。

在**新plane名义qpos[0]**重算原fixed-wheel-center pure-x模式，得到与原设计完全相同的M_eff=40.173203899817445kg，K_eff=2163.081163077078N/m，B_old=83.81939506923678N·s/m，Bcritical=589.5689972043975N·s/m，b=126.4374005337902。这是复核，不是按新表现调参。名义1.168Hz与观测约.75Hz有差异；该模式仍遗漏重力/支撑几何刚度、俯仰耦合、轮PI与滚动接触，不能声称实际闭环会达到临界阻尼。

只在原plane保存轨迹k400..799计算候选公式，未运行控制器：

| 固定b代数量 | 正向 | 反向 |
|---|---:|---:|
| 最大单关节增量 Nm | 7.84876 | 7.82589 |
| 原unlimited请求+增量 / 额定限矩的最大值 | .49351 | .49331 |
| 最大采样增量关节功 W | −3.29e−6 | −1.46e−6 |
| 最小采样增量关节功 W | −13.9668 | −13.8887 |
| 最大dt·lambda_max(M⁻¹D_new) | 1.08601 | 1.07776 |

没有发现明显超限/非有限值/采样符号问题，支持进入一次有界候选比较。非正功只属于未保护增量在采样时刻的代数性质；逐关节安全保护后的实际增量功应另外记录，10ms保持期间也可能发生速度变号。自由惯量指标仅是纯阻尼粗筛，不是离散闭环稳定证书。新轨迹上的限幅、接触和性能仍未知。

下一实验的具体范围见 `next_contract.md`：独立plane身份、只运行4条候选轨迹，最多新增4000控制步/20000子步；复用完整封存的plane zero对应基线，不重跑bypass。

## 复算及归档

helper读取原case的complete manifest，核对原记录及77冻结源；执行末尾再次核对全部输入hash。三项评分复算、相位对照和逐点恒等式均带断言；未来不同结果可使检查失败。当前分析没有执行修改控制律后的“绿色”轨迹，因此不声称完成修复验证。

```bash
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps:/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/plane_stop_diagnosis_01/audit_stop.py --repo /home/lyh/wheel-legged-control-lab --work /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920 --output /tmp/plane_stop_diagnosis_recheck_new.json
```

输出路径必须不存在。`report.json` SHA256为 `0ccbdfdc4cc7234ae1395585785b28b343667d0b52da614f7959bb916d30edc7`。目录成员由`checksums.json`保护；该清单不包含自身。
