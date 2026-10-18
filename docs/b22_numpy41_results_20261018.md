# C41：转向补验与两场 NumPy 兼容结果（2026-10-18）

C41 为 **COMPLETE**。本次只新执行缺失的 `yaw_1p2_B(seed181203)`：1600 控制拍完整结束，任务、安全、native/controller、转向双向积分和最终停车窗口均通过。将它与独立 C40 已核实的原 C39 flat 保存结果合并后，两项固定场景均通过预注册的 **SSE、torque cost 比值≤1.001** 后端兼容门。

这支持两场 NumPy 后端闭环兼容，不代表 RL 超过纯控制器，不授予完整矩阵资格，也不自动替换默认后端。`qualification_granted` 仍为 `false`；没有新增训练。

| 账目 | 控制拍 | 正常 native | compiler native | predict calls / rows | PPO / Torch load |
|---|---:|---:|---:|---:|---:|
| C41 新执行 yaw | 1600 | 8000 | 2 | 1601 / 1632 | 0 / 0 |
| C40 复核的原 C39 flat，复用证据 | 1600 | 8000 | 2 | 1601 / 1632 | 0 / 0 |
| 两场组合分析集合 | 3200 | 16000 | 4 | 3202 / 3264 | 0 / 0 |
| 原 C39 父回执总数 | null | null | null | null | null |

组合行是跨独立范围纳入分析的完整案例计数，不是历史总资源支出。C40 自身新增物理与 NN 均为零；原 C39 父回执仍为 `INCOMPLETE`，原 `actual_counts`、`actual_nn_counts` 仍为 `null`，没有改写历史或重跑 flat。

| 场景 | C35 Torch gain 0.5 SSE | NumPy SSE | SSE 比值 | C35 torque cost | NumPy torque cost | cost 比值 |
|---|---:|---:|---:|---:|---:|---:|
| flat 1.6 m/s，原 C39 / C40 复核 | 1.6770574345 | 1.6770574374 | 1.0000000017179276 | 19.8206290776 | 19.8206291022 | 1.0000000012426467 |
| yaw 1.2 m/s，独立 C41 | 20.2260826620 | 20.2260826704 | 1.0000000004140341 | 19.6460109560 | 19.6460109749 | 1.0000000009599406 |

每场均对齐固定 C35 同 seed、完整控制器初态/reset、编译机器人/场地及评分窗口。flat 的 drive 分母为720拍，yaw 为640拍；各自完整保持段400拍、最终段100拍，安全检查覆盖初始状态及8000个 native 样本。torque cost 是归一化力矩平方代理，不是机械能或电耗。此处未承诺逐位轨迹相等，也没有测量完整控制回合加速。

yaw 保持段平均速度为1.2015209 m/s，速度 RMS 误差0.00589671 m/s；yaw rate RMS 误差0.0863630 rad/s。正、负指令段各200拍，分别积分为+0.2580736和−0.2694116 rad。最终100拍平均绝对速度0.00009768 m/s、平均绝对 yaw rate 0.00038936 rad/s。原 yaw 任务及最终窗口门通过；flat 专用4.2秒/3.2米停车门不套用于 yaw。停车制动沿原基控执行，不能记为 RL 收益。

C41 总用时239.8250秒（含 scope 准备；父监督为239.2288秒），最低有效可用内存6,215,163,904 B（5.7883 GiB），高于4 GiB运行停止线；启动门6 GiB、Nice10、CPU单线程，未发生阶段间内存等待。worker/host正常退出，来源失配为空，父与host清理均无残留、无发送清理信号。user unit 已为 inactive/dead、退出码0。

冷 bootstrap 核查为零环境构造、零 NN、零 native。实际 yaw 的NumPy账本为一次32行probe加1600次单行控制预测；critic、PPO/Torch load均为零。32行probe的clipped16最大绝对误差3.7252903e-8，半增益动作误差1.8626451e-8，均低于原1e-6门；`probe_actions_byte_exact=false`如实保留。保存reader和本次独立审计都没有新增NN forward或物理步。

独立核对重新流式哈希426份文件、约135.53 MB，包括新yaw的26份绑定工件、原C39全部60份保存文件、两场C35参考102项和36版导出9项；新三份源码与提交Git blob一致。父→release→bootstrap/worker→保存读回→comparison的身份、typed计数、原始SHA、配对与指标算术均吻合。本次不重新解析大型gzip/NPZ、不重跑reader或模型，也不额外扫描数GiB的外部依赖；后者仍由运行时冻结清单和既有来源检查约束。详细路径、字节数、SHA及审计边界见[机器结果](../reports/b22_numpy41_results_20261018.json)。

执行源提交为 `6a3d84983b3f57514a6c810de368272567f4ba01`，范围与一次性限制见[41执行说明](b22_numpy41.md)。指定[CI运行39104167314](https://github.com/LYHrmer/wheel-legged-control-lab/actions/runs/39104167314)已核对为同一提交，撰写时仍在运行；CI状态与仿真后端兼容结论分别记录。
