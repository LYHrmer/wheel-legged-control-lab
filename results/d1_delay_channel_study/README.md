# D1 测量延迟与执行器延迟的开发诊断

本目录独立调用未修改的 `D1LocomotionEnv`，固定零残差 `wheel_leg` 控制器，分开测试测量延迟与执行器纯延迟。没有新增 RL 训练、增益搜索或留出集调参，也不宣称已修复稳定性。

正式结果目录是 `development60s_final/`。`development60s/` 是因补充低层限幅日志而中断的旧版本；它保留原始数据，但不构成完整研究。

## 完成的 24 例结果

正式运行于 2026-09-11 06:50:38 UTC 完成，累计案例运行时间 305.26 s。24 例均保存，无异常；10 例完成 60 s，14 例因 `fall_or_body_contact` 提前终止。71 份运行源码及模型资产结束时 SHA 全部一致，独立分析通过，完整指标在 `development60s_analysis.json`。

| 带噪声融合器的单一延迟 | 测量通道：完成数、失败时刻 | 执行器通道：完成数、失败时刻 |
|---|---|---|
| 0 ms 公共基准 | 2/2，均完成 60 s | 同一公共基准，不重复计数 |
| 10 ms | 2/2，均完成 60 s | 2/2，均完成 60 s |
| 20 ms | 0/2，20.23 / 20.65 s | 0/2，21.55 / 21.69 s |
| 30 ms | 0/2，1.90 / 5.62 s | 0/2，1.16 / 2.48 s |

两个 20 ms 通道都通过了 seed17 的前 3 s 站立检查，却未完成 60 s 道路。其非平坦几何曝光比例分别为测量通道 45.53% / 46.68%、执行器通道 48.86% / 49.15%。所有 30 ms 案例均在进入非平坦区域前失败，因此不能把它们解释为复杂地形上的验证。

| 来源消融 | 0 ms | 30 ms |
|---|---|---|
| 无噪声融合器，仅测量延迟 | 2/2 完成 | 0/2，均 2.31 s |
| 即时真值，仅执行器延迟 | 2/2 完成 | 0/2，均 1.09 s |
| 无噪声延迟真值 | 共用即时真值基准 | 0/2，均 2.88 s |

按实验前排序假设解释：H1「闭环延迟可在融合器之外引起失败」获得支持：相同控制器下，无噪声延迟真值以及即时真值配执行器延迟均失败。H2「融合器是必要条件」与 H3「合成测量噪声是必要条件」不被这些案例支持；来源与噪声仍会改变失败时刻，不能推断它们毫无影响。H4「执行器零填充启动队列主导失败」仍未解决：逐物理步队列符合声明模型，但没有暖启动队列消融，不能单凭失稳较晚出现就排除启动影响。

这是一项完成的机制对照研究，没有修复延迟稳定性，也没有定位某个具体 PD 分支或证明稳定裕度。14 个失败案例都曾出现控制器内部力矩限幅，10 个完成案例没有出现这一项力矩限幅；时间先后和共现不能证明限幅是起因。失败的具体触发条件为 7 例机身接触、7 例姿态超过 0.85 rad，没有案例因净空低于 0.22 m 触发终止。

## 固定协议

12 个配置 × v1 开发道路 road0 × 回合 seed 17/29，共 24 个 60 s 案例。轮速 PI 为 0.55/1.5，偏航反馈 4，腿部 PD 比例 1，姿态 PD 比例 0.25。执行器时间常数为零、增益为 1，所有域随机化关闭。

| 配置族 | 测量延迟 ms | 执行器延迟 ms | 配置数 |
|---|---:|---:|---:|
| 带噪声融合器，公共基准 | 0 | 0 | 1 |
| 带噪声融合器，仅测量延迟 | 10、20、30 | 0 | 3 |
| 带噪声融合器，仅执行器延迟 | 0 | 10、20、30 | 3 |
| 理想无噪声融合器 | 0、30 | 0 | 2 |
| 即时真值 oracle | 0 | 0、30 | 2 |
| 无噪声延迟真值 | 30 | 0 | 1 |

测量周期 10 ms，对应延迟 0/1/2/3 拍；物理周期 2 ms，对应执行器延迟 0/5/10/15 拍。初始化阶段测量保持 t=0，执行器延迟队列零填充。短检查截取原 60 s 命令的前 3 s，不缩放命令时序。

oracle 与 fusion 的对比同时改变状态和地面参考来源，不能单独归因于某一融合方程或支撑面算法。无噪声的种子案例没有随机噪声差异，不能把相同确定性轨迹当成两份独立随机证据。两条种子轨迹和一条已使用的开发道路不足以证明泛化。

## 复现与记录核验

`reproduce_delay_failure.py` 是最小红检查：真实闭环前 3 s、测量延迟 30 ms、road0/seed17。已重复两次，在 190 拍（1.90 s）以 `fall_or_body_contact` 结束，断言返回 exit 1。记录区分机身接触、姿态阈值和高度阈值，不把这个联合终止名称自动解释为同一种机制。

可移植的指标与分析器测试共 21 个检查通过，包括错误延迟单位、提前终止不得报告完整时域指标、缺案例、删改协议、源码哈希变化、篡改执行器队列、低层限幅和伪造 RMSE 的拒绝。分析器单测使用自包含的数值夹具，不要求主仓中存在本目录的数据；真实 12/24 例分析和源码恢复另行验收。

`logging_parity.json` 证明初版与最终补日志版 12 个短案例的所有共同 CSV 字段与 NPZ 数组逐位相同。`snapshot_recovery_parity.json` 证明仅使用归档源码、URDF/mesh 与依赖可重跑同样 12 个短案例，全部数组和 CSV 字段逐位相同；恢复运行没有导入主仓的 src/scripts。

`analyze_delay_channel_study.py` 独立使用 NumPy 从原始 CSV 复算 RMSE，与 runner 的 `delay_metrics.summarize` 不共享计算实现。它固定检查 12 配置、种子、道路、时长、控制器、两个时钟、源码快照、结束标记、每个物理子步的延迟力矩与测量时间对齐。失败案例的 `full_horizon_tracking` 始终为 null；只有双方都跑满时才提供配对的完整时域 RMSE 差。

CSV 包含：当前与测量时刻对齐的状态误差、控制/测量/决策时间戳、三个真值终止阈值、接触数、实际地形曝光，以及低层力矩限幅、关节限位保护和目标限幅。执行器通道的输入限幅为零不能解释成整个控制器没有饱和；低层在通道之前已有限幅。

低层各项 fraction 每个控制拍以 16 个轴为分母，再对已执行控制拍求均值，即「受影响轴拍数 / (16 × 已执行控制拍数)」，不是「至少一轴限幅的帧比例」。`controller_torque_clipped_fraction` 只比较限幅前请求与名义力矩裁剪；`controller_torque_changed_fraction` 还包括关节位置/速度保护产生的归零。执行器通道的 `torque_input_clipped_fraction` 则在每拍 5 个物理子步和 16 轴上求均值，统计层级不同。

NPZ 中的 `qpos/qvel` 是原生 MuJoCo 广义状态。`estimated_states` 的列依次为可见基座原点世界位置 xyz、RPY、COM 世界线速度 xyz、机体系角速度 xyz。`controller_trace` 维度是控制拍×字段×16轴，字段名在 `controller_fields` 中；`actuator_trace` 是控制拍×5物理子步×字段×16轴，字段名在 `actuator_fields` 中。CSV 的 t 为本拍积分后的发布时间，对应力矩数组的积分区间是 `[t−10ms,t)`。

## 移植后的命令

将三个脚本 `run_delay_channel_study.py`、`delay_metrics.py`、`analyze_delay_channel_study.py` 放入目标仓库 `scripts/`。安装该仓库依赖后，从仓库根目录运行，输出目录必须尚不存在：

```bash
rtk proxy env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B scripts/run_delay_channel_study.py --duration 60 --seeds 17 29 --output results/my_delay_channel_study
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B scripts/analyze_delay_channel_study.py results/my_delay_channel_study --output results/my_delay_channel_analysis.json
```

脚本默认仓库根目录为脚本目录的上一级，也可显式传 `--repo-root`。源文件、URDF/mesh 与候选脚本随结果保存，源文件 SHA 使用复制的同一份字节计算，运行结束再次核对一致性。输出中有 `finished.json` 且分析通过才算完整交付。

## 实际 Claude Opus 代码贡献

`delay_metrics.py` 的原始实现由本地 Claude CLI 的 `--model opus` 写入；返回模型标识为 `claude-opus-5`，session `8e1f5418-af04-4ac7-9b18-ee59f3d934b8`，exit 0，用时 55.46 s。原始 SHA-256 为 `53f7b358a381a56631cae66c006a94b675a02eb55b44cf5b66f5492d484711a2`，该版本保存在正式结果的源码快照中。结构化调用记录为 `opus_retry_evidence.json`。第一次调用超时 exit 124、未写出代码，单列在 `opus_evidence.json`，没有算作成功贡献。

## 合入后的检查

主仓脚本整理了格式及异常日志；非法类型现在抛出 `TypeError`。分析器补上了案例身份校验，正式 60 s 记录缺少低层力矩轨迹时直接拒绝。专项测试共 33 项通过，源码快照中的生成版本没有改写。

`integration_validation/final_smoke_parity.json` 记录合入版重跑的 12 个短案例：CSV 和 NPZ 与原始短案例逐位一致。同目录的 `extra_field_audit.json` 另外检查实际状态源与噪声、每拍命令及关节保护，24 例全部通过。这些补充核验依赖归档的仿真模块；仅用 NumPy 的主分析器仍可独立复算误差与延迟队列。
