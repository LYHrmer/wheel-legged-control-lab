# D1 开发场景偏航消融：轮间差分与腿残差

结论：**去除轮间差分残差后，6/6 模型的共同前缀航向 RMSE 下降，32 秒完成数从 1/6 增至 5/6；这尚不是总体优化。** 横向位移 RMSE 只有 5/6 下降，48002/unbounded 从 0.02427 增至 0.03077 m，两者本来都很小，不能写成“6/6 横偏大幅改善”。

本包记录同一个 `dev_straight_road` 开发场景的 19 个闭环 episode：共享零残差基线 1 次，以及三个种子、两种已训练模型各 3 种推理模式，共 **56,844 次真实环境 transition**。没有新训练，没有新最终场景。每个分支从相同初态开始，每拍使用该分支当前观测重新预测；干预动作通过标准环境执行并写入下一拍 previous-action。

| 模式 | 动作处理 | 完成 32 秒 |
|---|---|---:|
| zero | 8 维残差全零，仅运行一次 | 1/1 |
| full | 原始确定性预测 | 1/6 |
| zero_legs | 只清零 action[:4]，保留轮残差 | 3/6 |
| common_wheels | 只把 action[4:] 替换成其平均值，保留腿残差 | 5/6 |

## 结果及具体终止原因

- common_wheels 在六个模型上均降低共同前缀航向和偏航角速度误差。其中五个模型的航向 RMSE 降幅为 94–99%；48002/unbounded 降幅约 38%。它的 full 本来就未明显侧漂，主要问题是俯仰失稳。
- common_wheels 唯一未完成者为 **48002/unbounded 的前方 x 边界退出**：29.46 秒，x=5.300823 m，y=-0.045558 m，最终航向变化 -0.004996 rad。不能把该次 map_boundary 解释成侧向漂航。
- **48002/unbounded 的 full 是世界俯仰越限**：21.59 秒，pitch=0.8500047509 rad，刚超过 0.85 rad 阈值；roll=-0.0660 rad，离地间隙 0.596866 m，高于低间隙阈值 0.22 m，undesired_ground_contacts=0。因此 `fall_or_body_contact` 标签在此不能解释为已发生身体碰撞。其余提前终止的 full 分支均触及 y 边界。
- 只清零腿残差后，共同前缀高度误差在 6/6 模型下降，偏航改善却不稳定。common_wheels 只在 3/6 模型降低速度误差，并在 5/6 增大高度误差。轮间差分、共同轮速与腿残差必须分别评估。
- 零残差基线完成 32 秒：航向 RMSE 0.002486 rad，速度 RMSE 0.039719 m/s，最终 y=-0.009211 m。

每行共同前缀取 zero/full/zero_legs/common_wheels 四分支的最短记录长度，直接截取真实记录，不补齐已终止轨迹。完整时长及终止保留在 `summary.json`，逐个触发条件见 `termination_details.json`。

| 模型 | full / zero_legs / common_wheels 时长（秒） | 共同前缀（步） | 航向 RMSE full→common（rad） | 速度 RMSE full→common（m/s） |
|---|---:|---:|---:|---:|
| seed48001/unbounded | 29.50 / 30.99 / 32.00 | 2950 | 0.344936 → 0.020239 | 0.039349 → 0.060348 |
| seed48001/bounded | 26.30 / 27.35 / 32.00 | 2630 | 0.433095 → 0.005294 | 0.025687 → 0.044430 |
| seed48002/unbounded | 21.59 / 32.00 / 29.46 | 2159 | 0.020941 → 0.012980 | 0.142422 → 0.086023 |
| seed48002/bounded | 32.00 / 32.00 / 32.00 | 3200 | 0.238510 → 0.001990 | 0.122569 → 0.042889 |
| seed48003/unbounded | 28.62 / 32.00 / 32.00 | 2862 | 0.496420 → 0.004748 | 0.032864 → 0.036851 |
| seed48003/bounded | 26.73 / 27.90 / 32.00 | 2673 | 0.618730 → 0.005726 | 0.062175 → 0.039702 |

## 数据、来源与复核

`episodes/<label>/trace.npz` 保留全部逐拍指标、奖励项、指令、预测/干预/执行动作、终止标志和原因，以及完整 observation/qpos/qvel/time 状态。状态有 N+1 行，首行为 reset；其他数组有 N 行。原始 float32 观测和 float64 状态/指标均精确保留。列名、单位与终止缓存观测例外见 `compact_schema.json`。每例还提供初态和汇总 JSON。

`full_rerun_bitwise_verification.json` 绑定旧 full 与新 full 的四组状态数组 shape/dtype/字节 SHA：**六个模型全部逐位相同**。`local_raw_manifest.json` 记录本地完整原始 trace/NPZ/协议/汇总的 SHA 和字节数，原始记录未改写。打包只删除逐拍重复的结构字段和说明文字；本包不是只提供抽样点或汇总。

六个模型不重复拷贝。`protocol.json` 将模型及 sidecar SHA 绑定到本仓库 `results/d1_wheel_common_mean_development/models/` 的路径，并保留训练/环境源码和评测协议哈希。独立原始审计 1,298 项通过，见 `independent_raw_audit.json`；紧凑包的再审计见 `compact_audit.json`。两个审计器都不执行物理。

从仓库根目录复核紧凑数据及各分支策略推理：

```bash
PYTHONPATH=.local-deps:src:. python3 -B results/d1_driving_stability_development/yaw_ablation/audit_compact.py --repo . --output /tmp/d1_yaw_compact_audit.json
```

输出路径必须尚不存在。环境需安装本仓库 RL 依赖。`audit_yaw_ablation.py` 是针对本地完整原始记录的原审计器；`audit_compact.py` 可直接复核当前公开包。

Claude 实际回执为 `claude-opus-5`，费用 **$0.356285**，上限 $1。`opus_original.py`、`opus_prompt.txt` 和脱敏 `opus_receipt.json` 保留来源；实际运行代码为审阅后的 `run_yaw_ablation.py`，修正清单见 `runner_review.json`。

这是一组开发诊断的推理动作干预，不是新的训练策略。状态改变会带来耦合反馈，因此不能把模式差值解释为各部件可相加的独立贡献，也不能据此宣称跨场景总体性能已经提升。
