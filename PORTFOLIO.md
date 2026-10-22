# 求职向概览：Wheel-Legged Control Lab

> 给招聘方/面试官的 2 分钟入口。实验细节在 `docs/`，这里只放结论、证据和边界。
> 每条结论都能在本仓库找到可核验的运行记录；面试时只讲自己能讲清机制的部分。

## 一句话

在同一台 16 自由度轮足机器人、同一层 VMC 基控上，对照 **LQR / 约束 MPC / 16 维有界残差 PPO** 三种外环，并用配对种子、预注册门槛和独立读回保证结论可复现。

## 60 秒版本

- 平面 6 状态教学模型上，残差 PPO 把高度 RMSE 从 99.3 mm 降到 22.4 mm，速度 RMSE 从 0.565 降到 0.480 m/s（20 个配对随机种子，0 失败）。
- 完整 D1 模型上，先做诊断再改控制器：定位到闭环偏航增益只有 0.63，加入指令前馈后六任务合池跟踪 SSE 降低 28.2%，转向任务降到 0.18×，非转向任务逐位不变。
- 残差 RL 的贡献门（0.85）始终没有达到；我没有用单次训练结果硬凑结论，而是用 seed 复制和样本外检验证明了这个规模下**不可分辨**。这条负结果推动了另一个仓库的纯 PPO 路线。

## 架构

```
D1 整机 23 nq / 22 nv / 16 执行器 · 物理 500 Hz · 决策 100 Hz
  └── VMC 底层
        ├── LQR 外环
        ├── 约束 MPC 外环
        └── 16 维有界残差 PPO（叠加在基控上）

6 状态平面教学模型（base_x / body_pitch / leg_extension，2 执行器）
  └── LQR / 线性 MPC / 2 维残差 PPO
```

## 三条可核验结果

| 结果 | 数字 | 证据 |
|---|---|---|
| 残差 PPO 改善高度跟踪 | 高度 RMSE 99.28 → **22.38 mm**；速度 RMSE 0.565 → **0.480 m/s**；20 配对种子、0 失败 | `results/benchmark/randomized_audit.md` |
| 偏航前馈（基控改进） | 六任务合池 SSE 57.69 → **41.41（−28.2%）**；转向任务 19.93 → **3.65（0.18×）**；零指令任务逐位不变 | `docs/b22_yaw_feedforward44_results_20261018.md` |
| 残差 RL 的可分辨性 | 同配置换 seed，镜像合池比在 **0.9324 / 0.8586** 间摆动（7.4 点），臂间均值差仅 0.0095；样本外回落到 0.9284 | `docs/b22_seed47_results_20261020.md`、`docs/b22_sealed47_results_20261021.md` |

## 技术栈

Python、NumPy、PyTorch、Stable-Baselines3、MuJoCo、ONNX Runtime、systemd（cgroup 资源边界）、GitHub Actions；C++/ROS2 部署在姊妹仓库 [ddt-d1-rl-lab](https://github.com/LYHrmer/ddt-d1-rl-lab)。

## 三个可深挖的面试故事

### 1. 为什么没有把 0.85 门"凑"过去：先测噪声底

- **现象**：多个奖励/课程臂在 0.883–0.952 之间，看起来"某个奖励更好"。
- **做法**：同一父代、同一课程、同一预算，只换 PPO seed，得到同配置极差 0.0738；预注册"分离规则"（一臂所有 seed 都优于另一臂）而不是比较单点。
- **结果**：分离规则不成立；把"最好"的策略放到没参与选择的封存案例上，比值从 0.8586 回落到 0.9284。
- **可追问**：为什么不是 2σ？——n=3 时样本标准差的相对不确定度约 50%，2σ 会系统性误判；所以用分布无关的分离规则，并给出极差而不是伪造置信区间。

### 2. 从保存数据到控制器改动：偏航前馈

- **现象**：转向任务误差大，RL 怎么调都只改善 3%。
- **做法**：只读保存的控制记录，发现实测偏航率只有指令的 0.63 倍、零滞后——是基控的静态增益缺口，不是 RL 问题；据此设计 `3.9·cmd + 4·(cmd−实测)`。
- **结果**：六任务合池 −28.2%，转向 0.18×；同时验证了反例——整体提高反馈增益会让非转向任务退化到 2.05×，所以不能"一把增益调到底"。
- **可追问**：这算 RL 吗？——不算，是基控制器改进；仓库里两者分开报告，贡献门只用于 RL。

### 3. 在冻结协议上做单变量实验：24 处硬编码与一次系统漂移

- **背景**：想把训练预算从 32768 翻倍，验证"预算是不是杠杆"。
- **过程**：冻结协议在 4 个模块里分散写死了预算常量（派生共 24 处替换）；前 12 次失败每轮都暴露一处，最后又遇到系统库 `libXtst` 版本漂移。
- **做法**：改为计数式派生（替换次数不符即失败）+ 静态 API 检查 + "派生后不得残留旧常量"的测试；环境漂移时选择**归档而不重绑来源清单**，保住旧运行的逐字节复核能力。
- **结果**：实验按统计设计归档（n=2 无法判定 0.01 的效应），但 13 次失败的根因、修复与证据全部保留。
- **可追问**：为什么不继续？——因为 2× 臂的均值标准误约 0.037，是判定阈值的近 4 倍，跑通也给不出结论。

## 简历可直接用的条目

- 在 MuJoCo 中搭建 16 自由度轮足机器人控制栈（VMC 基控 + LQR/MPC/有界残差 PPO），20 个配对随机种子下残差 PPO 将高度 RMSE 由 99.3 mm 降至 22.4 mm、速度 RMSE 由 0.565 降至 0.480 m/s，0 失败。
- 通过保存数据诊断定位闭环偏航增益仅 0.63，设计指令前馈使六任务合池跟踪 SSE 降低 28.2%，转向任务降至 0.18×，非转向任务逐位不变；并用反例排除了"整体提高反馈增益"的方案。
- 建立可复现评测纪律（预注册门槛、独立读回、SHA 来源闭包、CI），用 seed 复制证明单次训练结果的摆幅（7.4 点）大于实验臂差异，避免了基于单点数字的虚假结论。

## 边界（建议主动说）

- 全部为 MuJoCo 仿真，未包含本仓库版本的实机验证。
- 单机 CPU、任务组 3 GiB、无 GPU；残差 RL 的 6 个评测任务各有单一种子，样本量远低于开源四足 RL。
- 机器人模型与原始策略来自与合作者共同开发的资源包；本仓库的工作集中在控制栈、训练/评测管线和验证方法。
- 负结果与撤回项原样保留：46 的"最佳"、45 的粗糙课程、48 的 2× 预算都在文档中标注。

## 推荐阅读路径（给面试官）

1. 本页 + README 架构图；
2. `docs/b22_final_status_20261021.md`（已站得住 / 已撤回 / 未解决）；
3. `docs/b22_yaw_feedforward44_results_20261018.md`（诊断 → 候选 → 验证）；
4. `docs/b22_paper_section_20261022.md`（方法学与可分辨性，含可写/不可写表述）；
5. `results/benchmark/randomized_audit.md`（20 配对种子对照）。

## English summary

A single 16-DoF wheel-legged robot, one shared VMC base controller, three outer loops (LQR, constrained MPC, bounded residual PPO), evaluated with paired seeds and pre-registered gates. Residual PPO cuts height RMSE from 99.3 mm to 22.4 mm and velocity RMSE from 0.565 to 0.480 m/s over 20 paired seeds. A diagnostic-driven yaw feed-forward lowers pooled tracking SSE by 28.2% (0.18x on turning tasks, bit-identical elsewhere). Seed replication shows a 7.4-point swing for one seed, larger than any reward-arm difference, so single-seed claims are deliberately not made. Simulation only.
