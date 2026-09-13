# D1 侧步开发证据

实际 Claude Opus 主笔快步控制模块，Codex 负责接入、数学/API修复和真实动力学验证。本目录记录可公开的小型证据；它独立于 v0.8 正式实验的25个冻结结果目录。

相同3cm侧步请求，实际 course/legacy_mixed 场景从请求到安全完成由40.81–40.83s缩短为10.57s，约3.86倍；flat/synchronized 的左右和90°朝向为10.55–10.58s，约3.95倍。均四腿实际离地、最终四轮接触。物理步长保持2ms、控制周期10ms，时间来自真实 MuJoCo `data.time`。取消场景先落脚才允许交接，不计作完成侧移。

- [comparison.json](comparison.json)：五个同条件对照，含受测源码SHA、完整验收结论及计时定义。
- [validation.json](validation.json)：最终14项pytest、Ruff、直接CLI检查、冻结77源一致性及适用边界。
- [runs](runs)：原始失败与通过场景的完整小型报告；报告保留各自实际受测版本SHA。
- [Opus贡献](provenance/opus_contribution.json)：真实 `claude-opus-5` session、用量、原始代码和响应哈希。原始生成代码仅作出处存档，当时未经执行，不能直接作为已验证实现使用。
- [本地修正](provenance/local_adaptations.patch)：从原始生成版本到最终实现的完整差异。
- [失败记录](failures/history.json)：MuJoCo API失败、过早结束判定，以及GUI03暴露的直接脚本导入缺陷及真实CLI复现。
- [大文件清单](large_artifacts.json)：完整物理轨迹的SHA和大小。模型、权重及NPZ没有复制到本目录，也没有加入git。

最终代码是 [d1_fast_side_step.py](../../scripts/d1_fast_side_step.py)；保守备选仍是原 [d1_side_step.py](../../scripts/d1_side_step.py)。`FastSideStepController` 默认采用快步配置。主运行器的 `conservative` 选项应调用原保守控制器，不应把快步模块内的调参配置误认作逐字节原基线。

数值对照来自保留的v2完整轨迹；后续非有限分配门禁、MuJoCo绑定兼容及脚本导入修复后，14项测试完整重跑通过。真实动力学运行版本为MuJoCo3.12；旧绑定仅有接口协议测试，未宣称另一MuJoCo版本已做同样物理验证。

适用范围是模拟器真值、世界z=0局部平面及实际course的平地起点。尚未证明硬件、随机地形、噪声或外部扰动鲁棒性；这些结果不属于正式PPO结论。GUI人工交互结果由集成验证另行记录，本目录的headless通过不能代替人工交互验收。
