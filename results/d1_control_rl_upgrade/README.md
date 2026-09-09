# 五项控制与 RL 实验验收

这次工作完成了五组可运行的实验，并把它们接入[学习入口](../../docs/control_rl_experiments.md)。新增功能在仿真中运行过，算法对照没有得到“RL 已稳定优于传统控制”的结论。默认键盘演示和旧 checkpoint 没有被新实验替换。

| 工作 | 实际记录 | 结果与边界 |
|---|---|---|
| 颠簸失效与参考动态 | [120 回合、48,000 步](../d1_reference_dynamics/README.md)；独立复算误差分型与滤波递推 | 前馈通过数仍为 13/24，50/100 ms 姿态低通降到 12/24。COM 速度近似及日志前后时刻另有说明，未推广到默认控制器 |
| 残差反馈对照 | [148 回合](../d1_residual_feedback/README.md)；18 对名义动作回放逐位一致 | 遮掉旧策略的 Fz 后，高度误差减少约 2.7 mm；扰动下的反馈收益没有稳定达到预设门槛 |
| 新动作方案训练 | [9 模型](../d1_action_ablation_analysis/README.md)，每个 131072 步；324 开发回合、240 新留出回合 | 基线 14/24。双通道三个 seed 为 14/14/15，有界均值 14/14/14，一维残差 14/15/14。重新训练一维策略没有复现旧策略遮罩的稳定收益 |
| IMU/编码器融合 | [144 回合](../d1_sensor_estimation/README.md)；8 例从磁盘测量包独立回放 | 无延迟闭环 36/36 跑满；20 ms 测量延迟时 17/36 跌倒。理想接触开关和已知初始位姿仍是先验 |
| 连续地形、停车与转向 | [3 个新 sensor45 策略](../d1_continuous_policy/README.md)，共 393216 个训练样本；[完整 45 s 策略视频](../d1_continuous_policy/videos/seed19000_sensor45_45s.mp4) | 18 次 45/50 s 留出均完成，质量 0/18。高度误差比配对基线增加约 3.34 mm；道路未更换，不宣称新地形泛化 |

## 怎样相信这些记录

动作实验的[独立分析](../d1_action_ablation_analysis/audit.json)检查 489 个训练产物和 255 个留出产物，重算全部开发与留出指标。原始 Gaussian 样本的 Normal log probability 与 float64 独立计算最大相差 `2.8174e-6`，执行动作逐样本等于 `clip(raw_action)`。

连续任务有[另一次交付复核](../d1_continuous_policy/final_delivery_audit.md)。仅污染评测真值，传感器策略的观测及关节力矩逐位不变；奖励按预期改变。磁盘 checkpoint 的完整复演与原记录逐字节一致。两份视频均完整解码，画面保留模型 SHA 和失败判定。该次训练没有记录 Gaussian 均值与 log probability，不能把动作专项的概率审计说成也覆盖了连续训练。

参考动态的 123 个原始文件及派生产物已重新核对 SHA；残差对照的 153 个原始文件、247 个输入文件和更正后的表也全部吻合。传感器审计重新执行后，报告 SHA 不变。原始负结果仍留在各实验目录，没有因为质量不合格而删除。

Claude 参与了 Gaussian 概率与 tanh 均值导数的讨论。代码以本机 SB3 实现、手算及梯度测试核验；没有把模型建议当作实验事实。学习文档保留失败原因尚未证实的地方，不虚构实习中完成过的工作。

## 回归检查

全仓测试实际运行得到 **987 passed**，耗时 129.38 s，无失败或跳过。[JUnit 原始记录](tests.xml)随报告保存。15 条警告来自本机 Matplotlib/Pyparsing 依赖，包括 3D projection 导入警告；本轮二维图和 MuJoCo 视频都已分别检查，未用忽略警告替代渲染验证。

```bash
rtk env PYTHONPATH=src:.local-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m pytest -o addopts='' -q --junitxml=/tmp/my_d1_upgrade_tests.xml
```

本轮新增控制、训练和评测文件的 Ruff 检查通过，`git diff --check` 也通过。动作图最后仅调整文字边距并重算派生产物，随后补跑 33 项分析测试；没有修改训练源或重跑留出挑选结果。

相关九份入口说明共检查 145 个本地链接，没有缺失目标。验收后移除了本轮临时计划文件；实验决策和复现命令保留在各正式报告中。

## 中断与清理

机器重启和后续会话中断产生的部分动作训练已从仓库可恢复移至 `/tmp/d1-action-interrupted-archive-mrPCOE/d1_action_ablation_interrupted_20260908`。其中一次完整的 seed11000 双通道训练，与从头重跑的六个原始采样数组及最终网络参数逐位一致；它不增加独立种子数。参数 SHA 为 `8b9a45f59987573ccf6b4388e9ef919dd0c8bf2d6ab1c44cf55c23f8bba831f7`。

连续任务的短 smoke 与重复复演恢复位置见[清理记录](../d1_continuous_policy/cleanup.json)。这些临时目录位于 `/tmp`，系统清理后可能不再保留。正式训练、失败案例及协议没有移走。16 GB 内存不适合同时运行十二个 MuJoCo worker，后续默认同一时间只安排一组四环境训练。

本次修改尚未提交或推送 GitHub。真机参数、通信时序和硬件安全保护仍需另行验证。
