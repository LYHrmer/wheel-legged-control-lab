# 多地形开发基线

本页的 CSV 和源码归档随 Release 提供，见[下载与复算](../../docs/reproducibility.md)。

这批数据用于确认环境和 CPU 预算，不是课程训练的留出成绩。

```bash
python scripts/run_d1_terrain_curriculum.py --mode baseline \
  --episode-seconds 4 --evaluation-split development \
  --output results/my_terrain_development_baseline
```

16 个固定 case，每个 4 s，使用零残差 LQR＋VMC。原始结果在 metrics.csv（`metrics.csv`），
地形与命令在 [protocol.json](protocol.json)，各控制步记录在 `evaluation/zero_residual/`。
[summary.json](summary.json)确认运行期间源码未变，文件 SHA 见 [manifest.json](manifest.json)。
本说明为运行后添加，不在原始 manifest 覆盖范围内。

全部回合到达时间上限，但跟踪并不理想。目标 0.35 m/s 的平地 case 前进 0.883 m，
命令积分为 1.241 m。`development_08_bumps`（10 mm 振幅、0.8 m 波长）最终 x 为
−0.0649 m。+2° 上坡、0.25 m/s 的 `development_14_ramp` 只前进 0.0379 m，
命令积分为 0.8863 m。存活数字不能替代这些误差。

16 回合实际评测部分每例约 1.2–1.6 s，含环境创建；这只是本机开发运行的时间记录，
没有隔离系统负载，不作为硬件性能基准。随后冻结了每类 PPO、每种子 32768 步的有限预算，
没有据开发结果调整 LQR 增益或奖励权重。正式比较另见
[课程训练结果](../d1_terrain_curriculum/README.md)。
