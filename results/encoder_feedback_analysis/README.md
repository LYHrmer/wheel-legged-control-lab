# 编码器反馈记录的独立复算

[analysis.json](analysis.json)核对了完整 162 个控制例、486000 行记录、15 个源码快照成员与 228 个生成产物的 SHA。全部案例完成，没有触发速度保护；这不是控制质量达标的判定。

使用[独立脚本](audit_feedback.py)，只需 NumPy 和 Python 标准库，不导入原实验脚本、拟合器或 MuJoCo：

```bash
python results/encoder_feedback_analysis/audit_feedback.py \
  --directory results/encoder_feedback_position_only \
  --output results/my_encoder_feedback_audit.json
```

输出文件必须不存在且位于被审计目录之外。不要使用 `python -O`，脚本显式拒绝关闭断言。原始逐步数据的获取方式见[复现说明](../../docs/reproducibility.md)。

检查包括采样时钟、延迟队列、噪声重建、两种反馈递推、PI 与前馈力矩、指标和案例配对。拟合部分检查候选选择与已知静止的日志约定，不重新求解优化问题；也不复演隐藏物理动力学。

`paired_comparisons` 汇总混合了三种任务，不能单看其中较低 RMSE 的数量。较低的案例均来自 stress，同时伴随更多饱和；tracking/reversal 的两种位置估速反馈均劣于独立速度参考。完整分任务数值见[学习文档](../../docs/encoder_feedback.md)。

实验后新增的原记录 README 被明确列为不属于原始 manifest 的说明文件，没有倒填哈希。这里是独立分析目录，不改写原记录。
