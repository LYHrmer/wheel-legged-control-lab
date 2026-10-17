# 连续任务 v28：只比较完整、同一时域的配对结果

新增入口 [compare_continuous_results_v28.py](../scripts/compare_continuous_results_v28.py) 修复了旧连续任务分析中的一种比较错误：45 s 完整零残差基线与 1 s 提前失败策略，仍可能得到负的 RMSE 差，并进入训练种子的平均差。短回合只经历任务前段，这个差不能解释成完整任务改善。

新版保留失败或缺失的配对位置、完成标记与双方实际时长。只有双方都完成协议指定时域，才输出六项 RMSE 差；其余为 `INCOMPARABLE`，JSON 中 `deltas=null`、CSV 中对应差值留空。真实零改善仍输出数值 `0`。原始六项指标保留在输入审计 CSV 中，其身份写入输出报告。

这针对旧[连续传感器 PPO 任务](continuous_task.md)，与 B22 v27 启动器互不依赖，不改变旧分析源码、原始结果或 v27 清单。本轮没有训练、模型加载、GUI、物理积分或新的控制资格评测。

## 使用已经审计过的结果

入口只用 Python 标准库，可以在关闭第三方包加载的环境运行：

```bash
rtk proxy /usr/bin/python3 -S -B scripts/compare_continuous_results_v28.py \
  --run results/d1_continuous_policy/sensor45_v1 \
  --audit results/d1_continuous_policy/analysis_sensor45_v1 \
  --output results/my_continuous_pairs_v28
```

输出目录必须是新的，且不能位于输入实验或审计目录内。若本机尚未恢复对应历史审计结果，先按[复现说明](reproducibility.md)取回；入口不自动下载或重新运行实验。

输出只有两份文件：

- `paired_cases.csv`：每个预定策略/零基线配对的身份、状态、原因、双方完成标记和时长；有效配对才写六项差值。
- `comparison.json`：完整配对、逐训练种子的留出组、缺失目录标签、输入 SHA-256 与字节数。该文件在 CSV 写完后生成。

退出码 `0` 表示全部预定配对可比较，`1` 表示已生成报告但存在 `INCOMPARABLE` 配对，`2` 表示身份、格式或输出路径错误。

## 输入身份与比较规则

入口读取五个小文件，核对以下关联：

1. 原实验 `manifest.json` 中的 SHA 与 `protocol.json` 一致。
2. 旧审计 `manifest.json` 中的 SHA 与 `metrics_recomputed.csv`、`independent_audit.json` 一致。
3. `independent_audit.json` 的实验清单 SHA 与本次原实验清单一致，且该审计声明已核对完整案例清单。
4. CSV 每行的目录标签、模式、训练种子、预算、阶段和条件标签符合该协议；重复或未知案例拒绝。

它使用已经审计产出的指标，不重读原始 NPZ、轨迹或模型，也不重新证明整个原始实验目录今天仍未变化。输入身份链用于防止拿错实验、协议或指标表；它不是历史真实性认证。

预定矩阵由该版本的固定规则恢复：首个开发 checkpoint 看第一个开发条件，中间 checkpoint 看第二个，最终 checkpoint 看全部开发条件；留出条件只使用最终 checkpoint。每个策略配对必须有对应零基线，双方 `completed=True`，实际时长与协议时长的差不超过 `1e-8 s`。这个容差只处理浮点时钟误差，不改变控制或质量阈值。

缺失的策略或零基线会保留为带原因的配对位置。每个训练种子的留出组只有在全部预定条件均可比较时，才输出 `full_horizon_mean_deltas`。缺一例、失败一例或时长不符，都置为 `null`，不以剩余成功案例的平均数替代。开发 checkpoint 和控制步数不增加独立训练种子数；入口不计算置信区间或跨种子汇总改善。

## 真实旧结果保持逐值一致

本机已读取五个输入，共 **54,076 字节**：42 行运行指标组成 **33 个配对，其中 18 个是留出配对**。训练种子 `19000`、`20000`、`21000` 的留出组各有 6 个可比较条件；无缺失条件、无无效配对。

新版与旧表逐项核对了 198 个配对差值和 18 个训练种子组均值，最大绝对差为 **0**。因此这项修复没有改写已有控制结论；它改变的是失败、不等时域和缺失数据的后续比较行为。

一次本机 `python3 -S -B` 调用观测到耗时 `0.04 s`、最大 RSS `16,128 KiB`。这是该次小文件比较的资源记录，不是完整原始数据审计的性能，也不是内存独占保证。

证据：

- [真实数据逐值对比与输入身份](../reports/continuous_comparison_v28_20261016.json)
- [新版完整比较报告](../reports/continuous_v28_20261016/comparison.json)
- [新版配对 CSV](../reports/continuous_v28_20261016/paired_cases.csv)
- [旧短失败反例与新版 `INCOMPARABLE` 对照](../reports/continuous_comparison_v28_short_failure_20261016.json)

## 测试

本机 **23 项测试通过，Ruff 通过**。覆盖短失败旧行为的直接复现、完成标记与时长分别失效、缺策略或基线、真实零差值、身份链替换、重复/非有限指标、只用标准库运行，以及真实 33 对的逐值保持。历史结果包未安装时，仅最后一项真实归档回归会跳过；其余 22 项使用很小的合成输入，不需要结果包、模型或仿真。

```bash
rtk proxy env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_compare_continuous_results_v28.py

rtk proxy .local-deps/bin/ruff check scripts/compare_continuous_results_v28.py \
  tests/test_compare_continuous_results_v28.py
```
