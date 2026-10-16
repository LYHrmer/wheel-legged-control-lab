# 源码身份与延迟日志完整性

2026-10-16 的工程检查修复了日志缺拍的判定，并给新实验增加独立源码清单入口。没有重新训练或运行历史物理试次，原始结果与冻结代码保持原样。

## 缺拍日志不能代表完整时域

`scripts/delay_metrics.py` 原先只要求时间严格递增。如果 CSV 只剩 `time_s=60` 一行，仍会返回 `completed=True` 和完整时域 RMSE。现在要求首行是 0.01 s，之后每行都与连续 0.01 s 控制时钟相符，容差 `1e-8` s。缺首拍、缺中间拍和只剩终点的输入会明确报错。

这修复的是汇总函数的输入检查。已有独立分析器也会检查时钟，历史结论没有因此改写。只读重算 `results/d1_delay_channel_study/development60s_final` 的全部 **24 个案例、70,784 行控制记录**，**216 项摘要值逐值不变**，仍是 10 个完整案例、14 个提前终止。浮点累加产生的合法时间误差保留兼容。

## 给新实验保存源码清单

旧 `capture_git_provenance()` 使用 Git 状态和 tracked diff，不能区分同一个未跟踪模块的两种内容。实际反例中，`src/controller.py` 从 `GAIN=1` 改成 `GAIN=999`，旧指纹不变。这个 helper 属于历史冻结输入，本次没有修改，也没有替换旧运行记录中的指纹。

新增 [capture_source_provenance.py](../scripts/capture_source_provenance.py) 可在新实验前后单独使用：

```bash
rtk proxy python3 -B scripts/capture_source_provenance.py --output /tmp/source-before.json
# 执行自己已获授权的新实验；不要更改该轮源码。
rtk proxy python3 -B scripts/capture_source_provenance.py --verify /tmp/source-before.json
```

第一个输出路径必须尚不存在，且位于源码目录之外。第二条退出码为 0 表示源码与 HEAD 相同；1 表示发现变化；2 表示输入或读取错误。工具本身不启动实验。

清单覆盖 `src/`、`scripts/`、`examples/` 中 Git 已跟踪和非 ignored 的 `.py/.pyi/.xml/.urdf/.toml`，以及根目录 `pyproject.toml`。每项记录路径、类型和内容 SHA；已跟踪但删除的文件记作 missing，符号链接只绑定链接文本，不读取目标。它会检测未跟踪源码的内容变化、删除和改名。

这是源码清单，模型、权重、二进制 mesh、依赖版本和实验数据仍需各自记录。它不扫描 `results/`、运行时归档或依赖目录，也不代替旧 `provenance.json`。采样前后会检查常见并发改写，但不承诺原子快照；应在源码稳定时使用。

定向测试覆盖旧指纹碰撞、CLI 前后核验、范围、缺失和改名、链接、拒绝覆盖以及缺拍输入。旧 77 项保护清单中的其余 76 项仍冻结；此前已解冻的 `experiments.py` 也未在本次修改。
