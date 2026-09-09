# 第二版地形实验源码快照

`source.patch` 保存这轮训练/诊断所需的 10 个变更文件，相对于提交
`cbfdb2d336111739f572f5527acd4067132906a3`。它包含第一版地形基础和第二版控制环境，
以及训练、诊断与绘图入口；不修改模型资产，不包含训练结果。

SHA-256：`49db98c4a0e46aa5ced45a1e0a22c77aa3b5fd27166aa039fcf2ace37224ae8f`。
已在生成时通过 `git apply --reverse --check` 核对与工作树一致。
随后在独立临时 Git 索引上从 HEAD 正向应用补丁，恢复的 48 项源码 SHA 与两组奖励 A/B
及正式 v2 的运行记录全部相同；当前工作树和真实 Git 索引没有改变。
逐文件结果见 [restore_audit.json](restore_audit.json)。这项检查只验证源码可恢复，没有重跑训练。

在独立的干净 worktree 中使用，不能在当前已有修改的目录中重复应用：

```bash
git worktree add --detach /tmp/d1-terrain-v2-reproduction cbfdb2d336111739f572f5527acd4067132906a3
cd /tmp/d1-terrain-v2-reproduction
git apply /absolute/path/to/source.patch
```

安装项目和可选 `.[rl]` 依赖后，按各运行的 `protocol.json` 重跑。对照实验目录的
`provenance.json` 列出具体运行的源码与依赖版本；这些记录比当前分支文件更新日期更可靠。
历史 v1 实验已有独立 `source.patch`，不要用这份快照解释其旧模型。
