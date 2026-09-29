# D1 主路线执行 13：RL 跟踪诊断与真实 GUI 短脚本验收

这是 11-S [紧凑进展包](../d1_rl16_progress_20260929/README.md)之后的增量工程记录。旧模型和原始运行保持冻结。阶段 13 的保存数据诊断发现四对 zero/policy 任务都在加速窗口改善 SSE，却都在保持窗口退化；flat 1.6 的共同轮速残差速度量级 0.02176 m/s，接近实测额外均偏速 0.02160 m/s。这是优先排查的控制通路，不是删除残差后的物理因果试验。冻结源码的 TimeLimit/bootstrap 静态审计未发现漏加或重复加的路径证据，历史 value/GAE 数值仍不可重建。

旧归档失败的纯测试已复现原异常被 `FileExistsError` 遮蔽。新 archive13 使用独占 `.partial`、载荷 close/fsync、不可覆盖硬链接及 manifest 最后提交；故障后不重试写块。归档 13 项、bridge 5 项、独立读回 5 项与 OS 进程回收 1 项绿色测试通过；首次 OS fixture 超时与旧归档红色复现均保留。软件测试执行 0 物理控制、0 机器人策略调用。

[真实配对读回](evidence/pair_readback_13_01.json)通过：冻结 final、seed 88813、平地 0.6 m/s 固定开发脚本，两臂各 600 controls / 3000 normal native + 2 compiler native。601 个状态、600 步控制及 3000 行原生记录逐位一致。GUI 有效 RTF **1.19042**、绘制 **12.49937 FPS**、poll 间隔 p95 **88.95671 ms**、快照年龄 p95 **140.19804 ms**，三张真实截图通过非空/摘要/序号核验。

共新增 **1200 controls / 6000 normal native + 4 compiler native**；load 2、各 32 样本 probe 共 2 批、正式 predict 1200，learn/save/optimizer 0。[阶段汇总](evidence/execution_summary_13.json)记录源不变、完整预算和无残留进程。headless/GUI 完整 wall 分别 82.10/197.46 s，含较重的依赖预检与后检；有效 RTF 不代表启动时间。77 份冻结文件全部匹配。

这只验收 **600 步开发脚本**，不重做原 1600 步资格，不声明自由键盘、硬件输入延迟、1.6 m/s GUI、多地形或 RL 收益。compact 模式未采样接触力，`qualified_for_default_GUI=false`。R 仍是仿真复位，物理自救、A/D 侧移与 Space 跳跃均未实现或验收。

![运动中原始帧](evidence/gui13_revision02/development600_01_gui/frame_drive.png)

本包收录新增源码、真实运行与独立读回，`package_manifest.json` 绑定公开文件字节：

| 目录 | 内容 |
|---|---|
| `source/archive13/` | `atomic_archive_13.py`、红色复现和绿色故障注入测试 |
| `source/gui13/`、`source/gui13_revision02/` | 原预检版本与实际执行版本；不覆盖原记录 |
| `source/sol_verify13/` | 独立通过审阅的 GUI13 验证器及纯测试，按实际来源署名 |
| `evidence/` | 诊断脚本/JSON、TimeLimit 静态审计、红/绿测试收据、审阅、冻结计划、两臂原生/控制/状态记录、截图及退出收据 |

原始 GB 级训练/持出记录、旧 `results/my_course_drive*`、私人试驾、Xauthority/cookies、Claude 私有 stdout/原始对话和未验收的 Claude verifier 草稿均不进入本包。模型仍由 11-S 包提供，阶段 13 不重新发布权重。API 返回的 Claude 模型标识为 `anthropic/claude-opus-5`，编码会话因 402 退出，其草稿未执行、未验收，不算已采用实现。

[执行记录与后续决策](../../docs/main_route_execution_20260929_13.md)给出证据边界。以下标准库检查只验证此包所有文件及相关文档的 SHA-256/字节数闭包，不导入源码、不加载模型、不执行物理：

```bash
rtk proxy python3 results/d1_main_route_execution_20260929_13/verify_package.py
```

文件完整性不等于实验资格，GUI 结果以独立收据为准。

诊断与执行源码保留原绝对路径及运行身份，依赖前一 11-S 包与原本地运行库；此增量包不是可独立启动的安装包。原 GO/输出目录已消费，不能用同一计划再次运行；后续输入验证和学习诊断需新的有限合同。
