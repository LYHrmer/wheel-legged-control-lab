# continuation14 源码复核 GO 02

2026-09-29，Astra。**GO：允许 root 在冻结执行 plan/inputs 和独立 host 下按既定预算进行唯一一次诊断。** 这只是运行前源码门，不是诊断结果或科学门通过。旧 `source_review_14.md` 的 NO-GO 记录保留。

本次复核仅阅读源文件、JSON 回执和文件摘要，使用标准库解析 selection JSON；没有导入或运行任何项目模块，没有测试、模型加载、forward/backward、训练或物理。

复核 worker 为 `run_value14.py`，SHA256 `819b1d9c45d93b7f06ced4edb45ab85c7054d9c1b9a6b2e2aaa84a0f91979ce5`。

两项阻断问题已修正：

1. 512 个端点返回后独占写 `endpoint_values.npz`；每批结束独占写 `batch_q.json`、`batch_q_gradients.npz`、`batch_q_values.npz`。`report.completed_batches` 在循环前即绑定已完成批列表，后续故障可以封存已有批次。中途残留文件仍需按 worker/host 状态视为 partial；不得把部分写入解释为全四批通过。
2. `primary_error` 先初始化并独立锁存；finally 的 journal、close 和 receipt 错误分别进入 cleanup_errors；最后优先重新抛出 primary_error。未发生主异常但清理失败时仍失败退出。plan 本身的退出 SHA256 比较已补齐。

核心计算与调用限制没有扩大：一次 PPO.load、最多三次同 ZIP torch.load、一次 512-row endpoint 与四次 128-row start、4 backward、0 actor/optimizer/learn/train/save/physics。每批仍为 `.5*mean((V−detached_target)^2)`；头 bias 解析核验和 policy/optimizer 前后 hash 保持。组内 S、U/O/C/M/D 逻辑与方案一致。

只读检查 `sample_01/selection.json` 与 `preparation_host_receipt_01.json` 显示：四组候选各 245，各选 128，总共 512；准备回执退出码 0，模型与物理调用 0。selection 登记 131 个已核旧来源文件，samples SHA256 `37c538e31aeb56a9d11db45ce03b20227c301655b75da61ce587279a981e98d7`。prepare/math/test 的当前 SHA256 与 selection 内冻结值和首次审阅一致。yaw-present 共 27 个样本，小于 n=32 门，只作描述；terrain 总体数量满足该描述门，不代表地形因果证据。

root 报告 7 项纯测试通过，定向 Ruff E9/F63/F7/F82 通过；完整风格 lint 的 11 项未通过须保留原记录。它们未被本审阅重跑，不因风格要求改变已冻结的数据源。该风格结果不是模型调用或数学门失败。

启动集成条件沿用首次审阅：具体 GO plan 必须收录实际 model/metadata/samples/selection、全部执行文件及实际导入源的 hash；旧 closed manifest 与唯一 final 身份、77 个冻结源及依赖前后核验；120 s worker、150 s host、单列 5 s kill 回收期；新输出目录、一次性 reservation、无自动重试。root 负责这些具体输入与 host，不能把本源码审阅写成已执行的模型或 host 回执。

之后仅以保存数值、梯度、计数及 hash 作独立只读审阅；若实际发生故障则保留失败，不恢复余额、不重新加载。
