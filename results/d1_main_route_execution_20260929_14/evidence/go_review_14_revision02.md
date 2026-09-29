# continuation14 依赖范围修订 02：静态 GO

2026-09-29，Astra。**GO：revision02 可作为唯一一次诊断的执行 worker。** 此次只审阅原 worker 与 revision02 的 diff 和 SHA256，没有运行测试、模型、项目模块或物理。

原 `run_value14.py` SHA256 `819b1d9c45d93b7f06ced4edb45ab85c7054d9c1b9a6b2e2aaa84a0f91979ce5` 保持不变。新 `run_value14_revision02.py` SHA256 `9bf8320c30a76df70f7382a15d4dcd9971fd55a8ccad2a24c3534c3d8752c05d`。

diff 仅新增计算库映射核验函数，以及 import 后、PPO.load 前和 finally 中的两次调用。函数读取 `/proc/self/maps`，对 torch/nvidia 映射拒绝 deleted 状态，将真实路径 resolve 后与冻结 inputs 的真实路径集合比较；不在集合中则失败。前检查失败不会进入 load；后检查失败进入 cleanup_errors 并使诊断失败。该改动没有增加任何模型调用或放宽原有限值、状态 hash、调用闭合门。

root 报告两套 NVIDIA 文件 `samefile=False`，因此不能按 inode 直接去重。排除未使用的 `repo/.local-deps/nvidia/` 副本、保留实际 user-site Torch/NVIDIA 文件，并由运行时映射门检查实际使用路径，属于执行依赖范围收紧；不是样本、科学目标或计算预算调整。若运行中实际映射到被排除的副本，则本次应失败，不能临场补白名单或重新加载。

新 `plan_go_14_02.json` 应保留旧 plan，登记被排除的路径前缀及原因，并加入本 revision02 源文件和审阅；其他来源身份不变。root 报告双端输入 hash 总量预期由约 7.15 GB 降至约 4.35 GB，这只是预计读取量，不是已完成耗时证据。

其余 GO 条件沿用 `go_review_14.md`：512 起点、1024 value row、4 backward、一次 PPO.load、0 actor/optimizer/learn/save/physics；120 s worker、150 s host 和另记 5 s kill 回收期；唯一 reservation，无重试。本审阅不宣称尚未执行的 load 或诊断已经成功。
