# 50 交付 CI：隔离历史绝对路径的测试环境

提交 `a75908a35a4746a8ddc79d920574010c0d831009` 的 [CI 40050167784](https://github.com/LYHrmer/wheel-legged-control-lab/actions/runs/40050167784) 在 Python 3.10 / 3.12 均为 **2 failed、5359 passed、2 skipped**。两项失败均在模型准入的纯计划校验，尚未进入模型或物理调用：

- `test_exact_admission_plan_has_no_training_or_native_budget`
- `test_admission_plan_rejects_budget_drift`

生产计划固定了历史输出 `/home/lyh/wheel-legged-control-lab/results/b22_sigma50_model_admission_20261024_01`，而测试私有模块的 OUTPUT 根据当前 checkout 推导。GitHub runner 路径不同，所以完整计划相等检查正确拒绝；本机原路径下的测试无法暴露这个测试隔离问题。不是训练预算漂移或 Python 版本差异，也没有改成接受任意 checkout 的生产入口。

`tests/conftest.py` 不在旧 v27、正式训练 release 或真实模型准入的来源清单中。修复仅在这个未冻结的测试配置中，为上述两个函数包装其原 `module50` 工厂，将每次新导入的私有模块 OUTPUT 设为写死的历史路径。没有从待校验计划反向读取输出，没有修改 ROOT、PLAN、验证函数或全局生产模块；每项测试结束由 monkeypatch 恢复工厂。其他测试不使用这份适配。

新增 `test_b22_sigma50_admission_location.py` 直接加载生产模块，先确认历史路径对应的完整计划通过，再仅将 OUTPUT 改成临时外来路径，确认仍抛出原 `ValueError`。原预算漂移拒绝测试也继续执行，不 skip、不 xfail、不缩减断言。已执行的源码、原测试、预注册和 v27 字节全部保持原身份。

| 验证 | 结果 | 证据 |
|---|---|---|
| 修复前，复制最小真实来源到全新 checkout 后运行原 5 项测试 | **2 失败 / 3 通过**，精确复现 CI 两项异常 | `results/ci50_location_red_20261024_01/` 及对应监督目录 |
| 修复后，另一个全新 checkout，原 5 项加路径拒绝回归 | **6/6 通过** | `results/ci50_location_green_20261024_01/` 及对应监督目录 |
| Ruff 检查两个变更测试文件 | 通过 | `ruff check tests/conftest.py tests/test_b22_sigma50_admission_location.py` |

两次验证均为独立串行受限单元：Nice10、单核/线程、MemoryHigh2GiB/Max3GiB/swap0、连续10秒内存准入。修复前/后分别耗时 **11.074 / 11.070 秒**，峰值 **42,938,368 / 40,529,920 bytes**，有效可用内存最低 **8,872,161,280 / 8,019,718,144 bytes**；全部 memory events 为0，结束进程组为空。失败单元仍归档为 INCOMPLETE/exit1，未覆盖或续跑。这里只测试纯计划和拒绝逻辑，未增加模型训练、模型加载或物理步数。

这是新增的 1 项拒绝回归，连同此前执行前的 63 项，新验证覆盖共 64 项；原训练报告里的 63 仍指执行前验证，不回写历史报告。红/绿日志与监督副本见 [CI 修复证据](../reports/ci50_portability_20261024/)。完整 CI 必须以包含该修复的最终提交重新确认；失败运行原样保留，不能称为已通过。

本修复不改变 [50 实验结论](optimization50_delivery_20261024.md)：控制消融被拒绝，正式训练首臂不完整、其余五臂未运行，均值和极差未能判定，也没有恢复任何训练或变更来源绑定。
