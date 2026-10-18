# NumPy 后端 39：独立修复来源闭包

39 只修复 38 实际暴露的依赖清单缺口。38 冷准入已中止，原输出、源码与 `INCOMPLETE / actual_counts=null / actual_nn_counts=null` 全部保留；不重试 38、不沿用其余额。39 使用独立的 `C39_B22_numpy_half_gain_v1`、新 source、release 和固定输出 `results/b22_numpy39_20261018_01`，默认 `PLAN_ONLY`，0 NN、0 环境、0 native。用户已授权的本机优化工作按串行流程推进，实际执行由根线程统一安排。

新入口仅一份 driver，另有测试、本文、包清单 helper 及其测试，共五份新受审文件。driver 逐 SHA 锁住 38 的四份生产模块、三份测试及文档；只对精确 contract/schema/路径、模块注入和新 release 事实作 AST 适配。每一处替换可逆恢复成原 AST，所有旧 64 位 SHA 常量必须完全不变。原控制、 prepared99、半增益 float32(.5)、PI/z、停止逻辑、native/NN 计数、监督清理、任务阈值及评分数学继续使用该已冻结代码。没有全局数字替换，也不复制八套源文件。

所有派生模块的真实 `__file__` 都是新 driver，运行回执明确为 39 schema/contract。worker 与 reader 使用同一个 `b22_numpy_backend39` 实例，thread-local counter 与类型不分裂；原 host 的实际 worker argv 没有 `--worker`，所以 driver 明确把 `--session ... --output ...` 分发到 worker，把 `--release ... --stage ...` 或 `--compare` 分发到保存结果 reader。若干函数名、`gain38` 字段、`bootstrap38_report.json` 和 `derived_sources38.json` 文件名保留为继承接口；它们位于新 39 输出中，内容为实际新 39 身份，并由新 source、session SHA 和 AST 证明绑定，不冒充旧 38 的执行证据。

`b22_runtime_packages39.py` 在不导入 MuJoCo 的情况下固定本机 MuJoCo 3.12.0 两个实际安装根：user site 为主根，`.local-deps` 为原环境中的 namespace 别名根。每根 125 份 `.py` / native 代码及 `METADATA/WHEEL/RECORD` 入清单，两根相对路径、字节数、SHA 必须相同，各文件再核 installed RECORD。保留原环境，不删除第二根改变依赖搜索。执行前重枚举精确文件集合和 SHA；运行时 `mujoco.__path__` 必须恰为这两个已登记根，实际文件来源只允许主根，拒绝第三根、包外模块和任何漂移。原 NumPy、Gymnasium 和 numpy.libs 的已绑定来源清单继续保留，不作动态补签。正常模块来源门仍逐项核实际 `__file__`，不会仅放行第一次报错的 `_enums`。

新预算为两项完整 B 场景 `flat_1p6_B(seed181202)` 和 `yaw_1p2_B(seed181203)`，各最多 1600 controls、8000 normal native、2 compiler native；总上限 3200 / 16000 / 4。每场仅原始 32 行 probe 一次和每 control 预测一次，总 3202 NumPy 调用、3264 行；Torch/PPO load、critic、训练均为 0。先进行独立的零 NN/零环境/零 native 冷 binding bootstrap，通过后才进入场景；每场自身的零步骤 bootstrap 也须通过。读取保存结果不新增 NN oracle 或物理，不以少跑的关门预测代替原预算。

启动门为有效可用 RAM 6 GiB，运行停止线 4 GiB、Nice≥10；总 wall 900 秒，bootstrap 60 秒，单 worker 240 秒，reader 100 秒，保留有界清理窗口。只在无活自有子进程、前一阶段清理完整且下一目录/请求/nonce 尚未建立时，允许 4–6 GiB 每 5 秒等待至多 300 秒，等待算总 wall；低于 4、未知、信号、超时或剩余 stage/readback 窗口不足立即停止。达到 6 后重核全部来源、release、剩余窗口及 RAM，再唯一 spawn；已开始阶段不重试，固定输出存在即拒绝覆盖。

两场需完整通过原任务/安全/native/controller/stop 门，且与冻结旧 35 同 B 场景严格同 seed、初态、reset、命令和计分窗口。SSE 与归一化 torque cost 各不超过旧值 1.001 倍（0.1% 兼容容差），原指标全部保留。旧 35 总资源仍为未知；旧两份闭合结果仅作比较锚。39 只支持这两场的 NumPy 后端兼容性结论，不承诺轨迹逐位相等，不授予完整任务矩阵、RL contribution 或实机资格，不自动替换模型。工程 `COMPLETE` 与上述兼容结果分别记录。

准备和验证命令（均不启动 bootstrap/NN/物理）：

```bash
rtk proxy python3 -B scripts/run_b22_numpy39.py
rtk proxy env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest --noconftest -o addopts= -q tests/test_b22_numpy39.py tests/test_b22_runtime_packages39.py
rtk proxy .local-deps/bin/ruff check scripts/run_b22_numpy39.py scripts/b22_runtime_packages39.py tests/test_b22_numpy39.py tests/test_b22_runtime_packages39.py
```

默认计划会验证旧八源与新包清单，不创建正式输出。实际运行前，五份 authored 必须已提交且干净，整个新旧来源闭包写入新 release；根线程在串行资源窗口中使用 `--execute` 与新的有界 systemd user unit，外层清理和源码身份检查继续生效。本文不记录或暗示尚未发生的真实 39 闭环结果。
