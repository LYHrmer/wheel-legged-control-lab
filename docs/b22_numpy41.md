# 41：缺失 yaw 场景的独立 NumPy 验证

39 的 flat 场景已实际闭合，随后保存 reader 因沿用 Torch/Nvidia map 必须非空的旧前提而停止，yaw 未启动。独立 40 只读取既有数据，已核完 flat 的完整来源、native/controller/安全/停止与计分；40 新 NN、环境和物理均为 0。39 的父记录仍为 `INCOMPLETE`、实际物理/NN 总数 `null`，不补写成功、不重跑 flat。41 是新来源、新预算、新输出的一次 yaw-only 尝试，不续接 39 的 attempt 或余额。

41 钉住原 39 五份源、40 三份源及其成功 report/inputs SHA。复用的旧 flat 保留执行身份 `C39_B22_numpy_half_gain_v1`，验收身份仍为 `C40_B22_saved_numpy39_flat_readback_v1`。40 的输入清单、旧 39 全部输出、原输入闭包与 flat 各工件都重核字节身份，既不能只信任任意 `COMPLETE` 字段，也不筛选 task_passed 阳性结果。报告与代码均不把旧 39 的 `null` 总计数改成已知支出。

新 41 使用 `C41_B22_numpy_yaw_completion_v1` 和独占输出 `results/b22_numpy41_yaw_20261018_01`，仅进行新的零 NN/零环境/零 native 冷 bootstrap，以及缺失的 `yaw_1p2_B(seed181203)`。最多新 1600 controls / 8000 normal native / 2 compiler native；NumPy 仅一次原 32 行 probe 加每拍预测，最多 1601 调用 / 1632 行。Torch/PPO load、critic、训练均为 0，不另做读回 NN oracle。

执行源通过精确的身份/路径 AST seams 派生冻结 39 工厂，再复用其逆 AST 检查的 38 数值核心；所有旧 SHA 常量完全不变。仅缩小新场景和预算，替换实际 worker/reader 源身份，新增旧 flat 的固定证据桥。worker 与 reader 共用一个 `b22_numpy_backend41` 实例，实际 `__file__` 指向 41 driver。40 已审适配器只把原两份 Torch/Nvidia map 文件要求改为恰好 `{'paths':[]}`，所有其他来源、MuJoCo 双根包、GOT、原32 probe、计数、controller/PI/z、native、归档和评分门保持原数学。空 map 的含义限于原函数收集的 Torch/Nvidia 清单，不能解释为进程没有加载其他动态库。

顺序固定为新 bootstrap → yaw worker → 保存读回 → 合并比较。旧 flat 不进入本次 physical stages，不能借 reader 路径重新执行模型或环境。继承的函数名、`gain38` 字段和 `bootstrap38_report.json` 等内部文件名保留接口，但位于新 41 目录，内容为新 41 schema/contract，由新 source、父 nonce、session SHA 与派生证明绑定。

合并使用原两 profile 的纯评分逻辑与固定旧 35 同 profile B 参考，必须同 seed、初态/reset、命令和完整控制窗口。两项完整任务、安全/native/controller/stop 门须全部通过，且 SSE 与归一化 torque cost 各≤旧值×1.001。合法短终止按实际 controls/native/NN 数记录，工程可完成读回，但兼容性不通过；不能伪装成1600拍。比较输出分别列新 41 actual、复用的已验证 flat case counts，以及二者的分析集合计数（完整时为3200/16000/+4）；分析集合不是历史总资源支出。只支持这两项 NumPy 后端兼容结论，不授予 full-matrix、RL contribution、ROS或实机资格，不自动替换策略。

资源保持启动6 GiB、运行4 GiB、Nice≥10、CPU单线程、串行，总900秒（包含来源和旧证据校验）。因固定40全来源重核实测约15秒，bootstrap上限120秒、worker360秒、reader及比较各240秒；GO请求的soft/close/hard同步为300/355/360秒，仅这三个时限字面量另有逆AST证明。父时限仍保留清理窗口，阶段准入必须足够容纳worker与reader的合计600秒，否则直接停止；子上限并非可相加超出900秒的新总预算。物理与NN预算完全不变。

一次validate或compare调用内，plan、closure与比较共享一次固定40字节快照，避免重复全哈希；嵌套调用返回副本，最外层正常或异常退出均清空。下一次校验、下一阶段和另一进程重新读回，不跨阶段缓存或略过来源漂移检查。阶段间无活自有子进程且清理完整、下一请求尚未创建时，4～6 GiB每5秒等至多300秒，等待计总wall；低于4、未知、信号或时限不足立即停止。已spawn阶段不重试，输出存在拒覆盖。外部任务占用内存时等根线程安排资源窗口；不降低资源门。

默认只显示静态 `PLAN_ONLY`，不创建正式输出或启动模型/物理：

```bash
rtk proxy python3 -B scripts/run_b22_numpy41.py
rtk proxy env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest --noconftest -o addopts= -q tests/test_b22_numpy41.py
rtk proxy .local-deps/bin/ruff check scripts/run_b22_numpy41.py tests/test_b22_numpy41.py
```

实际运行需三份新 authored 已提交且干净、40 固定证据仍有效、内存达标，再由根线程用独立有界 user unit 执行 `--execute`。原 host 的真实 worker argv 无额外 `--worker`，driver 按 `--session` 分发 worker，按 `--release --stage/--compare` 分发保存 reader；无法用孤立 CLI 跳过新父 journal、source、nonce 和固定路径检查。本文是执行前范围，不是尚未发生的 41 成功回执。
