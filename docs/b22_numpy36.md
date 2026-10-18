# 36：B22轻量NumPy数值与冷启动资源门（准备）

所有者目标为“把这些优化点，在这台电脑能力之内的都完成”。本轮只验证独立数值后端和两个新进程的启动成本，不训练，不导入模拟器，不构造环境，不授予控制、部署或实机资格，不替换默认后端。旧v27、31、32、34、35源码与结果不变。

新增[一个driver兼worker](../scripts/export_b22_numpy36.py)、[纯测试](../tests/test_export_b22_numpy36.py)和本文。默认入口只打印PLAN_ONLY，不导入NumPy/Torch/SB3、不创建输出。源码提交后才可在根代理安排的独占Nice10单元内执行一次`--execute --output /home/lyh/wheel-legged-control-lab/results/b22_numpy36_20261018_01`；worker子入口必须有实际父PID、当前nonce、精确argv、release和前序COMPLETE，不能单独调用。已有目录拒绝，不自动重试。

固定B22为ZIP `7bd58b5d…76691`、metadata `af700d1f…72830`、完整policy state `59440de0…7aa48`。原严格加载器必须完成一次PPO load、三次torch.load和原32row批量probe的字节一致验证。另显式核实际ActorCriticPolicy、DiagGaussian、FlattenExtractor及其唯一Flatten子层、flat99 float32 Box、当前predict路径不添加额外归一化、无squash/SDE、精确六个float32 actor张量。`load(env=None).get_vec_normalize_env()==None`只描述当前对象，不能单凭它证明训练历史没有VecNormalize。复现输入是已准备好的99D观测，不替代raw sensor→99构造器，也不反推训练历史。模型为99→128Tanh→128Tanh→16均值网络；31376个float32参数共125504字节。导出保存六数组、原ZIP/metadata/policy/probe/source/version身份及原严格加载回执。

所有192条输入均在计算任何真实预测之前固定，不按输出挑选：

- 32条原B22保存probe（原顺序）。
- 31的flat_0p6_B、yaw_1p2_B、rough_0p35_B，以及32 gated/flat_1p6，各取已保存`observation`的tick `0,50,…1550`，共128条；完整NPZ身份预先固定，读原始观测，不加载模型/物理。
- 32条固定边界：零、全±5、两种交替符号、正反等距斜坡、交替最小正常float32值，以及24个预定坐标的±5稀疏向量。

输入为192×99 little-endian float32，共76032字节，SHA256 `b1468a1d91007cecb3d8c8a706fb1f44fb2857819eb96f57fd9c72c5cfeda834`。父进程以标准库读取固定NPY1数值头和指定行，不执行pickle，不导入NumPy或Torch。每行来源另存。工作负载固定batch1，先把网络均值clip到±1，再乘float32(0.5)；两个输出都独立要求绝对误差≤1e-6，不能因半增益而放宽原始clipped动作门。NumPy可能有浮点舍入差异，本轮不继承原加载器的字节一致承诺。

| 新进程/阶段 | 固定请求与行数 | 墙钟/CPU上限 |
|---|---|---|
| 唯一cold Torch | 原32row probe一次＋192次batch1：193 predict calls / 224 rows；1 PPO load / 3 torch.load | 60s wall；CPU soft30s/hard35s |
| 独立cold NumPy | 192次batch1 / 192 rows；零PPO load、零torch.load | 60s wall；CPU soft30s/hard35s |
| 标准库父进程 | 无NN、无额外benchmark | 总180s wall；父CPU soft60s/hard65s |

全轮没有第三次benchmark或未计时重复。Torch worker完成原严格load和共同192预测后，才写导出数组与输出；NumPy worker不再加载PPO，且会阻断Torch/SB3/Gym/模拟器的间接导入。双方分别报告共同192预测循环墙钟、cold至共同工作负载结束墙钟、整个进程墙钟、gross峰值RSS与CPU时间。内部cold时钟起点为driver首次导入time后，包含后续导入/准入/加载；父进程的process_wall另覆盖进程创建、退出与清理。Torch还有必须保留的原32row严格probe，因此不能当作完全对称的纯NN微基准。导出I/O在共同计时终点后；整个进程墙钟/RSS仍如实包括后续验证和I/O。不预先宣称节省多少时间或内存，也不能由单次冷启动比较外推实时控制收益。

启动有效RAM至少6GiB，运行保留4GiB；BLAS/Torch单线程，实际nice至少10，串行两个独立进程。子进程CPU/墙钟与自有后代清理有界，源/输入前后核对。失败即停止，实际总NN计数未知时为null，可用的子回执前缀单列，不补零。数值失败会保留已计算输出供只读核对；父进程用标准库重算保存输出的误差和clip后增益，不仅相信子报告的PASS或SHA。

本机准备阶段只生成源码并做合成矩阵/假回执/冷导入测试及只读固定输入组装；尚未执行真实PPO load、真实策略forward、导出或冷启动数值比较。测试中的合成六矩阵不是B22模型，也不计入未来真实NN预算。只有真实数值和资源门通过后，才讨论新的独立headless运行链；现有导入修复、ModelCounter、来源检查仍依赖Torch，不能仅替换predict就声称运行链已摆脱Torch。

准备验证：86项纯测试、Ruff通过；独立只读审查未见剩余阻断。测试包括同名伪Tanh/非Sequential图的拒绝、原始动作超1e-6但半增益误差较小仍拒绝、保存SHA同步变化后的错误数值拒绝，以及父进程真实SIGXCPU的回执与清理（没有子进程或实际CPU耗尽）。这些结果不代替真实权重的数值/资源测量。
