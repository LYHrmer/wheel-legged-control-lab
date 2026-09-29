# continuation14 最终只读审阅

2026-09-29，Astra。**本次有界诊断通过完整性门；预注册的广泛欠估 U 和强 critic clip 压力 C 同时成立。下一项单因素机制试验选定 actor/critic 分组裁剪，并保留同权重、同优化器状态、同新增训练预算的原全局裁剪对照。** 本结论不代表该改法已经有效。

审阅读取 `model_01/receipt.json`、`independent_readback_14_01.json` 的实际结果，没有重复模型加载、前向、backward、训练、测试或物理。独立读回文件明确自身 model_loads=0，复核的是保存值、梯度和输出 hash；其列出的 backward_calls=4/value_rows=1024 是被核验的原 worker 消耗，不是读回再执行。

执行绑定的 GO plan SHA256 为 `79353185642c5cace24254d3d6657514a1bce1a330b29d2b9adf6565234a6e0f`，sample SHA256 为 `37c538e31aeb56a9d11db45ce03b20227c301655b75da61ce587279a981e98d7`。worker 返回 passed，耗时 31.983836 s；一次 PPO.load、同一 ZIP 三次 torch.load、5 个 value batch / 1024 state row、4 次 backward，全部 attempted=returned。actor、optimizer、learn、train、save、额外 autograd.grad、clip 和物理均为 0。source postcheck 通过、cleanup_errors 为空。

policy hash 始终为 `c20a8841dcf2174c2db584edca951adaafa12688b026172669ef0e07ad955212`，optimizer hash 始终为 `bd4c898d33f4a121489f393dfe0f083737305f76fb139ddef74db2bb94878bf9`；加载、计算后、cleanup 三处一致。四批 actor 的 7 个参数梯度均为 None；critic 梯度和输出 bias 解析核验通过。

512 起点的当前 V 均值 48.89430，保存行为加 final bootstrap 的 Y 均值 212.93801；平均 residual 164.04372，512/512 residual 为正。总体 B=0.875603、E=0.872718，四个训练时间组 B 约 0.8751–0.8760，末组仍满足 U。没有落入预注册的前期集中或足量状态组集中门。effective-partial n=30、yaw-present n=27，低于单组决策门，不能由这两组推独立训练选择。

| 训练时间组 | weighted critic-only gradient L2 | 由 0.5 cap 推算的 critic-only clip factor |
|---|---:|---:|
| 0 | 1860.565535 | 0.0002687355 |
| 1 | 1853.617543 | 0.0002697428 |
| 2 | 1850.752106 | 0.0002701604 |
| 3 | 1843.416132 | 0.0002712356 |

该结果跨四个固定批次明显超过预注册 C 门（Gc≥5）。源码与参数分组表明 actor/critic 的可训练参数独立，共享展平层无参数。因此支持进一步检验**全局裁剪标量耦合**，不支持“共享表示上的梯度冲突”表述。

这些是当前 final、大小 128 的保存行为 64-step 代理目标梯度，不是历史 PPO/GAE 梯度；没有测历史 actor 梯度，也没有测此次 actor 梯度。0.00027 不是已观测的历史 actor 更新缩小倍数，Adam moments 还影响真实更新。当前 Y 也不是最终策略 on-policy 真值。近常量 V 与强代理偏差没有直接证明网络容量不足，不能据此同时更换网络、reward、target_kl 或学习率。

下一阶段因此只改变裁剪分组：原机制 control 对全 policy 参数 clip=.5；treatment 对 actor 与 critic 不相交参数各 clip=.5。两者均从同一旧 final policy 和 Adam 状态开始，以同新 seeds、同课程和同新增控制/optimizer 上界学习。必须保存本次真实 actor/critic preclip、postclip 梯度与 optimizer delta，并在新 seed 物理任务上比较完整安全、跟踪、成本和制动；若无收益或出现能力退化，则拒绝该机制，不以 loss 下降替代物理结果。

本阶段到此闭账。下一阶段预算独立登记，不借用剩余 diagnostic 额度，不重新跑此诊断，也不改旧 77 个冻结源与历史记录。
