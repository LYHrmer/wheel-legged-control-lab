# 14：冻结 final 在保存行为窗口上的有界价值诊断

本阶段已完成一次预注册的纯保存数据与模型诊断。512 个训练起点按控制索引四等分，每组固定 128 个；每个起点使用连续 64 条原环境 reward 和同 episode 的第 65 行输入观测作为 bootstrap 端点。四组各有 245 个合格候选，选择由冻结 SHA256 排序确定。`sample_01/selection.json` 记录每个起点、端点、奖励与观测摘要；`samples.npz` 为实际模型输入。准备阶段 0 模型、0 物理。

模型阶段只加载旧 final 一次。实际完成 5 批 value 前向、合计 1024 个状态行，和四次 critic backward；0 actor 前向、0 optimizer、0 learn/train/save、0 物理。四批 actor 的七个参数梯度均为 None，critic 六个参数梯度有限，输出 bias 梯度与解析式吻合。旧 policy SHA256 `c20a8841dcf2174c2db584edca951adaafa12688b026172669ef0e07ad955212`、Adam SHA256 `bd4c898d33f4a121489f393dfe0f083737305f76fb139ddef74db2bb94878bf9` 在加载、运算后及清理时相同。worker 和独立读回均通过；host 退出码 0，来源后检通过、cleanup 为空。

保存行为 64 步 bootstrap 目标 `Y=Σₖ(.99)^k r[t+k]+(.99)^64 V_final(obs[t+64])` 的均值为 212.93801，当前起点值均值 48.89430；残差 `Y−V` 均值 164.04372，512/512 为正。奖励定标后的总体 B=0.875603、E=0.872718；四个时间组 B=0.8751–0.8760，末组仍满足预注册的广泛欠估门 U。未触发前期或足量状态组集中门。effective-partial 仅 30 个，yaw-present 仅 27 个，低于 32 的独立决策门。

| 训练时间组 | critic-only 梯度 L2 | 假设单独按全局 0.5 cap 裁剪的系数 |
|---|---:|---:|
| 0 | 1860.565535 | 0.0002687355 |
| 1 | 1853.617543 | 0.0002697428 |
| 2 | 1850.752106 | 0.0002701604 |
| 3 | 1843.416132 | 0.0002712356 |

四批均超过预注册强 critic 裁剪压力门 C 的 Gc≥5，且 actor/critic 可训练参数分开。由此选择下一项**单因素**机制比较：相同旧 final 权重及 Adam 状态、相同新种子与新增训练预算下，比较原全局 clip=.5 与 actor/critic 分组各 clip=.5。第 15 阶段合同另外授权最多 32,768 个新增训练 controls，以及有条件执行的评估；整个新阶段上限为 63,368 controls。该比较正在单独准备，**此处没有新训练或收益结果**。

本目标是旧行为奖励加当前 final 端点 value 的代理，既非历史 PPO GAE/TD(λ)，也非 final on-policy 真值。四批梯度来自 128-row 代理损失，未测历史 actor 或 critic 梯度，也未测本次 actor 梯度；0.00027 不是历史 actor 更新系数，Adam 还影响真实参数步长。因此 U+C 支持试验全局裁剪耦合，不能证明改法有效、网络容量不足或已找到旧跟踪退化的唯一原因。后续必须依据真实训练梯度与新 seed 的安全、保持跟踪、成本、制动结果决定，不以 loss 代理替代物理证据。

公开包证据见 `evidence/sample_01/selection.json`、`evidence/model_01/receipt.json`、`evidence/independent_readback_14_01.json`、`evidence/model_host_receipt_14.json` 及 `evidence/final_review_14.md`。数据准备 7 项合成纯测试通过；独立读回 5 项合成纯测试由 root 运行通过（0.004 s）。第 13 阶段 CI run 36585578717 已由 root 核实完成且成功。旧 77 个冻结源与历史记录未改，本诊断不重跑。
