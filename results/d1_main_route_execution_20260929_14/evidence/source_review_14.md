# continuation14 运行前静态审阅 01

2026-09-29，Astra。**当前结论：NO-GO，待两项归档问题修正；计算、采样和模型调用预算没有发现阻断项。** 此审阅没有运行任何 Python 模块、测试、模型 load/forward/backward、训练或物理。root 报告此前 7 项纯测试通过（0.040 s），本审阅不将它冒充为自己重跑的结果。

审阅源 SHA256：

| 文件 | SHA256 |
|---|---|
| run_value14.py | 83ec6ea92daf46a6d9a852ec3122e3cf568dcda86ca33acae8e0ce3e935b9885 |
| prepare_value14_data.py | 2fe15e76f199d79c63d3b9d1dabd8dfdfcd0d09f42c2ed7a8e0d200f8d077341 |
| value14_math.py | c080e4a8ba400250e0fcd97d7d872da68e3bc1ba991d662b4f2ec98f44c1a8b7 |
| test_value14_pure.py | 0abe1237295ab849704575f88995625057f1f08568197e0b8ccfd5838929c568 |

## 必须在首次真实加载前修正

1. **中断丢失已完成批次的诊断数据。** run_value14.py 在约 273–306 行把梯度/统计加入局部 `batches`，但到全部四批成功之后的约 400 行才写入 `report`，全部 value 数值也直到约 346 行才落盘。若第二至第四批失败或软超时，journal 保留消耗计数，却没有先前已完成批次的梯度和值。合同 §8 要封存已完成批次，且真实模型调用不可重试。最小修正是在每批成功后独占提交该批 JSON 与数值，或及时把完整批次及已完成数值挂入 finally 可以封存的 report；失败批次不得伪装成完成。端点已完成时也应留下可识别的端点阶段数据。无需增加模型调用。
2. **清理写盘错误仍可遮蔽首异常。** 约 426–458 行先捕获/重新抛出主异常，finally 中 `record`、`journal.close`、`write_json` 没有独立的异常保护；其中任一个异常可以成为最终抛出的异常，覆盖原始失败。这与合同和 continuation13 已修过的归档问题相同。最小修正为主异常独立锁存，清理步骤各自 best-effort、将次生错误另列；封存失败不能将已知第一异常换成次生 I/O 异常。若新 JSON 写不成，至少保留 stderr/既有 journal 中的第一异常。成功路径的封存失败仍须标为失败退出。

这两项为源级修正，不需要修改抽样方法，不需要真实模型测试或扩预算。修正后应保留本审阅记录，以新审阅文件记录复核结果。

## 已通过的静态核对

- **采样及边界。** 64 个块与 65,536 行数量、block identity、索引连续性、episode/tick/done 关系均有检查。每 episode 64 格点、同 quartile 与同 episode 的 t+64 endpoint、64 rewards 内不 done、确定 SHA256 排名、各组 128、排序后不重叠符合合同。端点确为保存的下一时刻 input_observation99，没有使用 timeout 后 reset observation。缺少候选直接失败，不改样本。
- **数值目标。** reward 原始 float64；G64 从 64 rewards 与 gamma 权重积算；Y=G64+gamma^64*V_end；delta=Y−V_start。端点在 no_grad 中生成并转 NumPy，loss target 无梯度；起点保留 graph、只走一次 backward。`.5*MSE` 的输出 bias 梯度期望为 mean(V−target_float32)，符号和 reduction 正确。
- **组内尺度。** 总体、每个时间批次、每个状态组均各自按该组 G64 算 S，分母 `(1−gamma^64)*S` 正确；U/O/C/M/D 的阈值逻辑与静态方案一致。n<32 标志不支持单组决策。EV 为当前代理目标 EV，近常数目标返回 null。
- **前向及 backward 预算。** 模型计算源只有一次 512-row endpoint `predict_values` 与四次 128-row starts；没有重复预览起点。API 计数和 value output-head hook 交叉计 row/batch，caps 为 1024/5；loss.backward 经 torch.autograd.backward 封装限制 4 次。actor `_predict`、forward、evaluate_actions 和 actor MLP/head 拒绝路径均有拦截。所读 SB3 load/_setup_model/policy 构造路径只构建和初始化 MLP/optimizer，不在加载时对样本前向；随后 ActorCriticPolicy 类型与层结构核验限制了本冻结源的实际路径。
- **加载次数。** 源只有一个 PPO.load 调用。torch.load 最多三次是同一 ZIP 内 policy/optimizer/变量成员反序列化，不是三次 PPO 模型加载。旧带 32-state actor probe 的 loader 没有使用，因此本次不冒称复验行动探针。
- **禁止更新。** PPO learn/train、optimizer step、save、autograd.grad、clip_grad_norm_ 均有拒绝入口。policy.zero_grad 只清梯度；set_training_mode(False) 只设 evaluation 行为，不是 PPO.train。实际梯度裁剪没有发生，clip factor 为由范数推算的标量。
- **参数和优化器 hash。** 调用旧 `hash_state`，其递归编码覆盖 tensor dtype/shape/bytes、字典、参数组及 optimizer moments，policy.state_dict 包含 buffers；梯度和 Python 方法封装不属于该状态。初值同时匹配旧 metadata final policy/optimizer hash，正常与 finally 路径均检查末值。分组 6 个 critic tensor、7 个 actor tensor、无可训练 FlattenExtractor 及参数 identity 交集为零；critic-only backward 要求 actor 全 None，并检查每个 critic 梯度有限和 bias 解析关系。
- **失败关闭。** 任何 finite/身份/调用闭合/参数变化错误均不会给科学门通过；模型失败不重试。source postcheck 在 finally 执行；硬 kill 无法承诺 Python finally 执行，需由 host 回执明确 partial/缺失状态。

## root 在 GO/启动文件冻结时需确认的集成条件

本次尚未看到最终具体 plan JSON、sampleprep 回执或 host 实现，不能将这些描述为已通过审阅。root 应在首次调用前确认：plan 的 model/metadata/samples 实际路径及 worker/new math/loader imports 都确实列在 inputs；samples/selection 与原 closed manifest 来源吻合；77 个源和已选依赖 hash 同集前后核验；worker 120 s、host 150 s 并单列 5 s kill grace；纯数据 180 s 上界；输出目录和 reservation 唯一且不可覆盖。host 的额外 kill grace 是进程回收上界，不是额外计算预算。

worker 最終结果中的 attempted/returned 差可得到未返回调用数，报告时应明确列 failed/incomplete，而不把未返回判为完成。独立结果审阅只读保存的数值和计数，不再执行模型。若中断后只留下部分结果，则 U/O/C 的全阶段门保持未判定。
