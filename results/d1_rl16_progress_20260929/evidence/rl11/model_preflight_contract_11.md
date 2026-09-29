# 11-M 唯一零机器人学习接口资格

本合同只验证新BudgetSpec与world-upright严格final sidecar接口；不运行机器人、任何MuJoCo导入/模型/step/forward，不load/predict旧T或任何既有checkpoint，不重跑旧08 synthetic或已闭合pure批次。root实现/执行唯一新进程，Astra静审SOURCE GO后运行。

固定新output rl11/model_preflight_01，一次启动、120s、CPU线程1。新pure Gym假环境返回finite shape99 float32及shape16动作，明确synthetic并无物理意义；只用学习模块11。BudgetSpec(1024,1024,256,4)，显式PPO seed88612。恰一次learn(total_timesteps=1024)，1024个成功synthetic env.step、一个1024 rollout、一次PPO train、4 epochs、16次实测optimizer.step。可在learn完成后对假env额外直接尝试第1025步，必须拒绝且不得增加成功计数；不是第二次learn/采样。synthetic reset最多3次以允许至多2个真正TimeLimit截断及SB3末端auto-reset；使用真实post终局观测而非旧pre观测。不得为满足次数补采样。

证明actor mean、log_std、optimizer和全policy参数hash真实变化与有限，callback Gaussian→clip→effective与actual synthetic return计数一致；完整BudgetSpec反序列化与derived计数、hash和显式seed一致。学习模块数值超参沿08，仅预算/seed/新任务来源参数化。

实际只保存一个新的synthetic final checkpoint（写一次final_model.zip），调用save_final_and_verify内置的一次PPO.load(env=None,device=cpu)完成严格reload。不得随后为了重复验证再load同文件；load不能reset/step环境。最多32个固定synthetic probe observations，原模型和唯一reload模型各最多一个deterministic batch predict（模块内部既有double-check若需多于此必须执行前由Astra静读明确调整，不可运行后解释）。记录实际save/load/predict调用次数，而非只声称无额外采样。

sidecar必须标synthetic_only=true、new world-upright task/reward/reference schema与完整reference_definition、BudgetSpec1024/hash、新seed88612，来源绑定实际新源。此次模型永不获得机器人/GUI/heldout资格，不作为后续65536初始化。

在同一次进程做至多3个只读错误manifest拒绝：错/缺world reference、把budget改为65536、错seed；均应在PPO.load之前失败，不能生成额外模型/预测/文件。原正确manifest和checkpoint字节不得改写。记录每个拒绝的原因及load计数前后不变；单靠shape99/16不能放过任务身份错误。

运行前冻结新runner/learning11/BudgetSpec、world-reference常量源、解释器及选中Torch/SB3/NumPy/Gym依赖py/ELF；沿原cv2 optional禁用与engine import guard，报告actual provider/dependency版本。唯一启动预扣全部上限；失败真实封账，无自动retry/第二round，不挪用11-R/旧T物理预算。

只有全部通过，才支持后续真实新训练worker使用已审BudgetSpec/strict final接口；它不授权真实65536训练，也不证明新任务可学习。11-R坡道基线与其独立读回另计、另GO。
