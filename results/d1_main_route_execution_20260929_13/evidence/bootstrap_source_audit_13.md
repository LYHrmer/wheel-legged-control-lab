# 保存训练代码的 TimeLimit / bootstrap 静态审计

2026-09-29，root 仅阅读源码并计算文件摘要；0 引擎、0 策略加载、0 value 前向、0 backward。下列六个当前源码的摘要均与原 `rl11/training_run_01/session.json` 的冻结值一致。

| 文件 | SHA256 |
|---|---|
| rl11/rl16_learning_11.py | 837b1b7c814fcb174cc8f1d8ce0ad1bfcbf623e2cc4aba63ad36c3304f5ff673 |
| course_impl08/full_drive_env_08.py | d3895e09e7f3492e58dce3ffe8d572b0223f1639f0ee6c1581c673a7ed1ff787 |
| upright11/world_upright_course_11.py | 504cf329cabf889be5e2a6803b442114a6153437c87c992318bb12f8049efdb0 |
| stable_baselines3/common/on_policy_algorithm.py | ce1984ea8e8a2d969443a484ab159194b9d70ff92a7ef1bc25fcc183e391e35c |
| stable_baselines3/common/buffers.py | 21db27f000048615f5848facf18af9afc22e6e2a1c26148911c2cba542d1988d |
| stable_baselines3/common/vec_env/dummy_vec_env.py | e9086bccdee89800a03e32fad09b2baa4195bb72ada97a12f381afe211c7c32b |

`full_drive_env_08.py:334–385` 只在未 terminated 的 horizon 到达时设置 truncated。结束观测来自当前真实 post-step 状态、刚消费的命令与已提交控制状态，不再 prepare 下一命令。`world_upright_course_11.py:96–117` 保持这一只生成观测的语义。`dummy_vec_env.py:59–72` 在 reset 之前保存该返回观测为 `terminal_observation`，并将 `TimeLimit.truncated` 规范为 truncated and not terminated。

`on_policy_algorithm.py:213–262` 先调用 callback；随后只对 done、存在 terminal observation 且 TimeLimit truncated 的转移，将 `gamma * V(terminal_obs)` 加入 reward 一次，再写 rollout buffer。`rl16_learning_11.py:528–560` 的 AuditedPPO 仅围绕 train 做计数，不替换 collect_rollouts。其 callback 在 `:768–807` 保存的是明确命名的 `reward_before_bootstrap`，因此先前零尾折扣和不能解释为实际训练 target。

`buffers.py:419–439` 对 episode 边界通过 next_non_terminal=0 阻断跨 episode 的值项及 GAE 传播。普通 rollout 边界未 done 时使用该 rollout 末状态的 value；done 时该项被门控，timeout 的终态 value 已在 reward 中。按这条冻结代码路径，没有看到 timeout value 漏加或重复加的源码证据。

这是代码路径审计，不是对历史每次 value 返回值的重放或独立核验。历史逐转移 value、终态 bootstrap 数值和 GAE 未保存，不能重建精确历史 value loss。原训练数据有 65 个 timeout，0 termination；这不单独证明全部历史运行时 wrapper 行为。下一阶段若诊断 critic，仍需另立模型前向/backward 的有界合同。
