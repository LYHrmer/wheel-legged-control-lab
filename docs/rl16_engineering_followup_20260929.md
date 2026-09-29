# 11-S 记录边界与后续工程修正（只读诊断）

2026-09-29。本文件是后续方案输入，不修改或重新解释已冻结的 11-S 运行，也不授权额外训练、评估或 GUI 仿真。

## 已保存的事实

- [training_segment_receipt.json](../results/d1_rl16_progress_20260929/evidence/rl11/training_run_01/training/training_segment_receipt.json) 记录 65,536 个训练控制、完整 64 个 PPO rollout，训练耗时 **1348.3807352290023 s**；[learning_receipt.json](../results/d1_rl16_progress_20260929/evidence/rl11/training_run_01/training/learning_receipt.json) 与 [final_checkpoint_manifest.json](../results/d1_rl16_progress_20260929/evidence/rl11/training_run_01/final_checkpoint_manifest.json) 保存已完成训练、final checkpoint 和严格 reload 的证据。它们证明一次训练完成，不证明 12 场 heldout 合格。
- [heldout_progress_01–09](../results/d1_rl16_progress_20260929/evidence/rl11/training_run_01/heldout_progress_09.json) 共九场完整、各自 `record_valid=true`。第十场 `rough_0p35/final_policy` 的 policy 预测记录为 1024 次；没有该场完整 `case_receipt.json`，不得列入 12 场汇总。
- [launcher_receipt.json](../results/d1_rl16_progress_20260929/evidence/rl11/training_run_01/launcher_receipt.json) 记录 host 的 1800 s watchdog 到期、子进程 exit -15、源哈希零差异、清理后无 orphan；[worker_receipt.json](../results/d1_rl16_progress_20260929/evidence/rl11/training_run_01/worker_receipt.json) 的 `execution_complete=false`。末端 C 已返回 404,800 native step，即 80,960 个五步区间；Python 控制 `attempted=80960`、`completed=80959`。这相差的一个区间发生在归档 flush 边界，不能被当成完整返回的 Python control。
- [worker_stdout.log](../results/d1_rl16_progress_20260929/evidence/rl11/training_run_01/worker_stdout.log) 明示 SIGTERM 的 `TerminationRequested` 在 `course_native_guard_08._write_gzip_rows` 的 JSON/gzip 写入期间发生。其后 `record_heldout_case(...).finally` 调用 `guard.finish_segment()`，再次按相同 `native_block_0000.jsonl.gz` 路径用 `xb` 打开，因上次已创建而抛 `FileExistsError`。后者遮蔽了原始 timeout。该 gzip 文件在异常中断后不可作为已提交证据使用。不能把这解释为物理数值失败。
- 运行入口 [run_short_rl_11.py](../results/d1_rl16_progress_20260929/source/rl11/run_short_rl_11.py) 在整个训练和 heldout 期间安装全局 `sys.setprofile(count_model_calls)`。它会收到全 Python 调用事件；这是**可能的开销来源**，但现有记录没有 profiler on/off 对照，不能量化性能损失或宣称其为主要瓶颈。全量 heldout 接触力采样、JSON 序列化和 gzip 同样可能贡献时间；具体占比未知。

## 建议的新版本，需另立有限合同

1. **保留这次真实边界。** 发布时明确“训练和 checkpoint 完成、九场 heldout 完整、总体 11-S 资格未完成、rough policy 文件部分写入”，原运行、失败文件和 host/worker 两份收据原样保存。GUI12 不得用 synthetic 11-M 模型、08 失败 checkpoint 或这份未完成的总体读回生成可启动 capability。
2. **先改变归档事务，再做后续物理。** 新版 block 写到同目录独占 `.partial` 文件；gzip close、文件 fsync 成功后，以不可覆盖的原子提交（如同文件系统 `link` 到最终名，随后 fsync 目录并清理临时名）发布最终文件，再增加 manifest 行。失败时标记 `partial/invalid` 与精确已返回 5T 计数，`finish_segment` 不重试同一 block、不覆盖首异常；只读地保存失败前缀。不要删除或修补旧 partial 来伪造完成。
3. **设软停止和闭账宽限。** 外层在硬 wall limit 前预留经审定的封存时段；SIGTERM handler 只锁存一次 stop 请求，控制线程在当前完整 tick 和原子 block 提交后停止发新物理调用，生成 partial receipt。若封存仍超时，host 只清理自己绑定的子进程并保留截断证据。信号回调不在 `json.dumps`/`gzip.close` 中直接抛异常；停止不退款、不重试。
4. **缩小模型调用审计热路径。** 新源码可为明确的 `learn/save/load/predict` 入口提供窄计数适配器或只在调用前后审计真实方法身份与返回，而非全程 `sys.setprofile`。须先在独立小预算比较次数、最终模型/动作字节、C/Python/native 帐与保存输出；再量测各阶段和 gzip block 的实际 wall/CPU 时间。现在不能给出加速倍数。
5. **评估续接必须另写资格定义。** 先用九场已完成结果判断缺失评估能否改变下一步决策；只有能改变时才提出所需场次。应严格加载本次 final checkpoint，独占新预算/进程，冻结原九场与新场来源，逐场初态位级配对，且在报告中区分两进程 C 帐。原合同的“训练和全部 heldout 同一 model/data 地址”不能跨进程冒称满足；如需跨进程组合，必须先由新合同重新定义并独立审阅其证据门。不重训、不重复九场完整评估，也不声称旧 1800 s 已通过。

## GUI12 当前源状态

`gui12/full_drive_controls_12.py`、`latest_frame_mailbox_12.py`、`async_course_renderer_12.py` 是未执行的 09 源候选复制；`run_world_upright_gui_12.py` 目前只有预检、owner runtime、输入 TTL/servo admission、copyData fence，**没有控制线程/main/完整入口，不可启动**。这些候选 AST 与 Ruff 静态通过，未运行任何 GUI、模型、engine 或测试。Root 所有的 launcher/capability 源亦仅候选。等待实际 11-S 最终独立读回与按案例资格，再完成和审查 GUI；RL 贡献比较须独立报告，不能用物理任务合格替代。
