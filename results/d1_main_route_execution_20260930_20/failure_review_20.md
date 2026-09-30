# C20 失败审阅与下一合同候选

审阅者：gpt-6-astra，ultra，`/root/astra_mainplan19`。审阅只读保存资料与源码，未执行模型、物理或测试。root 执行了本次训练、独立读回及保存数据诊断。

**结论：C20 以课程几何资格失败结案；课程覆盖能否改善 RL 的假设仍未定。** A1 完成且独立读回有效；B1 在固定课程重置预检处中止，没有合格 final，不能进入成对开发、续段或最终保留评估。C18 原有能力与“RL 贡献尚未成立”的结论不因此改变。

## 已成立的证据

| 项目 | A1 | B1 |
|---|---:|---:|
| 实际完成 controls | 32768 | 18000 |
| normal native / compiler native | 163840 / 2 | 90000 / 2 |
| 完成 rollout / train | 32 / 32 | 17 / 17 |
| 实际进入 epochs | 101 | 57 |
| evaluated batches / optimizer | 373 / 351 | 213 / 202 |
| 正常 KL early-stop / hard-stop | 22 / 0 | 11 / 0 |
| 最大实际 batch KL | 0.056403350085020065 | 0.062323182821273804 |
| 证据状态 | 完整训练与覆盖独立通过 | 部分封存，无合格 final |

合计实际 **50768 controls、253840 normal native、4 compiler native、553 optimizer steps、200912 actor rows、200882 critic rows**。B 的 14768 个未执行 controls 不退回、不转为补步；已消费的失败运行不能在同一保留目录重试。后续阶段未执行，不能把它们的候选预算当作已取得结果。

A1 的每个 postwarmup 槽均有 3 个完整、有效的 episode；其独立 reader 确认 archive、learning、coverage、training 有效。唯一 final ZIP SHA256 为 `1a630229fe9b44abe75abe4e47c98be50a55a043f0163f20f57b0b97640904e5`。这只证明训练与归档有效；尚无 A1 的正式 heldout 任务或 RL 收益结论。

A/B 首 1024 条记录的 64 项完整初态、reset、numeric/Gaussian 配对均通过。配对成功不能补足 B 的后半段。B 实际完成第 18000 个物理控制后，`DummyVecEnv` 在自动 reset 时抛异常，该 terminal transition 未返回 callback：PPO `num_timesteps`、已接收 transitions、Gaussian 行数均为 **17999**；full/numeric 为 **18000**，pending Gaussian control index 为 **17999**。保留这个差异，不推造缺失记录。17 个完整 rollout 共 17408 条；其后已接收的 591 条未形成下一次完整更新。

root 核对了 B 的 323 份原子 payload 身份。B 未运行完整独立 training reader；原子文件匹配与原生前缀记录有效不等于完整训练资格。失败 ZIP `6a602ad7ca98a8890f6c238042521b2021aeab227c424b984b40423f4c2e25ee` 仅保留为失败证据，禁止用于正式 heldout 或恢复补步。

## 直接原因与审阅遗漏

固定 source episode 18 是 B 的 cycle1、slot2，vx=1.0，原 yaw 为 `[425,575):−.25`、`[575,725):+.25`。前 18 个 episode 已关闭；source18 保存了 schedule、reset metadata、实际 compiled reset geometry 与 nominal path，但尚未执行任何 source18 控制，也没有成功的 footprint precheck 回执。

`curriculum20._nominal_path` 的中心线门检查 `|x|<10.5`、`|y|<5.8` 及 flat lane；随后冻结的 `assert_nominal_corridor_clearance` 还要求完整 reset robot envelope 满足 `|x|+radius<11`、`|y|+radius<6`，并避开非 floor 障碍。

保存数据给出的实际 reset 半径是 **0.4721638197714839 m**，nominal path 从 `[-8.0,-4.7]` 结束于 `[-0.8557316257144817,-5.65134683140746]`。最大 `|y|+radius=6.123510651178944 m`，超界 **0.123510651178944 m**；首次不合格为 0 起算 point882，1001 点中有 119 点越限。中心线可通过第一道门，完整包络不能通过第二道门。

这是预定命令与真实保存 reset 几何不相容的工程前提失败，不能描述为该 episode 的策略摔倒、越界轨迹、RL 学习退化或覆盖假设被否定。保存 path 是命令经真实 servo 规则积分得到的名义线，不是机器人已执行轨迹。原始 yaw 净积分为零，也不能证明实际 servo 末航向或名义横移为零。

此前 source GO 与纯接口检查遗漏了 B 新时序的完整名义路径加 reset 包络资格检查；本审阅明确记录这一遗漏。运行期安全门正确阻止了未取得资格的课程，应保留原门。不要删除检查、缩小包络、移动安全边界，或把异常改为可忽略的普通任务终止。

A host 558.27 s、A reader 312.75 s、B host 300.03 s；当前失败不是原时限不足。训练 soft/close/hard 1200/1380/1440 s、reader 900 s 均不放宽。

## 唯一修复候选：先静态资格

只选一个候选，将 B cycle1 改为：

- `[425,525):−.25`；`[525,725):+.25`；`[725,825):−.25`。
- vx 仍为 1.0；其他周期、地形顺序、warmup、出生点、1000-tick horizon、175 起步、控制器、优化与安全门不变。

这是 100/200/100 拍三段命令；原窗口从 300 改为 400 ticks，raw 绝对 yaw 积分从 0.75 改为 1.0，raw 净 yaw 仍为 0。改变必须在下一合同明列，不能称作原命令的等价实现。实际 yaw slew 为每 tick 0.006，±.25 间反向约需 84 ticks，100-tick 段才有到达对侧幅值的机会；75/150/75 方案不作为本轮运行对象。

候选只做一次已预选的纯数值验证，无参数搜索。沿冻结 `FullDriveCommandServo` 重算全部 1000 个间隔，用 source18 保存的实际 geometry 检查包络边界和障碍；报告末航向、全程路径极值及安全余量，不能先宣称实际净 yaw/横移为零。该结果最多支持新合同中的几何候选，不能成为训练 GO 或真实机器人运动资格。

root 的 `offline_corridor_20_02.json` 已完成，host 8.24 s；共 161000 次纯 servo advance（160 条原始 recipe 加一个候选），新增模型、物理、训练更新均为 0。A/B source0..79 的 160 条原始命令全部保留且中心线粗门均通过；其中 35 条由各自保存的实际 reset geometry 通过完整 flat guard，B18 被拒绝，52 条 flat 因缺少各自保存 geometry 而未资格，72 条 nonflat 不适用该 flat 避障门。不能写成“160 条完整几何资格通过”。

原 B18 经实际 servo 积分后的末航向为 **−0.10415999999999995 rad**，证明 raw 净 yaw 为零并未传递为实际 servo 净 yaw 为零。唯一 100/200/100 候选在同一 B18 保存 geometry 上通过原 guard：最大 `|y|+radius=5.522235862823916 m`，Y 边界余量 **0.47776413717608435 m**；到半径扩张后的编译障碍 XY 盒的连续线段距离保守下界为 **1.9618785396364158 m**。该附加距离只辅助说明；碰撞判定仍来自原 guard。

候选 nominal end 为 `[-0.7859669110467529,-4.906375020862564]`，末航向 `6.180630009525179e−17 rad`。末航向近零仍伴有约 **−0.206375 m** 净横移，不能称横移严格抵消。候选 raw command SHA256 为 `6c542dede9be02762b3b5e02aed7d46410637599fa6ba3d04eed1ee9405c9a87`。保存了逐拍 raw/servo、完整 nominal XY 与原 guard 回执；没有执行候选机器人轨迹或改写 C20 课程。

首次离线启动因 ROS 的 `scripts` 包抢占，在导入、进入 main 前失败，未发生 servo/model/physics 调用。原日志、回执及匹配原 SHA 的脚本另行保留；第二次使用已冻结的干净项目 PYTHONPATH 后完成唯一实际数值验证。候选选择没有依据多次参数试跑，也没有读取任何新策略评估分数。

## 后续合同的有限入口

1. 保持 C20 运行源、模型与历史记录冻结；后续修复只写新模块和新合同。原 first80×两臂的有限命令诊断全部保留，不删除失败条目。只有有对应保存 reset geometry 的 flat 场景能直接声称该实际 geometry 的 guard 结果；非 flat 不套用“避开全部地形障碍”的 flat 门。
2. `recipes20` 的公共课程含连续 uniform 取值；first80 或少数端点不能证明全部参数域或任意后续 source。新合同须明确有限预检 source 范围并在超范围时停止，或先提供连续范围的保守证明；复用 reset 几何则须显式证明相同 reset 语义，不能把未保存的未来 reset 宣称为实测。新开发/最终命令也须完成适用的静态资格；不读取其策略表现来选命令。
3. A1 可作为未来冻结 A 基线复用候选，仅当 A 的共同父模型、PPO/Adam 起点、seed/warmup、奖励/观测/动作、C18 控制、地形/出生点/horizon 与运行语义全部保持，并完成来源及初态桥接。未来 B 必须从共同 C15 grouped 父模型重新开始，并重新对 A1 核首1024配对；不续用 B 失败 checkpoint。若改到 A 的共同条件，则上述复用前提不成立。
4. 静态资格通过后才拟定新的独占预算与 source GO；任何新训练都不是补用 C20 余额。本审阅不授权启动该候选训练、不预占新 controls、不降低开发/回归/封存门。GUI、当前99D/16D的15mm、侧移、跳跃、自恢复仍各自未获本次资格。

## 绑定的保存证据

以下 SHA256 对应审阅实际读取的文件；路径相对于本目录。账本与诊断的生成者为 root，不能把它们描述为本审阅者运行了实验。

| 文件 | SHA256 |
|---|---|
| `source_review_20.json` | `467f63a7374e1cf6ff0a5098391e15e9f9a33eb59408848c58c9fea62021962c` |
| `plan_go_training_1_20.json` | `17d8cb0966716833ca490691c69a2f1f4f0eb8822c9da1eee407c0af20f312ba` |
| `train_A_1_readback.json` | `189792963dc0ff54513fe5091b0feb34d9547dfc0f9ccf45f517f06a68d26925` |
| `train_A_1_readback_host_receipt.json` | `1633777e831238730ea5f4866bd6f26b9fb3ffb12ab97ade0a7e1da61d2dd489` |
| `train_A_1/host_receipt.json` | `aa82d484f1944edd0b4bba35c21b08e9ed82d781549d734edeb0eed9c716d11d` |
| `train_B_1/host_receipt.json` | `6ac2533a37823f1a626a4503aecf0efb93dc32d72667bc52ca94b210c2d52b0e` |
| `train_B_1/worker_receipt.json` | `e2581d517c498ea1ca75869028e6a1e0d51691f3bb515ad45c45f56d63906da5` |
| `train_B_1/failure_learning_receipt.json` | `ccee7e444143042349080230c4288048917e97a3bbd9255ad3dd2962c6066a23` |
| `train_B_1/failure_checkpoint_manifest.json` | `2a4a6ebf8e67eb05ffaa35e7c644f7445f52f750614e98d756b342c9f31119ef` |
| `execution_ledger_20.json` | `4745a00b38e167e2dbb387d12c2e8fe462d9cc24ef17e60e53cadc7087f68e20` |
| `first1024_pair_20.json` | `b946c76c20c96595417fe36e38e88397ba7e7874eb9041e88b62722c947c3d32` |
| `failure_geometry_snapshot_20.json` | `e017b1b2abe56e0210d5a6aa29108b8a052230144130476fdf2186ef2329f550` |
| `train_B_1/training/training_episode_000018_flat_corridor_geometry.json` | `dc633768980a188785a7da41828d96f01e13d1208b57b51b80cbf66cee87cd0a` |
| `train_B_1/training/training_episode_000018_flat_corridor_nominal_path.npz` | `88e1f12c69a35ee0cbea3644adefd86cc91444543d3a0e3536385f19fd0c4834` |
| `offline_corridor_20_02.json` | `dc84e8b4d31ed40c126b8e5ed0d09dec67fd6832203e3983161b0e9f2198c1aa` |
| `offline_corridor_20_02_host_receipt.json` | `e07349a98adb9e03fbd049d07e3a379ac39696263aca3123d02cc1d1dac02f5e` |
| `offline_corridor_20_02_reservation.json` | `2a45babd290df8a00d9b7aee250b9765c6185986141102ff2035ef172c11c171` |
| `offline_corridor_20_host_receipt.json` | `47c38185e2446d43e1ad91c8088fdbda6bbb3bd302697365a2a86a9543a7a429` |
| `offline_corridor_20.log` | `041ce7ecc148d5f106c98341d9896c11e658366f1a90f64ca3e0ddbb69262323` |
| `reporting20/diagnose_corridor20.py` | `91401372b9a96e048628f165f029c3ad5d642ee1c87ee4426f2438317c2158f2` |
| `reporting20/diagnose_corridor20_attempt01.py` | `840df7711c46259c63d3b7cde0c02e05a3036a01cc9862e476deca110e635e8f` |
| `reporting20/corridor_contract_20.md` | `317107c1e93a3b42a230b1f0db769d0a2ec700426c534a031be3b52e8c9713dd` |
