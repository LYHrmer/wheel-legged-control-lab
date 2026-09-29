11-S 执行前独立审阅：GO。实际 GPT-6-astra ultra 只读审阅新课程、训练核、持出记录器、评分器、最终 CLI 与 host；本审阅没有导入或执行模型、引擎、测试或物理。

本轮新纯测试由 root 实际执行一次：8/8 PASS，1.1172713 s，0 engine/control/native，输入哈希无漂移。两个接口测试实际构造全部六种持出日程，检查 1001 点路径和 1000 tick 截断、最后 done 后零控制 reset 与 Gaussian 回调的配对；六个评分测试覆盖动态窗口、世界参考、坡道三段载荷和完整轮形状、合法任务失败、非法记录、停车与贡献成本。测试前后源码及读取的真实 92 geom fixture 已逐项核对。首轮通过，不启动第二轮。

执行前确定问题已修：持出 release/final 时间窗拒绝谓词方向、1000 课程误用旧 1601 点路径、外部 reset 种子与 provider 种子混称、内部 profile 覆盖总入口计数，以及纯测试隐式导入 Gym。最后以 root CLI 唯一真实 sys.setprofile 记录 PPO.learn 与 BaseAlgorithm.save/load/predict；训练核不再安装或清除 profile。已按本机 SB3 源码核对方法文件名。旧 learning11/world-upright/引擎/控制增益和评分阈值未改。

唯一新过程上限 85136 control / 425680 normal native + 2 冷 compiler，1800 s，无重试。训练必须恰 65536、64 rollout/train、256 epochs、1024 optimizer，随后同唯一最终字节完成 1 save / 2 strict load / 4 批真实训练观测 probe。正式 heldout 为五对 1600 和一对 1800，zero 预测 0、policy 每 control 一次；合法任务提前终止保留并核实际计数，段间余量不得借用。零残差和策略初态五数组逐位配对，坡道最终投影读取已同步 measurement cache，真实 qpos/qvel/time 一致，不增加 forward 或模型。

首次32训练观测作为 probe，持出数据不参与训练、目标重抽或 checkpoint 选择。外部 reset seed 按 SeedSequence([88623,i]) 固定，SB3 首次 seed 仅记录；真实 provider seed 从 frozen Env.reset metadata 分别归档。训练 compact 原生记录与持出完整接触力记录的证据范围不同。CLI 将构造、C/Python/native guard、真实调用者、警告、末 phase/target 和异常前缀单独闭账。

此 GO 仅接受精确冻结源码在该预算下执行，不是结果通过声明。训练完成、六场策略能力、1.6 m/s、RL 相对同任务零残差贡献与 GUI 性能分别等待独立读回。旧失败 T、不完整训练和无效资格不被本轮覆盖；A/D、Space 与 GUI12 均不在此额度，不能据此宣称已实现。
