# C28 zero_left：保存记录性能诊断

本备忘只读 `continuation28/zero_left_01` 的已保存记录及 C26 `keyboard_gui_01` 参考；未重跑模型、MuJoCo、测试或控制。C28 的 1883 controls/9415 native/+2 compiler 已实际完成、侧步 1055 拍，并在收尾时因 `recipes18` 未导入而使 `runtime_module_origins` 拒绝；记录不能当成有效结案或把读回失败改为通过。以下墙钟以 `deferred_archive_receipt.json` 的 active 起止为准，因为本次失败的 `performance.json` 两个 active 字段为 null。

## 量级与阶段

active=33.586562118 s，1883×.01=18.83 s 模拟时间，RTF≈0.56064。达到原门 RTF≥.8 需 active≤23.5375 s，即至少省 **10.0491 s，平均5.34 ms/control**。C28 准备时间戳在实际 side `[204,1259)` 的相邻间隔均值18.795 ms、中位15.271 ms；side 前 `[0,204)` 为16.026/12.755 ms，side 后 `[1259,1883)` 为17.302/12.748 ms。C26 保存的对应粗分段为约10–12 ms/control，active=11.247761 s/1102 controls；两次 run 的命令、接触姿态、宿主负载和时间不同，不是配对性能因果实验。C28有290实际render、290 poll、378发布帧；C26为140/141/223。C28原frame snapshot age的最近秩p95约242 ms，也接近250 ms门。

**主新增计算热点是冻结 Fast 的 scratch IK/接触几何链。** C28 `final_accounting.side_access` 明确记录 1055次compute、4次prepare_leg、42,678次 scratch `mj_forward`、12,910次 scratch `mj_jacBody`、3,853次 measurement `mj_jac`、359次fullM、2,111次已存在objectVelocity；逐scope静态CCD增量和为424,730。`python.forward_counts_by_runtime_model_data_identity` 中 scratch forward 恰为42,678，另有正常model 1,885、原nominal compiler model 2,571、构造2；C26无scratch而总forward仅3,679。这些不是新增normal physics积分，但会真实消耗墙钟。C28总CCD=487,732，剔除side-scope静态CCD后还剩63,002；C26该计数为0。记录不能把剩余CCD全部归给某一阶段或把调用次数直接换算秒数。

这一个热点**尚不能解释全部不达门**：C28非side前后已约16–17 ms/拍，超过达标平均所需12.5 ms/拍。即使假设把side相对本次非side的约2–3 ms/拍增量全消掉，估计也只省约2–3 s，离需要的10.05 s很远。纯侧步scope快照/诊断瘦身或只修 `recipes18` 来源都不应被当作RTF修复。时戳包含调度、render线程争用、原5T求解与 Python记录；它是端到端观察，不是每函数采样。deferred flush在active结束后用13.558 s完整压缩归档，不计上述active，不能靠改压缩等级声称让RTF达标。

## 唯一建议候选与离线判别

先把候选限定为**核实并消除 `_ik` 中同一scratch物理状态上的重复 `mj_forward`**，不改腿顺序、phase时钟、目标、权重、力矩、安全门或5T。冻结源码 `scripts/d1_fast_side_step.py` 的 `_ik` 在每个outer开始forward、每条腿inner开始forward、每腿结束forward和函数末forward；相邻调用之间某些路径没有写 `qpos/qvel/act/ctrl/time`。从控制流看，每次 `_ik(outer=n)` 至少有约 `5n` 个这种候选重复位置；正常 `_targets` 为outer=3，按1055拍可形成约15,825个静态候选位置，另有四次准备腿。但 `mj_forward` 可改写scratch内部的约束、接触、加速度和warmstart数组，**相同qpos并不证明第二次调用可删且字节等价**。因此此处只是下一版的单一待证伪优化接口，不能直接改已冻结Fast或宣称确定省时。

唯一有意义的0模型/0物理保存数据微基准：root可从这1883条已保存controls取side段完整 `side_calculation`/`diagnostic_before/after`，在新独占目录用同一Python/NumPy线程环境对原力分配的12×12线性代数及当前诊断数组复制/JSON转换做一次真实1055拍输入的回放计时，另记录读取/解析时间并核输出数值字节/来源hash。它只给出**可由纯Python/线性代数优化收回的上界线索**，不能测42,678次真实forward或证明删除forward等价；不重新导入Fast（其顶层导入MuJoCo），不把保存行重新送进引擎。若纯路径成本远低于10.05 s，暂停任何以诊断记录瘦身为主的物理重试；若够大，再按新合同隔离纯计算优化。要验证scratch forward省略，必须另立受控模型/物理资格并比较完整scratch内部状态和真实端到端墙钟，不能从当前保存轨迹推出通过。

原 `recipes18` 是收尾来源静态阻断：新冷预飞应实际调用最终 `runtime_module_origins`，不能只验证另一份导入名单；修此源闭合与性能问题应分开判定。C27/C28已执行源码、GO、失败和已保存数据均应原样保留。
