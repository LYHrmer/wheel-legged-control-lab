# B22 42 修订02：学习前接口失败

修订02实际运行约164.31秒，依赖bootstrap完成，随后首个训练环境被旧控制器的精确类型检查拒绝。训练、checkpoint加载和控制循环尚未开始。该次执行按INCOMPLETE关闭，拥有的子进程已清理；没有新的候选策略或资格结果。

[紧凑回执](../results/b22_tracking_reward42_revision02_summary_20261018.json)保留源commit、原文件SHA、模型API计数、失败和资源记录。原运行来源为`cd32a85f7815fae69f89b0b8d2c769e8f0d2262d`，输出为`results/b22_tracking_reward42_20261018_02`，release SHA为`33f2323fc20647481810995af81886be9acdb28cde236547785b4992d4acd83f`。

## 实际问题与修复范围

原`controller18.install_controller18`要求`type(env)`与原生产Env完全相同；42训练用奖励子类，因而在构造后报`TypeError: C18 installer needs the exact production world-upright env`。此前已经适配了课程的同类检查，但遗漏了控制器安装入口。真实失败表明原pure测试没有覆盖这个生产接缝。

03修复在固定SHA的原installer函数上建立私有globals，只将Env精确身份映射为本次注册的奖励Env。其他Env子类仍被拒绝；原loop、controller、never-reset、同plant/provider/caps及唯一几何cache命中检查保留。评测沿用原Env与原installer，冻结runtime源码不修改。回归测试直接使用原生产函数体，覆盖这一错误及不应放行的身份条件，不创建模型或做积分。

## 已消耗的计算

| 口径 | 原02结果 |
|---|---:|
| 已知controls / normal native | 0 / 0 |
| 已知compiler native | 2 |
| learn/load/train等模型API | 未调用，ledger为空 |
| 已知纯servo advance | 0 |
| 完整actual_counts | **null** |

环境构造发生了2573次`mj_forward`，不能把本次写成零计算。上述已知前缀也不能补写为完整精确总账。worker正常返回失败退出码1，监督与父清理remaining均为空；旧02回执和产物保持不变。

03作为新的接口修订独立登记，从完整B22和Adam开始两组对照，不使用02 partial。新执行上限仍为96136controls / 480680normal native / 6compiler native，旧02额外2次构造积分单列；所有尝试精确总量仍未知。

## 本机资源

本轮采用6GiB连续10秒启动、4GiB运行保留量，以及整组MemoryHigh=2GiB / MemoryMax=3GiB / SwapMax=0。父回执最低有效外部余量为5,755,703,296B（约5.36GiB）。末次owned快照记录kernel peak为2,148,880,384B（约2.001GiB），high事件32783，max/OOM/OOM-kill均0；这是最后可读峰值，不能冒充退出后的完整峰值。

组外采样显示高水位占用主要来自文件缓存，已发生回收节流。明确异常是环境类型检查；现有证据没有显示3GiB硬限额或OOM导致此次失败。修复继续使用同一内存配置，无需恢复更高启动门槛。
