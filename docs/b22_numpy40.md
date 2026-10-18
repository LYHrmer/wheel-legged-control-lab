# 40：独立复核39已完成的flat保存记录

39的cold bootstrap与`flat_1p6_B` worker已结束，但saved reader因继承的Torch库清单非空假设失败。原39仍是`INCOMPLETE`，其`actual_counts`与`actual_nn_counts`保持null；旧源码、父回执、归档不修改，也不重跑flat。

旧`worker22.mapped_libraries`只筛选`/proc/self/maps`中的`/torch/`与`/nvidia/`路径。39禁止导入这些框架，所以保存的两份文件都是`{"paths":[]}`。旧reader的`mapped and all(...)`将合法空表误判为“actual DSO left closure”；现场没有新的未绑定DSO。这个文件只覆盖上述计算库，不能代表进程全部动态库；MuJoCo的ELF/GOT来源证明另外保留。

40是新的saved-only读取范围，只有一处语义适配：原`script_sources`中两份Torch/Nvidia清单必须精确等于`{"paths":[]}`。非空表即使每个路径已绑定，也会拒绝；缺字段、附加字段、错误类型亦拒绝。其余来源、session/GO、NumPy计数、32行probe、包/ELF/GOT、控制器、PI/z、每拍5次native、终止与完整窗口、原任务阈值及评分均不变。AST回退逐节点核验没有修改其他metadata逻辑。

三份新source是`read_b22_numpy40.py`、对应测试及本文。实际运行前须提交且与HEAD一致。40固定旧39的五份源码、原38八源、39的release/父回执/完整worker回执/cold bootstrap身份。执行时先冻结整个39输出树（包含完整flat归档、launch、session与所有保存证明），再由原39的数学入口读取；前后重新核验输出树、source和原39闭包。40仅在新`results/b22_numpy40_readback_20261018_01`写清单及报告，拒绝覆盖或重试。

默认只打印计划，不加载NumPy、MuJoCo或模型，不创建输出：

```bash
rtk proxy python3 -B scripts/read_b22_numpy40.py
```

得到源码审查与串行资源窗口后，运行保存结果复核：

```bash
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B scripts/read_b22_numpy40.py --execute
```

它不创建环境、不进行积分、不重新执行checkpoint、probe或NN oracle。进程禁止导入Torch、SB3、MuJoCo、ORT等模型/仿真运行时，仅允许NumPy做已有归档的数值复算。启动有效可用内存须≥6GiB，执行时每秒检查≥4GiB；Nice≥10、单线程，整体最多600秒，无训练或物理子进程。source身份检查使用有超时的Git命令。失败、信号、输入漂移或资源越线只写新的40 `INCOMPLETE`结果，原39保持原字节。

新报告明确区分：

- `new_actual_counts`始终为0：40没有新增任何控制/积分/预测/模型加载。
- `verified_saved_execution_counts`仅在原数学全部通过后记录39保存的1600 control、8000 normal native、2 compiler native。
- 保存的NN计数同样单列：1601次调用、1632行（32行probe一次，加1600次单行控制），不是40执行的新预测。
- 工程复核完成不代表任务通过；原评分与失败项全部保留，不授予完整矩阵、RL contribution或实机资格。

40不生成伪称39原始成功的readback，不评价尚未执行的yaw，不拼造两场兼容结论。后续补yaw须另行登记新版本、预算和输出；39原父记录仍不能改成COMPLETE。

准备阶段只执行stdlib/合成文档/假进程状态测试，不调用真实保存结果的数值入口。真实saved-only复核须在source审查、提交后，由根线程单独安排。
