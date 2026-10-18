# 37：独立补齐0.5增益的缺失场景

37是新预算、新owner、新C37 worker／reader的有限评测，不是35重跑、续跑或余额恢复。没有训练，不修改模型、增益、指标或任何旧输出。旧35仍为INCOMPLETE，旧outer.actual_counts与active_stage_actual_counts保持null；已知前缀18400 controls／92000 normal native／22 compiler不能被改写为整轮实际总消耗。

## 纳入证据与新预算

旧目录`results/b22_gain_suite35_20261018_01`中的parent完整readback map恰有11项：step_pair以及flat0.6、flat1.6、yaw、bumps、rough的zero/B五对。纳入集合必须与这11项完全相同，任务通过或失败都纳入，不能挑阳性。冻结旧parent、release、reader、worker、所有11份独立closed读回与全部原始工件SHA，再复用其已有数值评分；不把旧C35身份重贴为C37。

旧ramp_zero目录已经含preflight_ready、construction和live_status，说明旧尝试已经启动，不能声称0消耗或尚未启动。它没有完整独立closed读回，不用于37评分或资格；未知消耗依旧未知，不回填旧report。

唯一新四会话沿原场景键：

| 新场景 | controls上限 | normal native上限 | compiler native |
|---|---:|---:|---:|
| ramp_0p45_complete_zero | 1800 | 9000 | 2 |
| ramp_0p45_complete_B | 1800 | 9000 | 2 |
| yaw_1p2_mirror_zero | 1600 | 8000 | 2 |
| yaw_1p2_mirror_B | 1600 | 8000 | 2 |
| 新37合计 | **6800** | **34000** | **8** |

若合法短回合则记录实际较小计数，不补拍。新actual_counts/known_actual_counts始终只属于37；旧11份validated_case_counts单列18400／92000／22。完整时combined_validated_case_counts最多25200／126000／30，仅表示两次独立执行中纳入分析的已验证case之和，**不是历史总资源支出**。历史总量未知这一事实不会因为37完成而改变。

## 不变的数值路径

`run_b22_worker37.py`先钉住worker35及其C34来源，再由完整35 script owner派生。全模块AST除C35→C37准入合同literal外相同；外层只准入上述四个固定spec。动作仍是同一个B22 99D／16D stateless策略输出乘float32(.5)，旧账本字段名gain35沿用以保持原数学与逐拍核验。未加入32省调用、控制器／PI/z变更或其他策略。

新reader逐tick复用原31／C26控制、PI／z、原生积分、接触、reset、终止、配对和评分检查；新session/GO/实际worker argv、source和derived AST均记录C37，不伪装C35。ramp末端几何仅引用旧35已closed的step_pair编译机器人运动学：旧完整SHA链先核，原robot_binding_bridge24再确认新ramp编译机器人绑定一致，reset/controller/source仍按原新场景记录逐项验证。没有宣称37新跑过step。

把11份原C35与4份新C37按原15-stage顺序交给冻结31纯aggregate，不修改任何输入报告标签；原六RL contribution门、real15mm两arm资格、mirror另列与七项B任务门都沿原义。相对31的no_task_regression只指原task_passed/task_qualified由真降假的情况，不等于每项连续SSE/cost必改善；连续指标和原六SSE/cost门保留原score。没有自动替换默认模型／控制器，也没有多seed泛化或实机声明。

## 资源、来源与唯一入口

全程串行，启动有效RAM≥6GiB、运行≥4GiB；启动磁盘≥2GiB、运行≥1GiB，Nice至少10。900秒整轮上限，890秒触发有界收尾；每worker最多300秒、saved reader150秒，INT／TERM／KILL只清自有session/subreaper后代并核birth tick。Popen尚未返回handle也使用原可靠host清理；工程／资源／来源错误立即停止，未知计数为null。任务失败如实归档后继续剩余原定场景。

默认只列计划，0NN／物理，不claim输出：

```bash
rtk proxy python3 -B scripts/run_b22_gain_completion37.py
```

新五文件须先提交且与git对象一致，实际来源闭包还绑定原35输入及590项旧归档身份。启动前检查新版本与旧证据，不加载模型做准备。唯一固定输出`results/b22_gain_completion37_20261018_01`，存在即拒绝；NOT_READY未claim时可等资源，claim后不重试、不resume、不退款。

源码与纯fake回归、独审通过后，根线程在其他重任务结束后以独立有界systemd用户单元和Nice10安排：

```bash
rtk proxy python3 -B scripts/run_b22_gain_completion37.py --execute
```

本轮交付仅准备源代码、fake／metadata回归及已有工件SHA核验；没有实际新模型调用、仿真、GUI或积分。
