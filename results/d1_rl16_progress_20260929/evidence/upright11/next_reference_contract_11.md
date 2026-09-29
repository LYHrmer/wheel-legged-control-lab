# 11-R 世界竖直参考的唯一坡道基线合同

本阶段仅新增upright11源与证据；旧T、10及所有冻结代码/档案不改、不重启。实际Astra主审，Sol具体实现新env/loop/纯seam，root实现和独占执行worker/launcher/纯测试/独立读回。执行前提：10独立读回通过，下面新源码及实际纯测收据获得单独SOURCE GO。当前文件不是即刻发动引擎的GO。

## 唯一候选与真实性

新任务期望机身世界roll=pitch=0。真实compiled地形height/normal/geom ID、provider publication、scan、contact位置/法向/force原样；不删除82度端盖、不改MuJoCo模型/引擎/控制gain。新loop在prepare和terminal observation用同一纯工厂生成world command，姿态常量0且height=真实height+.455，其余servo/raw字段不变；单次原compute/step/5native继承。新env只在never-reset构造换新loop，同plant/provider/controller，无新模型或辨识。原16D residual mapping与99D数值编码不改。

新task的姿态误差=实际世界roll/pitch减0，20度task终止；45度native硬安全界不改。新reward只有attitude项使用此误差，原系数/尺度及其余terms不变。记录保留原geometry-relative姿态字段，并新增明确task-relative字段及新task/reward/reference schema。非轮contact、地图边界、clearance<.28和所有记录/有限/几何规则不改。旧T失败不改判。合法task_fail应正常返回terminated；时间截断用真实post-state并保留truncated语义。

## 纯资格

最多2轮，每轮<=8个新case、30s；首轮过即停，只允许具体首败修正后的第二轮。0MuJoCo/模型/predict/learning。至少覆盖：实际82度fixture真实hit仍保留但新可执行目标0；prepare和terminal两条真实新seam参考一致且一次compute/5假子步；新reward/endpoint/scorer世界参考一致；健康truncation/bootstrap；真实几何或非有限错误不可吞；新schema/strict任务绑定；flat世界0与原flat目标一致。可在同一case内核BudgetSpec整数与整除等纯接口，不能重跑旧6/16/8 suites。

## 唯一物理基线

一个新独占进程和run目录，最多180s；最多1800 controls /9000 normal native +2冷compiler。每control原5×.002s，control .01s。0policy load/predict/learn，严格zero16。无独立static probe/模型预演，只有原构造/reset/同步所需已计数static；同一CoursePlant真实92world geoms。

spawn=(2.75,0,.455)，reset seed=88611。先保存全部1800 raw命令及纯servo积分的nominal path/hash：
ticks[0,175) raw vx0；
[175,1355) raw vx+.4m/s；
[1355,1800) raw vx0。
全程vy/yaw0、jumpfalse、clearance.455。servo原Δvx≤.005/tick，故纯名义最终x约7.47；它只是预检查，不充当实际进展。不得依据运行结果移动release、延长episode、改speed/seed或补步。

使用原native guard full qualification记录分支（heldout），逐native真实接触/force、before/after integrator、actuator、endpoints、raw/servo、实际world command、controller chain和任务reference存档；按构造真实绑定geom/body/wheel、base ipos、qpos/dof/actuator映射。所有normal步进/静态调用/C/Python计数真实记录。合法task提前结束即停止这场，记录有效task_fail；fatal几何/record/engine/warning按原规则停止。均无retry，剩余额度作废。

## 基线出口，严禁等同旧资格

execution/record_valid要求真实C/Python/guard attempts/returns/clock一致、每返回control5native、compiler2、warnings/forbidden/nativefailed/violations0、source前后不变、phase/targets归零、无orphan。

reference_baseline_passed另外要求完整1800，没有task终止；全native及初态绝对roll/pitch各<=10deg、clearance>=.28；无solver active非轮terrain contact；base全程|x|<=10.5、|y|<=5.8，且从spawn heading的横向偏移<=.25m；所有真实geometry检查通过。三指定坡道geom（terrain_ramp_up/deck/down）分别至少一次真实正wheel-contact normal load；不能用候选contact或body下方ray代替负载。

最终状态每个轮碰撞geom的世界X最小投影均严格超过所有三ramp geom世界X最大投影（由真实compiled primitive姿态/尺寸求精确投影，不能用nominal path/轮中心冒充）。末100个post控制样本mean(abs(COM_vx))<=.05、mean(abs(body_yaw_rate))<=.05、真实clearance population std<=.02m。另报告完整drive与[600,1000)固定窗口的真实COM速度mean/RMS和停止距离，速度误差在本reference基线是描述项，不自创1.6/低速正式速度资格。

独立读回检查新world目标、保留真实几何字段、实际扭矩/force、全部计数和full-horizon/partial区分。基线仅支持世界竖直新任务的后续研究，不能据zero通过声明RL贡献或整course高速可用。若record/geometry无效，RL硬关；若有效task失败，先依据明确失败源作下一有限决定，不自动长训。新65536训练、12个正式heldout及GUI数值验证必须另立执行合同与SOURCE GO；本阶段不继承它们的预算。
