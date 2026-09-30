# C17-C：固定组合的七场景确定性资格验收

2026-09-30，Astra。**本合同须在B的两个定向候选均通过完整任务和独立物理读回、B来源/计步闭合及配对初态核验后生效。** 目前只冻结C设计，不以尚未到达的B结果预判GO。root核B汇总后，由Astra给精确来源GO，再运行两个C worker；常规开发与验收已获用户授权，不另设确认。

## 1. 在任何C执行前修订重复设计

原A/B方案提过“第二独立seed的原yaw与ramp”。B baseline换到171103/171106后，得到与C16对应grouped任务逐浮点相同的yaw/ramp成绩。这与源码一致：FullDriveCourseEnv seed只生成measurement_seed；D1ControlLoop把seed传给oracle provider/truth source，而plant reset仅依赖固定spawn/姿态；当前provider为无噪声oracle、policy predict为deterministic。**seed名称不同不能提供独立随机物理证据。**

因此在任何C场次执行或结果出现前，将C设计修订为**原六任务加一个镜像yaw，共七个实际命令场景**，取消仅换seed的重复原yaw/ramp，不修改任何原任务门。修订理由是消除没有新物理条件的重复执行，并明确证据范围；不是删除已经看到的C失败。C primary为最终组合的首次完整六任务zero/学习策略配对，mirror为尚未运行的负→正→负yaw序列。

本阶段称“预注册确定性资格验收”，不宣称独立随机试验、跨seed统计可靠性、未见地形泛化或随机初态鲁棒性。B的base/yaw/ramp单因素对照与C的固定组合资格分别报告；C primary不能伪装成此前从未参与开发的任务分布。

## 2. 唯一固定系统与脚本

两worker均使用冻结 `controller_variant=combined`：yaw内部上限1.2、gain4；common active时积分采用B已选eI共同误差和相应退绕方向；其余动作、保护、观测、reward、servo、几何与物理步长不变。全部16RL通道保持执行，不增加任务判断gate。

唯一策略继续为C15 grouped final，ZIP SHA256=`1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`。每个worker只strict load一次并核原32行deterministic probe；不训练、改权重、清Adam、重存模型或挑checkpoint。zero是同一combined控制器上正常传入零残差的明确对照，不代替学习策略资格。

`primary` worker按六种原case顺序，每case内zero→grouped；floor seed171190，六case seeds171201–171206分别为flat0.6、flat1.6、flat1.2+yaw、bumps0.4、rough0.35、ramp0.45complete。`mirror` worker floor seed171390，唯一case为flat1.2+yaw、seed171403、mirror=true、zero→grouped。新seed作为归档/重置身份，不作为增加样本独立性的理由；root在执行前核历史没有同一场次身份。

所有case的spawn、时长、raw/servo幅值与速率、drive/hold/release/final窗口仍为冻结recipes17的原值；mirror只把yaw原正100/负200/正100全部反号，保持vx1.2、|yaw|.3、hold[415,815)、release815、final[1500,1600)。不通过移动窗口、延长保持、提前制动、降速或改变地形使其容易通过。两个worker各先600步floor；floor全链独立读回通过才运行所属正式任务。

同worker内每个case的zero/学习策略初态qpos/qvel/ctrl/qacc_warmstart/observation逐位配对；controller与servo每次reset，积分从零开始。同一配置贯穿worker，不在场次内部切参数，不根据首个case成绩另选组合。

## 3. 新增预算

| worker | 固定场次 | controls上限 | normal native上限 | compiler上限 |
|---|---|---:|---:|---:|
| primary | floor600 + 六任务×zero/学习策略 | 20,200 | 101,000 | 2 |
| mirror | floor600 + 镜像yaw×zero/学习策略 | 3,800 | 19,000 | 2 |
| **C合计** | **2 floor +14正式轨迹** | **24,000** | **120,000** | **4** |

zero最多11,400 controls。学习策略正常predict最多12,600次/行（11,400正式+1,200floor）；probe2次×32行，predict API总≤12,602，actor rows总≤12,664。PPO load≤2、torch.load≤6；critic/value-only、learn/train/evaluate_actions/backward/optimizer/save均0。失败的attempted仍计账，未用额度不退款、不追加到别的worker。

primary host soft/close/hard上界1,200/1,440/1,500s；mirror上界600/720/780s。准确值写入C spec，可向下收紧，不得执行后延长。两个host合计预留≤2,280s，各额外最多5s仅kill/reap。floor clean reader每个≤60s，含所属host时间。正式独立读回复用新纯host v02，每worker正常等待295s、回收最多5s、总≤300s，两次合计≤600s；发生超时必须留失败，不把未完核验当通过。

若B已完整完成，B+C实际控制上限合计32,600、normal native163,000、compiler10；这只是C17新阶段合计，不覆盖C15/16既有累计账。B的原reservation和失败诊断原样保留。

## 4. 来源与验证复用

复用B已freeze且经过实际执行/独立读回的新controller/math/checker/reader/host/worker/eval/recipe/score/floor桥，不修改这些文件。generic combined/explicit seeds/mirror接口在B冻结前已经存在；C只是冻结新spec、来源GO和资格记录。无运行代码差异时不重复B的11项测试或旧C15/16批次。root对新spec做静态场次/预算/seed检查，对纯host v02读取新增源身份与语法验证即可。

C plan绑定B的完整开发汇总、三个worker/reader receipts、匹配初态结果、本合同、C spec/GO、原checkpoint以及全部依赖hash。沿用exact WorldUprightCourseEnv、cold构造、C/Python/native账、5T真实contactForce、原子归档、独立reader导入保护与前后同集hash。reader校验每tick实际variant、pre-state反馈、nominal yaw/轮速、eI/PI连续性、stop latch、RL执行及最终实际扭矩，不凭worker自报任务成功。

每worker独占reservation/output；软件/来源/模型/计步/归档/reader错误停该worker，保留首异常与清理；合法任务失败按冻结顺序留完整证据。失败不原位重试、不换seed凑通过、不用zero成功补学习策略失败。控制源被冻结后若需要修改，必须新源版本和新有界合同，不能在C进行中调整。

## 5. 逐场绝对门与单独贡献判断

每条正式学习轨迹必须完整通过原记录、安全、完整horizon、terrain channel、最终停止及原任务速度/yaw门。yaw hold RMS≤0.12；ramp hold均速∈[0.41,0.49]且vx RMS≤0.05，实际三个坡面均有正轮载荷、四整轮几何全清除；其他四任务全部按原门，含1.6 m/s任务高速要求。镜像yaw门完全相同。既不四舍五入失败值，也不以平均多个场次补偿某一失败。

只有学习策略在primary六项与mirror yaw全部通过，且两worker完整独立读回和来源/预算闭合，才能写“转向、完整坡道跟踪以及六任务回归完成”。zero成绩全部列出；zero过关不替代学习策略，zero失败不自动取消学习策略已证实的任务资格。

总体RL贡献只用primary六任务同一combined基控下zero/学习策略的配对结果，按原C15的drive SSE、逐任务/hold与成本门另判；mirror另列绝对/相对成绩，不事后改共同任务集合来改善贡献结论。若贡献门失败就保留false；基础控制修复的收益不能记作学习收益。力矩平方仍是归一化成本代理，不称机械能或电耗。

不自动授予default GUI、自由键盘、实机、侧移、跳跃、15mm台阶或自救资格；R仍为仿真reset。最终报告明确只有固定oracle仿真脚本证据，seed没有在此改变物理条件。若C任一所需项失败，应如实记录并按用户已授权目标继续下一有界修复，不把B改善或纯测试当作完成。

本合同编写只读源码和既存结果，Astra没有执行测试、模型、优化器或物理；此时C实际执行尚未开始。
