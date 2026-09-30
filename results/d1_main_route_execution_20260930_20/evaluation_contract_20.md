# C20 开发、回归与最终保留评估合同

场景的机器可读完整值以 `spec20.json` 为准。所有新命令数组在首次模型/物理前序列化并hash；root须封存最终集，首段开发决定完成之前不执行、读取或使用其任何分数。

## 1. 旧七回归与初态复用

原六脚本保持C18的数值任务、horizon、窗口和门，镜像另列。原C18实际seed依次为181201–181206；镜像181403。单actor总11400 controls。A/B分别新评，共22800；zero/旧grouped复用C18保存记录，不重跑历史七场。

跨会话复用须由新合同先声明，clean reader核旧来源和实际场景：编译model/geometry/binding、数值controller/residual与action scale、dt/native实现、完整raw/servo规则、评分与窗口；五数组qpos/qvel/ctrl/qacc_warmstart/observation逐位相同。初始PI、z、stop latch、previous servo/action、命令servo和provider状态须相同；C18旧记录若没有独立初态文件，则用已保存reset/第一拍pre字段和连续性证明，不臆造缺失值。新旧model地址无需相同，但真实结构/数值须有绑定。任一关键证明不足，停止该复用比较和续段资格，不临时加跑zero/旧批次。

原六RL贡献门原样独立计算：记录、安全、无能力回退、pooled成本≤1.20为共同前提；A分支至少一项zero失败→policy通过，B分支至少四共同通过、pooled SSE≤.85且至少四单对≤1.02。镜像不进入分母。本合同的训练臂A/B和历史贡献分支A/B是不同概念，报告采用完整标签避免混淆。

## 2. 新开发与最终场景

每场1600 controls，final[1500,1600)，出生点沿原terrain固定。所有窗口左闭右开，raw vx在[start,release)取表值，其余0；yaw只在表中区间启用。净空.455、vy=0、jump=false。

| 集合/场景 | vx | start / hold / release | yaw raw时序 |
|---|---:|---|---|
| dev yaw left/right | 1.20 | 175 / [430,830) / 850 | [430,550)±.30；[550,710)∓.30；[710,830)±.30 |
| dev bumps375 | .375 | 200 / [285,685) / 710 | 0 |
| dev bumps350 | .350 | 225 / [305,705) / 730 | 0 |
| final yaw left/right | 1.10 | 200 / [440,840) / 880 | [440,570)±.28；[570,730)∓.28；[730,840)±.28 |
| final bumps300 | .300 | 225 / [295,695) / 720 | 0 |
| final bumps385 | .385 | 190 / [277,677) / 702 | 0 |

每集合是4场，不是6场；yaw left/right各占一场。dev seeds201201–201204，final201301–201304，只作记录身份。实际命令内容与进入时间不同才构成新场景；本次未随机化初态/摩擦/载荷/噪声/时延，不把seed变化说成独立物理随机样本。

每场按zero→old→A→B，严格reset配对；旧七A→B。命令计划相同不意味着分歧策略之后有相同轨迹。正式前先对每个新A/B唯一final做600-control低速floor：175 settle、250 ticks raw vx=.4、175 stop，原C18门不变。first-stage floor seed201090，第二段新final用201091。old已经有C18资格，不重复floor。无续段时最终worker重载同一A1/B1，不重复floor/旧七。

floor失败的actor跳过其余正式场，关闭续段资格；其余合格actor可完成已列场景，但不能宣布完整两臂比较。软件/引擎/来源/归档无效则停止worker；合法物理早停保存原始失败并继续合同内其他场次，绝不补跑或删掉失败。

## 3. 数值门、评分与一次继续决定

原七全部按原门。新场景沿同一C18绝对标准：保持段400点，真实base-body惯性COM前向均速在目标±.04 m/s、RMS≤.05；yaw相对已消费servo RMS≤.12 rad/s且正/负servo对应积分符号正确；世界roll/pitch≤10°、clearance≥.28 m、无非轮接触/地图越界/引擎warning；final100 mean|vx|和mean|yaw|≤.05、clearance总体std≤.02。bumps须原地形通道（横漂≤.25 m、航向≤.15 rad、净前进≥.75倍正servo积分距离及真实正轮载荷）。停车时间/最大原生超行程均报告；4.2 s/3.2 m硬门仍只用于原flat1.6，新场景不能冒称已验证更严制动能力。

新集drive为每场自己的[start,release)，使用每tick `((post vx-servo vx)/.25)^2+((post yaw-servo yaw)/.4)^2`。先取每case drive均值E=SSE/N，再对4case等权平均；yaw、bumps各2case等权。hold使用同样误差在表定400点平均，过渡误差另报。成本取每tick五原生×16电机的实际归一化扭矩平方平均，再按相同case均权；不叫机械能或电耗。原六回归仍用原raw pooled SSE/成本定义，不能混用权重。

续段要求以下全部成立：

1. 两臂完整有效训练/final、实际更新与覆盖门通过；所有16条开发记录完整horizon并安全，B的4条开发任务和原七任务全部通过，所有比较来源有效。
2. dev总体误差B/A≤.90、B/old≤.95；yaw和bumps每类drive B/A≤1.02，且每类hold B/A≤1.02，避免仅用加速改善掩盖保持退化。
3. **开发等权pool和原六回归pool分别**满足B/zero成本≤1.20、B/A≤1.10；镜像成本另报。

所有比值使用冻结的全部对应场景；不能把安全早停的短前缀作为较优误差或只聚合存活case。分母0时不加epsilon：两者0可满足≤1.02/成本无增，但不能建立严格改善；正数/0不通过。缺失/无效为未定，不生成通过。有效但不满足为数据不支持扩训；覆盖/证据不足为假设未定；两者均不补预算。

## 4. 最终判定与预算

继续决定完成后，最终模型固定为A1/B1或A2/B2，不按最终分数切回较好旧checkpoint。封存四场仅跑一次。

“本次课程覆盖收益得到有限保留场景支持”的必要门：B原七和全部最终任务通过；final总体E的B/A≤.90、B/old≤.95，每类B/A≤1.02；final成本B/zero≤1.20、B/A≤1.10。必须同时报告hold/过渡与逐场失败。超过zero的RL收益另按原六完整贡献门对A、B分别判定；课程B赢A不自动获得RL贡献或可靠性资格。新final pool是命令验证结论，不回填历史贡献结果。

| worker | floor | 新旧七回归 | 新4场×4actor | 总controls / normal native / compiler |
|---|---:|---:|---:|---:|
| 首段开发 | 1200 | 22800 | 25600 | 49600 / 248000 / 2 |
| 无续段最终 | 0 | 0 | 25600 | 25600 / 128000 / 2 |
| 有续段最终 | 1200 | 22800 | 25600 | 49600 / 248000 / 2 |

每评估worker冷构造一次课程+一次nominal compiler；加载old/A/B各一次（PPO≤3、torch.load≤9），各32行probe。首段开发或有续段最终的正式predict≤43200，probe后actor rows≤43296；无续段最终predict≤19200、actor rows≤19296。critic/backward/optimizer/learn/save均0。headless评估不借道GUI，不额外重载或进行模型诊断。

full reader沿C18的来源、reset、真实pre-state/servo/PI/z、原生5T/contactForce与geometry链扩展到新actor/新时序，不能把纯score输出的asserted bool当证据。每stage完成后才作继续/最终判定，clean reader不得导入项目policy/physics或继承LD_PRELOAD；floor同样走独立清洁子进程。host、reader和退出预算见plan/spec；实际partial调用与失败始终计入消耗并单列。
