# C22 固定模型的开发与最终封存合同

完整值见spec22。只有四条yaw的分段窗口发生一次有明确几何依据的修正：开发正向[430,530)+.30/[530,730)−.30/[730,830)+.30；最终正向[440,540)+.28/[540,740)−.28/[740,840)+.28；两个右向逐段反号。速度、drive/hold/release/final/horizon、seed、出生点和所有门与C21/C20保持一致，原七/floor/bumps不改。旧不合格时序是120/160/120和130/160/110，修订均100/200/100；400tick时长与raw绝对yaw积分分别1.20/1.12不变，raw净积分从±.240/±.224改为0，实际servo与名义轨迹必须独立保存验证。

C20/C21均未执行开发或最终模型；这次修改只来自已保存名义几何失败，不读策略分数、不挑有利case。修订测试平衡转向的时刻/幅值/速度组合，不再声称测试旧净转向命令。最终1.1m/s/.28rad/s组合仍不等于B课程任何完整yaw命令；有限新组合结果不能替代真实随机化或多seed证据。

A为C20唯一A1；B为C22唯一完整32768步B1；old为共同C15 grouped；zero为零残差+C18同基控。开发先对A/B各600-control floor，再原七A/B共22800control，再开发四场×四actor共25600，总49600。旧七zero/old继续复用C18保存原始记录及已冻结reader，原actualseed181201–206、mirror181403不变，要求完整命令/评分/compiled binding/初态及reset源码桥接。缺证明则未定，不加跑旧批次。

新四场每场1600，次序zero→old→A→B，按相同完整初态和控制隐状态reset配对；floor seed201090。每个正式actor只load一次，32行probe一次。最终重载同一old/A/B，各32行probe；不重跑floor或旧七，复用开发阶段冻结的相同模型资格与回归结果，只执行原四封存场×四actor，共25600control。新模型/动作分歧后的实际轨迹不同不能作为配对失败。

新场绝对门保持C20：hold400点、速度均值目标±.04/RMS≤.05，yaw相对已消费servo RMS≤.12与正负积分符号，世界roll/pitch≤10°、clearance≥.28、无非轮接触/地图越界/warning，final100 mean|vx|与mean|yaw|≤.05、clearance总体std≤.02；bumps原通道和真实载荷门不变。4.2s/3.2m制动硬门仍仅原flat1.6。

每case drive均值E使用vx/.25与yaw/.4归一化平方误差之和，再四case等权；yaw/bumps每类两case均权，另报400点hold与过渡。力矩成本是实际5native×16电机的归一化平方均值，再按case等权，不是能耗。原六仍原raw pooled SSE/成本，mirror不进分母。合法早停是失败样本，不能用短前缀或存活子集提高pool。0/0可满足非增加门但不建立严格改善，正数/0失败，禁止epsilon修饰。

开发描述性门沿C20：完整来源/覆盖、全部16场完整安全、B四开发+原七通过、总体B/A≤.90和B/old≤.95，两类drive/hold分别B/A≤1.02；开发等权pool与原六pool各自成本B/zero≤1.20、B/A≤1.10。现在这些数值不触发续训或模型选择。

最终执行资格只要求A复用/B完整训练有效、新首1024配对、开发读回来源及归档有效、所有应运行记录存在且合法、C18回归复用完整、A/B floor都通过、没有软件/引擎/来源/归档错误。合法正式任务失败或coverage不足须保留并报告，不用于挑选性取消已经固定的final。任一floor失败则该actor不跑余下开发正式场、其他合格actor可按列表完成；由于缺少完整A/B资格，最终worker不启动。坏记录/软件错误中止worker且无原地retry。

最终课程收益门保持C20数值：两臂配对与预期暴露有效、16条final记录完整安全，B原七与全部final任务通过，final E B/A≤.90、B/old≤.95，每类drive B/A≤1.02，final成本B/zero≤1.20、B/A≤1.10；hold/过渡和开发/原六成本一并明示。覆盖不足是机制未定，不能写成假设被否定或课程收益成立。最终数值不回写开发，不挑checkpoint。

A/B分别按冻结原六完整贡献门判RL是否超过zero：共同记录/安全/无能力回退/pooled成本≤1.20，至少一次zero失败→policy通过，或至少四共同通过且pooled SSE≤.85并至少四单对≤1.02。历史贡献分支与训练臂A/B不得混淆。课程B胜A不自动等于RL贡献；一组训练和有限命令不构成多seed统计可靠性或物理随机化资格。

各评估host soft/close/hard1500/1740/1800s，reader1200s；开发predict≤43203(43200单行+3个32行probe)，actor rows≤43296；最终predict≤19203，actor rows≤19296。各PPOload≤3/torch≤9，critic/backward/optimizer/learn/save均0。全部实际尝试与失败计入独占预算。
