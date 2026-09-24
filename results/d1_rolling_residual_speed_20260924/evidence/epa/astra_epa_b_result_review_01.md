# Astra EPA B result review 01

**B固定24例全部通过；允许继续实施隔离引擎集成C/D。** Root完成唯一B进程，24 CCD attempts/returns、48 init attempts/returns，退出0、无warning/stderr/源变化。加A16，本合同40/40已耗尽，无任何机器人积分。实际日志SHA `5a7566ad946ebafee8ed33aea774b1fc2d781a340ec8c67fc89cc5544255467a`。

独立复算结果存于 `astra_epa_b_result_checks_01.json` SHA `9d2055b87e842f2c58c283a41ef6a0d5247e384e093a1e4816f2154d8db5b4e9`，不调用root geometry函数或引擎。全部24条保持有限、nx1、负penetration；实际半margin膨胀表面的最大绝对SDF为2.262580994108563e-7m，小于原1e-6门。box局部支持cone和原位置容差全部有效。8个预定刚体变换逆变换后的最大坐标误差约8.89e-16m、normal差约1.39e-12，明显在原界内。15个旧健康query的距离和witness逐位不变。

query7按机制预期在k4拒绝global gap小而local gap大的旧face1，继续一次扩面后于k5返回face19。实际local gap8.299370525674185e-10m；terrain→robot法向约[0.00100000214,8.58e-13,0.9999995]，witness surface误差分别−1.850535e-10m和−3.396505e-12m。没有通过丢接触、翻转normal、提高迭代预算或改容差消除问题。Root另按原wrapper distinctness及主距离覆盖重组候选，四组数量仍为3/3/3/1，全部原cone/position门通过；本审阅已检查其实现采用max(margin,abs(dist))+1e-7并对照输出。

可确认层级：这一个局部EPA停止证书修补已修复固定最小复现并通过预定静态回归。未声称官方通用MuJoCo已修复、真实全机器人已合格或RL已完成；原box record_invalid仍原样。原epa_status0仍只表示初始化成功，收敛判断来自实际停止trace。

下一动作已经下发Sol实施，不停止在计划：按 `next_engine_readiness_plan_01/next_contract.md` SHA `3bd7315f39dbc82e9a500a233c2e756c39e9e53fe07ed1da03c2b46a45863d65` 和RL门澄清 SHA `207d4c73d838a395031de765e96b3c4e19a5e7f787411208af0ae2aa3eb8f1a9`，在新engine01目录构建隔离DSO，证明实际PLT/GOT绑定和compiler hidden-step计数，再新预算执行两场资格。有效task_fail可作为有界RL基线；记录/plant/几何无效才硬关RL。后续速度目标0.25m/s须配对同速zero并诚实区分RL贡献。

本审阅只读取日志/源码并做纯NumPy/JSON分析，新增engine/test调用0。
