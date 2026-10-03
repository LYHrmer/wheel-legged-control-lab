# C33 固定时序可行性草案（无物理 GO）

改 body shift 与 recenter 的时间曲线，保留 .015 m 原峰值加速度 ZMP 预算。r=.05 指完整 T 的比例；无量纲峰加速度 K=4/(1−2r)=4.444444，峰速度2，有限峰 jerk=K/r。q、q′、q″ 由 JSON 中同一 hinge 多项式给出，位置/速度/加速度连续，端点速度及加速度为0。实际 jerk/峰速度可能高于原 quintic，不能隐去。

固定 alpha=.85、.925、1.0 各左右一场，依次提高预算利用率，最多6完整场；T=max(.30,sqrt(K·|span_xy|·h/(g·.015·alpha)))。实际 pose/body_accel/记录的 body_reference_velocity 必须一致，并覆盖 recenter。其余沿 C31 fixed 的8mm回缩与lambda=.5；不加RL动作。

同一 alpha 必须左右均通过原物理、成本、误差与保留门才可入选，以左右最慢周期最小选最强固定对照，固定平局规则见 JSON。全部结果公开，不挑偏慢对照。获选者左右各做一次原取消试验，共最多8场、17600控制、88000正常native、2 compiler native，0新actor/value/optimizer。硬安全/来源/入口失败即停；受控任务失败关闭该点及更高 alpha，不重试。

新固定曲线若左右各自满足相对旧fixed至少5%、相对zero至少10%的周期降幅，只记固定工程收益。RL任务仍继续；后续同权限强固定对照及未用持出条件在训练前冻结，既有RL收益门不放宽。root 唯一执行；源与纯检查、既有精确gate/API帽绑定完成后由Astra另行GO。
