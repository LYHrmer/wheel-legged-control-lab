# C20 失败后的离线几何补充合同

只读 C20 已保存命令、实际 compiled reset geometry 与原始名义路径；不修改冻结运行源、原始档案或安全门。实际 gpt-6-astra ultra 指定检查范围及唯一候选，root 执行。

原始检查为冻结表 A/B 各 source 0..79，共 160 条 × 1000 次纯 servo advance。每条都报告中心线结果；只有该 source 已保存自身 compiled reset geometry/path 的 flat 条目运行原包络 guard。无保存几何的条目明确未资格，非 flat 不套用平地避障门。原 B18 失败必须用原路径/原几何复现。

唯一候选为 B18：vx=1.0，yaw [425,525)=-.25、[525,725)=+.25、[725,825)=-.25，其余字段不变。只做一次 1000 次纯 servo advance，使用该 episode 原始保存几何。记录全部候选 raw/servo/heading/XY、边界与障碍余量。窗口从 300 增至 400 ticks、raw 绝对 yaw 积分从 .75 增至 1.0；不声称只改变对称性。75/150/75 候选不运行，无参数搜索。

总共最多 161000 次纯 servo advance；model、physics、compiler、training 全为 0，host 600 s 上限。失败不重跑同一执行。该检查不验证动态跟踪、当前策略、未知初态或连续课程全域；静态通过也不是训练 GO。保留 C20 A1；B 失败权重不得补训。后续新训练需新的完整来源与预算合同。
