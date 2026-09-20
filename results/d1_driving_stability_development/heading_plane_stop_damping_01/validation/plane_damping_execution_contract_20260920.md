# 固定 native-plane 停车腿阻尼：主代理执行登记

用户已授权常规开发、验证及及时上传；本登记采用 gpt-6-astra / ultra 制定的 `plane_stop_diagnosis_01/next_contract.md`，SHA256 `9f3acb12ad8949e65b237333eddecdd927443b1cab17e76d083658603bab5967`。其“规划，未授权或执行”是规划代理当时的状态，不改变用户对主代理物理验证的授权。报告由主代理独立离线重算，SHA256 同为 `0ccbdfdc4cc7234ae1395585785b28b343667d0b52da614f7959bb916d30edc7`，没有新增积分步。

## 实施说明

唯一候选为真实 Claude Opus 已实现、主代理集成的固定 b=126.4374005337902 N·s/m 停车腿阻尼。保留原命令、原始评分、原保护、轮 PI、heading servo 和 .6 yaw cap。采用经过只读独立审查的 cooperative MRO：PlaneStopDampingEnv → StopDampingEnv → D1FlatPlaneHeadingEnv → D1HeadingTrackingEnv，先构造 plane，再在首次 reset 前替换一个 controller。父控制执行和 reset 只调用一次；没有启动旧 hfield 8k runner。

全部其余规则沿用上述合同。仅执行四条 candidate（正倒停车各800拍，正负航向冲击各1200拍），新增最多4000控制拍/20000物理子步；既有 flat_plane_02 四条基线只读复用。停车前400拍及状态/观察至400完整逐位比较，无停车冲击全1200拍/1201状态逐位比较。新增完整 native entry 哈希比较包括接触、时间、ctrl 和 wrench。same-case 不应用任何 signed-zero 豁免。

固定顺序一次执行，首次终止即止；任务门槛失败可继续其余固定case。数值、物理表示、来源或配对失败立即停止批次，完整保留已经执行的步数与部分区间，禁止重试或补时。记录未保护代数功和同态保护后增量瞬时功，均不称作积分区间耗散证明。结果不自动进入默认控制器或GUI。

## 前检来源说明

采用已经通过的 Opus damper 纯测试、archive 纯测试及原 plane 非积分前检；新组合纯测试额外检查双层checkpoint拒绝、非法action和关闭异常。全部新组合测试阻止 mj_step/mj_step1/mj_step2。

原 plane 前检测试文件在证据归档后仅被 Ruff 调整一个 from-import 列表顺序；原测试原件保存在仓库 `results/d1_driving_stability_development/heading_flat_plane_01/source/tests/test_d1_flat_plane_env.py`，匹配原前检 SHA。主代理核对除该 import 名称顺序外 AST 一致，物理模块和其余输入不变。不能把原报告错误地声称为当前测试文件的同 SHA 报告；本次单独记录此差异，不改写原报告。

本轮此前已新增16800控制拍/84000物理子步；本合同若四场全跑完，累计20800/104000。未重跑三次65k训练和旧24场G1。77冻结源保持。完整“稳定直行→可靠越障→提高速度”仍未完成。
