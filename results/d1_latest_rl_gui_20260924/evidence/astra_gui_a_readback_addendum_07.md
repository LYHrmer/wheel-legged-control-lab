# GUI07 A 归档复核及纯读程序修正审阅

实际 GPT-6-astra ultra 独立只读复审；未导入或调用 MuJoCo、策略、模型或测试。A 已完成 1200 控制 / 6000 正常原生步 + 3 编译器原生步，实际在线策略调用/返回均 1200。

A 与已合格的 06 `.20 box` 在共同输入范围完全一致：控制 tick 0–400 的全部旧 trace 字段（含 8 个旧 raw-command 字段）、endpoint 0–401、native 0–2004 均逐值相同；qpos/qvel/ctrl/warmstart/time 前 402 行逐字节相同，策略输入 observation 前 401 行逐字节相同。唯一有意排除的后继 observation[401] 已在 poll400 换挡后准备，38/56–59 维随 .20→.25 请求改变；物理 endpoint[401] 仍是相同 tick400 的返回。详见 `astra_gui_prefix_readback_07.json`（SHA256 931dfadd2ee139d4b923508e2402c482faf2f19216a955bfce38f59a34403f40）。

已亲自查看 3 张真实 PNG：2.0s 的 .20 挡、6.5s 的 .25 挡和 R 后初态。画面显示 robot/15mm box、oracle、真实 checkpoint 前缀、learned residual ON、06 controller ON、engine bound、观察速度/误差和计数。R 后 time=0、request=0，但累计 control/native/compiler 仍为 1000/5000/3。保存的 1202 个 render guards 均保持 actual integrator 和 measurement 数据，C/native 与 Python 计数增量均为零。显示的完整绝对归档路径超出画幅；完整路径在 session 记录中，非运行阻点。

原 GO 绑定 `verify_gui_session_07.py` 保持不变。新 `verify_gui_session_07_readback02.py`（SHA256 a308613f8246d0419a93be74c4c94ef6269f736364845df3f088ae1ec2cb1f15）仅修正三项纯读接口：使用明确文件路径导入纯 integrity 模块以避开第三方同名 scripts 包；读实际 `episode_metadata` 嵌套对象；匹配真实 `two_saved_probe_observations_checked_in_one_batch` 字段。已检查完整 diff 和被导入模块，无算法、验收阈值、旧归档或生产执行源码变更；允许根代理执行该版离线 readback，失败旧版保留。根代理随后报告新版本 A readback exit0/pass；B 可按已冻结顺序及新进程独立额度执行。

该共同前缀证据说明新的 UI/渲染接入在相同输入下保持已验证闭环。它不证明 RL 相对收益，不将两个不同地形的 GUI/headless profiles 当成因果对照，不扩展旧 06 固定两速度越障资格。B 尚待真实完成与独立读取；不追加任何物理运行。
