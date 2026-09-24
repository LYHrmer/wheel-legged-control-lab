# GUI07 最终有界验收：通过

实际 GPT-6-astra ultra 指挥与独立复审。新的试驾入口已经真实连接 65,536-transition PPO final checkpoint、06 控制律和实际修复的 MuJoCo 引擎；本轮没有训练、调参、复用旧物理配额或重跑旧资格四场。验收物理全部由 root 唯一执行，本审阅只读归档与图片，未导入/执行引擎、策略或测试。

从仓库目录启动：

```bash
rtk proxy python3 scripts/play_d1_latest_rl.py
```

默认是 RL + 15mm box。启动时可选 `--actor zero` 或 `--terrain plane`；zero 仅关闭学习残差，06 控制器仍运行。W 按住前进，1/2 选择 0.20/0.25 m/s，松 W 或 Space/X 停止请求，R 为一次模拟重置，Esc 保存退出。改变命令在正常 prepare 链下一控制 tick 生效，首次175tick稳定期仍优先；actor/terrain 不在运行中切换。手动会话最多两段各1200控制、总2400/12000正常子步，早重置不返还未用额度。

## 已核验的结果

| 新验收 profile | 真实控制 / 正常子步 / 编译子步 | 实际在线策略调用 / 返回 | 结果 |
| --- | --- | --- | --- |
| GUI + final policy + box | 1200 / 6000 / 3 | 1200 / 1200 | 完整执行与独立归档读取通过；626+25个前进行动使用非零共享腿残差 |
| Headless + zero residual + plane | 1200 / 6000 / 3 | 0 / 0 | 完整执行与独立归档读取通过；所有学习残差为零 |

每场均为1000控制→真实R重置→200控制→Esc关闭。合计两个新进程、2400控制、12000正常原生步、6显式编译器原生步，预算全部闭合，不授权进一步 rollout。C 与 Python attempted/returned、5T、monitor 数一致；警告、数值失败、越权、输入变化和遗留子进程均零，结束phase/target关闭。GUI实际CCD为23569、plane为0；后者是实际解析平面路径，不是漏计或虚构convex caller。

Root 使用原06完整stage/PI/torque validator与原geometry integrity，对每段全部原生接触、端点、状态、动作/扭矩链及准备命令独立读取；all-time姿态/横偏/非轮接触等07门通过。A在箱上出现真实正轮载荷。按瞬时载荷逐样本求和的417213.74 N仅是归档汇总量，不应称为持续支撑力或冲量。换挡轨迹分区速度统计是描述量，不冒用旧601点恒速RMS资格门。

我另独立核对 A 的共同输入前缀：401条控制trace全部旧字段、402条endpoint、2005条native与合格06 `.20 box` 逐值相同；对应物理数组402行及实际策略输入401行逐字节相同。index401后继观测因新的 .25 prepare 改变五维，这是预期一tick命令边界。详见 `astra_gui_prefix_readback_07.json`。

GUI真实渲染1202帧、保存3张PNG，已亲眼检查两档前进与重置画面。所有渲染帧的actual/measurement状态不变、C与Python计数零额外增量；R后的时间0、请求0，而总计数仍1000/5000/3。画面显示oracle、真实checkpoint、残差/06控制器使用状态、准备目标/观测COM速度/误差与剩余额度。验收输入经私有Xvfb中真实GLFW窗口的XSendEvent输入，明确为脚本注入，不称为人工用户测试。

## 证据与真实性

最终复核 `astra_gui_final_acceptance_checks_07.json` SHA256 **1a49d4da9a3a6671ca8d529ebb890e0714e9a6ab3deca47d7ae725ce63b95527**：重新读取 GO 的51项输入和两个完整关闭manifest（54+44文件）全部hash一致；直接核C/Python/worker/launcher/清理收据，并引用root两次纯readback通过结果。原worker/manifest生成时的 `independent_readback_pending` 保留原样，本新增审阅明确完成后续独立读取，不回写旧证据。

原GO版verifier首次纯导入失败，无物理增量。新 `verify_gui_session_07_readback02.py` 仅修正同名scripts导入冲突及两处实际字段路径/名称；旧版保持不变，没有改验收阈值或运行源码。独立审阅见 `astra_gui_a_readback_addendum_07.md`。

实际first-party Claude Opus 5编写8个controls和2个渲染纯测试，root补2个接口/预算测试；两轮各12通过，共24次case执行，用尽本地纯测试预算。实际provider调用日志、原稿、root的AST命名空间适配和最终执行hash分别留存于 `actual_claude_contribution_final_07.json`，不伪装其他模型为Claude。既有GitHub CI若随后运行，须单独报告，不能追加到本地零引擎声明。

## 推荐发布范围与未完成项

接受并发布本机固定Python3.10/MuJoCo3.12 ABI与已固定ELF的可试驾入口、确切源文件/模型/引擎来源、两场完整原始记录、截图、纯测试及独立readback。无需再加物理演示或再次训练来完成07。此刻发布上传与远端CI由root另行收尾，不能预称已成功。

07证明最新闭环真正接入GUI。06仍单独支撑固定plane/15mm box、0.20/0.25m/s的确定性资格；原RL已经训练完成，但相对zero独立增益没有证据，06速度改善不能归功RL。本轮两个不同地形profile也不是RL因果对照。保留oracle辅助、不能转弯/倒车/跳跃、R不是物理自救、没有跨场景泛化或真机安全验证的范围。

GUI A的12秒模拟实际耗时130.25秒，headless B耗时27.39秒（含初始化/记录和脚本传输）；不声称达到实时100Hz墙钟性能，手动桌面流畅度仍未做人工测量。完整输出路径在当前截图画幅右侧截断，但完整保存在session；这是可用性限制，不影响本次真实控制/记录验收。后续如需性能改善，应另行基于计时剖析，不在本轮追加rollout或改已验证链。
