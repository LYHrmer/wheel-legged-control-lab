# 07 普通手动 GUI 启动静态核对

结论：未发现阻止当前机器从 `/home/lyh` 执行 `rtk proxy d1-rl` 的代码路径问题。本次只读源码和已闭合的 A 场验证记录；没有启动 GUI、导入 MuJoCo 或运行物理测试。当前 shell 的 `DISPLAY=:1`，`XAUTHORITY=/run/user/1000/gdm/Xauthority` 且该文件存在；实际桌面连接仍需用户启动时验证。

- `/home/lyh/.local/bin/d1-rl` 权限 755，位于当前 `PATH` 的 `/home/lyh/.local/bin`，用 `/usr/bin/python3 -B` 按绝对路径执行仓库 `scripts/play_d1_latest_rl.py`。launcher 按自身 `__file__` 定位仓库，所以从 `/home/lyh` 运行不依赖当前目录。默认 `--actor rl --terrain box --speed .20`；worker 严格加载发布包中的最终 checkpoint，再安装 06 body-speed controller，零残差只通过显式 `--actor zero` 选择。
- 普通 `manual` 分支的 `freeze_path` 和 `freeze_sha256` 均为 `null`。767 项旧工作区冻结文件只在验证分支重哈希；普通启动仍严格核对仓库内 05/06 发布包 manifest、checkpoint、隔离 DSO、运行源码和当前依赖身份。普通输出在仓库 `results/d1_latest_rl_sessions/<unique>/`，最多两个各 1200 control 的段，整体 2400 control、12000 native step、3 compiler hidden step，900 秒墙钟限制。
- 窗口聚焦并按住 W 才提出前进请求；1/2 选择 0.20/0.25 m/s，Space/X 请求停止，松开 W 后方可重新前进。R 在第一段结束或提前结束时做至多一次模拟 reset；提前 reset 不退还第一段剩余预算。第二段预算耗尽后自动保存退出；Esc、关闭窗口和父进程 SIGTERM 也走保存边界。单次 reset 使用同一模型/数据地址，重新核对五种初态数组。
- A 场 `gui_policy_box_01` 的 launcher `exit_code=0`、无源码 hash 不一致，worker 记录 1200 control、6000 native、3 compiler，`execution_complete=true`；键盘验证记录的 W、2、Space、R、Esc 各边沿均通过，渲染记录 1202 帧。A 场使用私有 Xvfb 和脚本事件，所以证明同一 GUI/键盘路径可运行，不直接证明当前用户桌面的焦点、Xauthority 或人工操作体验。

仓库 `docs/latest_rl_gui.md` 首行仍写“awaiting 07 verification”，现在与 A 场已通过的记录不符；这是发布文案待更新项，不影响命令运行。B 场或普通人工会话尚不能由本次静态核对宣称通过。
