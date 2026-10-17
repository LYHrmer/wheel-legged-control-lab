# 当前环境键盘与失焦验证 33

状态：仅准备新入口与纯测试，**尚未启动本轮模型、物理、GUI 或 X11 服务**。本轮依据所有者目标“把这些优化点，在这台电脑能力之内的都完成”，准备在新目录复验已有 C26 自动交互能力；旧 C23/C25 失败、C26 通过、v27 与 closeout30 的输入输出均保留原字节。

33 只新增串行监督，不复制控制器或数值判定器。原 `run_gui26.py` 和 host23 按原 `C26_B22_GUI_owner_close_v1` 执行，原 `reader26_deferred.py` 直接用新的 `--run` / `--output` 读回；内部 C26 身份表示控制、键盘和判定语义未改，外层 release/report 的33身份表示新的来源、授权与运行。历史 GO 只提供已固定事件表和隔离依赖证据，不充当新运行授权。

## 固定范围

沿用 B22 SHA `7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691`、99D/16D、C18 combined、`yaw_1p2`、seed231002。全会话最多 1,200 控制拍 / 6,000 正常 native / 2 compiler；R 复位一次后仍累计同一个全局上限，不退还预算。模型最多 load1 / torch_load3 / predict1201，包含一次32行固定探针；训练、反传、保存、critic调用均为零。

原27次 OS 注入逐条不变：W 前进、Q/E 双向转向、Space 和 X 停止、鼠标 Stop、失焦/回焦/重新按键、R 复位、第二次 Reset 按钮被拒、Esc 退出。tick530 开始停止输入快照发布0.35秒，OS轮询和绘制继续，用于验证真实过期快照消费。tick510 的 W 松键保留为一次 OS 清理；26个有意义回调必须按原时钟界限一对一匹配。请求、注入或 callback 成功不等于控制拍已消费，原 reader 还核对 owner 准备记录、当拍99D观测、raw/servo命令及全部5T native链。

原活动性能与停车门不变：RTF≥0.8、绘制≥8FPS、轮询与显示年龄p95≤250ms；末100实际点 raw/servo 前向和偏航全零，平均绝对前向≤0.04m/s、偏航≤0.05rad/s、净空标准差≤0.02m。原95s soft / 115s绝对owner close / 120s worker hard / 600s外层host均保留。新监督总 wall≤900s（含来源检查、读取器和清理），独立保存数据读取器≤180s。每次模型启动 effective RAM≥6GiB，整个运行采样保留≥4GiB；磁盘启动≥2GiB、运行≥1GiB。采样不等于预留或独占。

## 独立 X11 和来源

XTest 原驱动只允许独立`:100..:199`服务，不能向用户桌面注入按键。复用仍存在的本机历史隔离脚本和已解包 Xvfb，不安装依赖，也不修改它们：

| 文件 | 固定 SHA256 |
|---|---|
| `/home/lyh/wheel-legged-control-lab-work/recovery-20260912/gui_isolation/isolated_x11.py` | `33dceb3402f2a73c664e9f4cfefff67423cf117c78ba588ae43b31eb3bc8e9c4` |
| 同目录 `rootfs/usr/bin/Xvfb` | `cf71d78eab8321f7c568271053af449f966a75422140117ea1f244bf72a7e81e` |

这两个身份来自保留的 C26 GO（895449字节，SHA `f819a94455a5561eed702545e53d901cb301ed712f904434acf4a5beb77e1315`）。新闭包包含完整v27输入、该事件来源、上述两个隔离依赖、已提交31的memory/监督helper、以及本轮脚本/测试/本文档；源码须提交后执行。当前 libX11 等环境差异沿用已审 v27 登记，不照搬旧 GO 的过时环境哈希。此入口仍依赖上述本机路径，未声称任意机器可移植。

隔离服务禁用TCP、使用独立认证文件和软件渲染，完成后须停止自有服务并删除认证文件。监督在启动前启用 subreaper，沿用31已测的祖先关系/出生tick清理，异常不误杀外部任务。任何工程、资源或来源异常保留现场并停止，不覆盖、不自动重试。合法且完整的原判定门失败可保留为 `status=COMPLETE, qualification_passed=false`；COMPLETE只说明收集和读取完成。

## 入口和当前验证

默认仅列计划，不读模型、不创建输出：

```bash
rtk proxy /usr/bin/python3 -B scripts/run_b22_interaction33.py
```

实际执行须等待根代理安排唯一重任务窗口；当前没有执行：

```bash
rtk proxy /usr/bin/python3 -B scripts/run_b22_interaction33.py --execute \
  --output results/b22_interaction33_20261018_01
```

输出为新的直属`results/b22_interaction33_*`目录：`release.json`、`launch/`、原生`keyboard_gui/`、隔离`keyboard_gui.x11/`、原读取器`readback.json`、资源日志和外层`report.json`。截图位于`keyboard_gui/frame_{initial,drive,final}.png`，须另行实际查看；脚本不会把未进行的人工目视验收写为完成。

纯测试只读小JSON、检查新请求、fake回执、路径与AST，不导入worker/旧reader或重型库：

```bash
rtk proxy env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python3 -B -m pytest --noconftest -o addopts='' -q tests/test_b22_interaction33.py
rtk proxy .local-deps/bin/ruff check scripts/run_b22_interaction33.py tests/test_b22_interaction33.py
```

准备阶段56项纯测试通过，Ruff通过，包含实际资源采样器在4GiB运行线和6GiB启动线两侧的边界检查。另将9份必要源码/小JSON复制到独立临时checkout路径，56项再次通过；未复制模型、环境、轨迹或外部X11依赖。计划和纯测试支持不同checkout路径，实际`--execute`明确限于已登记的本机`/home/lyh/wheel-legged-control-lab`，不会将CI路径伪装为本机环境通过。

这是原自动 XTest/GLFW/owner 链在当前环境的一次有界复验，不是人工键盘试驾，不验证任意驾驶、多种子泛化、RL优越性或实机能力。31负责当前多任务/镜像/台阶配对，32负责独立推理性能对照；33不会混用它们的物理样本或预算。
