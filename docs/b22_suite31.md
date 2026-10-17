# B22 当前环境七脚本与真实台阶配对验证 31

状态：新入口与合成测试已准备，**本轮尚未执行模型、物理或 GUI**。依据所有者目标“把这些优化点，在这台电脑能力之内的都完成”，先补当前环境下的完整固定矩阵，保留历史 C22/C24/C26、v27 与 closeout30 的输入和输出原字节。它不重复训练，也不以旧 B 样本和新 zero 样本拼接成精确配对。

已有能力不是从零开始：C22 已有七项 B 脚本资格与失败的 RL 贡献门，C24 的 zero/B 都越过真实 15 mm 碰撞箱体，C26 已完成真实 OS 键盘/失焦/输入过期/一次 reset；closeout30 重新通过当前 v27 的平地 1.6 m/s GUI 演示。本入口重用这些冻结数值规则，并新增一次性串行编排、流式保存数据读回与 B/zero 比较。脚本使用显式派生的新 worker31，仅改变版本身份与合法任务短终止的收尾缝隙，控制数学不变。

## 固定矩阵与预算

先运行一次原 C24 `step_pair`：同一新编译模型中 zero 与 B 各 1,200 控制拍。随后按固定顺序逐个运行七个脚本，每项 zero → B，各自使用新会话：`flat_0p6`、`flat_1p6`、`yaw_1p2`、`bumps_0p4`、`rough_0p35`、`ramp_0p45_complete` 和独立镜像 `yaw_1p2`。原六 profile 的种子为 181201…181206，镜像为 181403；坡道每 arm 1,800 拍，其余脚本每 arm 1,600 拍。

| 项目 | 固定上限 |
|---|---:|
| 逻辑 arm 记录 | 16 |
| worker 会话 | 15（C24 的两个 arm 共用一次编译） |
| 控制拍 | 25,200 |
| 正常 native 积分 | 126,000 |
| 编译 native 积分 | 30 |
| 训练、反传、保存 | 0 |
| 整体 wall / 单 stage / 单 reader | 3,600 s / 900 s / 180 s |
| RAM | 每 stage 启动 effective≥6 GiB，运行保留≥3 GiB |
| 可用磁盘 | 启动≥2 GiB，运行保留≥1 GiB |

资源检查是采样，不宣称独占或预留。旧 host/worker 的更短截止时间继续生效。资源、源码、工程或归档异常停止后续 stage，不自动重跑；合法的自然任务失败完整保留实际拍数与原惩罚，继续收齐预定矩阵，不补齐短回合。一个样本未达标不能靠修改原门槛变成合格。

## 入口

仅查看固定计划，不读取模型、不构造仿真器、不创建输出：

```bash
rtk proxy /usr/bin/python3 -B scripts/run_b22_suite31.py
```

实际执行须在全部新交付输入提交且根代理安排本机独立运行窗口后明确调用；当前没有执行该命令：

```bash
rtk proxy /usr/bin/python3 -B scripts/run_b22_suite31.py --execute \
  --output results/b22_suite31_20261018_01
```

输出必须是新建的 `results/b22_suite31_*` 直属目录，拒绝覆盖。本次 `release.json` 绑定固定矩阵、全部新交付输入的提交/SHA、原 v27 manifest 和核验通过的完整输入身份。七脚本通过原 host 启动新 `run_b22_worker31.py`（`C31_B22_fixed_suite_v1`）；台阶通过原 host 与 worker24 启动，内部 C24 字段表示保留的数值执行协议，外层 31 是新的运行和验收身份，不改写旧回执。

冻结 GUI26 的脚本一旦早于预算终止，会在建立最终 `result` 之前抛异常，无法把合法任务失败交给外层收齐配对。因此31显式复制其完整源码并保留原 SHA，以整模块 AST 回归锁住其它可执行语句：只换 C31/headless/script 的 preflight，只有 `terminated=true`、`truncated=false` 且原终止理由为 fall/nonwheel/map boundary 时正常完成原归档和计数；未知短截断或工程异常仍失败。不清旧 failure、不补造拍数。继承代码的少量 Ruff 风格豁免只用于保持 import/owner-cleanup AST 原样。

## 保存数据如何判定

新 reader 本身导入标准库；执行读回时仅加载 NumPy 与已固定 SHA 的纯函数，禁止 Torch、MuJoCo、GUI 和模型加载。原 C18 控制器、PI/z/停止记忆、五条 native 积分链、实际电机力矩、接触分类/载荷与完整状态端点逐项核对。正常原生归档每次只保留连续五行，不能把十余万条 JSON 先整体装进内存。C24 使用冻结 read24 数值与碰撞几何检查，只替换文件读取接口为单遍流式序列。

所有脚本的速度、偏航、完整最后一秒停车、释放后最大超行程、原生姿态/净空/非轮接触和地形载荷继续使用冻结 C18/08 评分数值。坡道还必须证明上坡、台面、下坡实际正载荷，以及全部四轮完整碰撞柱体越过坡道末端。

原 GUI worker 没有保存旧 C22 的 final projection 文件。31 不伪造该文件：先要求新 C24 保存的完整实际 `kinematics24` 编译绑定通过独立读回，再用冻结 `robot_binding_bridge24` 核对它与脚本的共同机器人字段及同源建模证据，从脚本最终 qpos 纯重建四轮柱体边界。这是来源约束下的独立运动学推导，不是脚本里直接测存的 geom cache，不调用模型或 `mj_forward`。

B/zero 配对要求初始五类数组的 dtype/shape/逐字节哈希与完整控制 reset 记忆相同，实际机器人/地形、seed、命令表和后端来源一致。原六项送入冻结 `score18.score_pairs`，实际计算 cost 门及两分支贡献门；不会硬编码 `RL_contribution_passed=false`。镜像与真实台阶逐项独列，不偷偷加入原六池。任务合格和 RL 收益分开输出，zero 同样通过时不能把整个系统成绩归于 RL。

## 不在 31 内的工作

键盘/失焦将使用单独的固定新预算；C++ actor 闭环需要新后端及同轨/数值验证；跳过关门拍推理的性能改动将保留为独立 32 版本，不能混入本次基线。B22 的 99D 已有四轮 PI，当前平地日志也没有有效动作/力矩饱和；未显式进入观测的 0.2 s common-reference z 仅是后续 100D 单变量对照候选，尚无收益或因果结论。

本机实现与有限预算不能保证策略通过 RL 贡献门，不授予统计泛化、实机或 sim-to-real。任何运行前合成测试和 CI 均不替代真实矩阵证据。

## 轻量验证与来源

新交付闭包包含四个脚本、两个测试文件与本文档。测试使用 fake 进程、少量数组或直接抽取的 AST；不导入旧 worker、不加载模型、也不执行积分。独立审计测试保留了旧入口对三类合法任务失败抛异常的反例，并验证新入口允许它们进入原归档/回执路径，未知短截断仍拒绝；这不等同于完整物理归档实测。

```bash
rtk proxy env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python3 -B -m pytest --noconftest -o addopts='' -q \
  tests/test_b22_suite31.py tests/test_b22_suite31_audit.py
rtk proxy .local-deps/bin/ruff check scripts/b22_suite31_contract.py scripts/run_b22_suite31.py \
  scripts/read_b22_suite31.py scripts/run_b22_worker31.py \
  tests/test_b22_suite31.py tests/test_b22_suite31_audit.py
```

若后续31确有贡献缺口，100D 候选应只新增当前未显式携带的 z，保留原99D顺序，并以新增首层列权重为零实现初始行为相同。必须另定同初始策略、同种子、同交互数的99D/100D训练与评测预算；31不启动这种训练，也不能从一个内部状态映射反例认定 z 是性能瓶颈。
