# 两任务残差幅度对照 34

状态：仅准备源码与纯测试；**未执行本轮网络、模型、物理或训练**。不修改31/32/33的源码、历史回执或已冻结输入；新源码未提交时不能执行。

31在当前环境完成了固定矩阵与保存数据独立读回。选定的平地1.6m/s与偏航1.2m/s任务，zero和B均完整通过，但B相对zero的总SSE比分别为1.16872018、1.03400797，归一化力矩平方成本比分别为1.07362083、1.06482796。它们支持一个可证伪的局部假设：当前残差幅度可能偏大，降低幅度是否能减少这两个任务的误差和成本。它们不证明原因，也不能说明缩放一定有收益。

## 准入和预算

34必须核对完整31：外层与comparison均工程COMPLETE，原六配对`prerequisites.passed=true`且原RL贡献门为false；本次两个profile各自zero/B任务通过且full_horizon，两者各自B总SSE与成本都高于zero。其它地形是否全部任务通过不额外决定这两个任务的局部探针。31 release固定为857698字节/SHA`17c6926c731015e06174a592427546f0a69d7614830e5242a63b248b1caa6310`，最终comparison为SHA`f5dd040d05fe2a7b19811763ffc0b45fe6ab5c201a25a4da249f3431f6822456`；实际执行还须将外层、comparison及全部15份readback身份绑定到新release。

固定四个新会话，顺序为flat_1p6 gain1→gain0.5，yaw_1p2 gain1→gain0.5，seed分别181202/181203。每个1600控制拍，合计最多6400 control / 32000 normal native / 8 compiler；训练为零。每会话最多load1 / torch_load3 / predict1601（包含原32行探针）。启动effective RAM≥6GiB，运行保留≥4GiB；磁盘启动≥2GiB、运行≥1GiB；整体wall≤900s，原worker180/235/240s更短门保留。采样不等于资源预留。工程、来源或资源错误停止，不重试；合法任务失败保留实际计数和原惩罚。

全部四份数据均新采集。同一profile的gain1/gain0.5必须五类初态数组的dtype/shape/字节以及完整控制reset相同，并固定模型、环境、命令、频率、后端与数值规则。旧31的zero只提供触发和诊断锚，不能伪称本次新配对，也不拿旧gain1数据替换新会话。

## 唯一控制变量与读回

新worker34明确从固定worker31 SHA`49feaecf2c6e68cf39c99238d5bde04b88dafb10de72479d0f3ce5c999e5d7c0`派生。全模块AST回归只允许C34身份/两profile/float gain准入、网络输出到实际控制输入的float32乘法，以及新增ledger；其余原控制、观测、奖励、停止、5T native和归档路径逐语句不变。两个增益均调用同一冻结B22确定性策略，不混入32的关门省推理优化。

`network_output16`记录原SB3 clipped确定性网络输出，`gain34`记录1.0或0.5。实际发送给控制器的是float32的`gain34 * network_output16`；`raw_action16`、`action16`、`policy_input_action`都保持为这个实际输入，避免旧reader把网络输出误当实际动作。gain1的乘法需逐字节保留原float32输入。没有新增clip或放宽阈值。

新reader对每一实际拍检查上述缩放、16D有穷值和ledger计数。完整固定SHA核验后，只把31纯来源检查函数中的两个身份字面量明确改为C34/worker34，不能将回执伪装为C31；控制器、PI/z/停止记忆、状态、全部native/contact/force检查直接调用未改的31纯canonical函数及原C18评分器。不复制物理数学，不加载模型或MuJoCo。

必须四回合均完整才能给两个任务的完整时域相对SSE/成本比；任一短回合使这些比率全部为null，保留原任务失败与单项数值。完整但任务失败时也不能输出“局部任务通过且误差成本都更低”。成本是归一化力矩平方proxy，不是能耗。即使两项都降低，也只报告局部候选，不自动采用，不声称六任务RL贡献门通过、多种子统计显著性或其它地形收益。

## 入口

默认只列计划，零模型/物理且不创建输出：

```bash
rtk proxy /usr/bin/python3 -B scripts/run_b22_gain34.py
rtk proxy /usr/bin/python3 -B scripts/run_b22_gain34.py --check-trigger
```

实际运行仍待根代理审阅、提交新源码并安排唯一资源窗口；当前未执行：

```bash
rtk proxy /usr/bin/python3 -B scripts/run_b22_gain34.py --execute \
  --output results/b22_gain34_20261018_01
```

纯验证命令，不导入真实worker或模型：

```bash
rtk proxy env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python3 -B -m pytest --noconftest -o addopts='' -q tests/test_b22_gain34.py
rtk proxy .local-deps/bin/ruff check scripts/run_b22_gain34.py scripts/run_b22_worker34.py \
  scripts/read_b22_gain34.py tests/test_b22_gain34.py
```

准备阶段65项纯测试及Ruff通过，包含实际worker动作缝隙的fake predictor、gain1普通/符号零/次正规数逐字节保持、实际gain准入与6/4GiB采样门、原模块AST锁、缩放ledger篡改拒绝及短回合不得冒改善。复制23份小源码/manifest到临时checkout后65项再次通过，没有复制或加载模型、依赖和轨迹。真实31的零模型`--check-trigger`已核实READY并固定18份小JSON身份；这只是允许准备的事实，不是34已运行或已证明收益。

完整元数据检查也已功能验证，而不只检查AST字面量：两个C34增益fixture调用真实`case_completion`与冻结的归档/`check_ledgers`函数均通过；旧C31标签和错误native计数反例被拒。`case_completion`本身只核共同回执结构、合法terminal和计数，不限制C31合同，C34合同由新适配后的来源函数显式核对。另一次轻量探针在明确的合成C34来源/会话fixture中置入已核SHA的真实31 worker result，完整元数据检查仍返回1600控制拍；它不产生或冒充新的C34物理回执。

外层在继续下一会话之前核对每份C34 reader的合同、gain、release/reader身份、实际control/native/compiler数与gain ledger行数；最终comparison的四份readback SHA必须与外层逐阶段收集的身份完全相同。subreaper在全部受监督启动前启用，异常清理沿用已固定31的祖先关系/出生tick机制，收尾异常保留INCOMPLETE回执。
