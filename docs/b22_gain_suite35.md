# C35：半增益完整任务配对验证（准备）

所有者目标为“把这些优化点，在这台电脑能力之内的都完成”。真实34的四个新会话均完整通过两个任务；gain0.5相对gain1在两profile的SSE和力矩cost均下降。这只是局部筛查，不能据此采用半增益或声称全任务RL贡献。35用新的zero/gain0.5完整配对回答下一步问题，不重跑gain1，不训练，不自动改默认模型或控制器。

15个新物理会话含：一个真实15mm台阶zero/B共享编译配对（两逻辑回合各1200拍），以及原31七组脚本zero/B配对。七组保留flat_0p6、flat_1p6、yaw_1p2、bumps_0p4、rough_0p35、ramp_0p45_complete及镜像yaw；种子、命令、停车和native窗口、完整坡道通过证明、原六任务RL贡献门均不变。B表示同一B22最终权重经float32乘0.5，zero没有网络加载或预测。台阶共享进程先严格加载B并完成原32row验证，zero回合仍不预测，B回合的predict及所有禁用API次数按原C24逐项核验。

最大25200控制拍、126000正常native、30编译native，16逻辑回合/15会话；总墙钟1800s，单host300s，独立saved-only reader180s，末尾保留清理时间。启动有效RAM至少6GiB，运行保留4GiB；磁盘启动2GiB、运行1GiB。执行入口检查实际nice至少10，需外层systemd设置Nice=10及全局串行调度，计划字段本身不设置调度优先级。真实31同规模约1151s（19.2min），只是本机历史成本参考，时间或通过率没有保证。

新增五文件为[父入口](../scripts/run_b22_gain_suite35.py)、[实际worker入口](../scripts/run_b22_worker35.py)、[独立reader](../scripts/read_b22_gain_suite35.py)、[纯测试](../tests/test_b22_gain_suite35.py)与本文。默认`rtk proxy /usr/bin/python3 -B scripts/run_b22_gain_suite35.py`只输出PLAN_ONLY，不导入模型/物理依赖、不创建结果目录。源码提交后，新的独占资源窗口才可在外层有界单元内运行：

```bash
rtk proxy /usr/bin/python3 -B scripts/run_b22_gain_suite35.py --execute --output /home/lyh/wheel-legged-control-lab/results/b22_gain_suite35_20261018_01
```

启动需固定31/34的release、outer、comparison及所有19份readback原SHA和exact bytes；35release绑定这些只读触发、v27完整输入、新五文件和明确使用的旧源码SHA。每次GO指向新worker35实际入口，所有新回执显式C35；不能给旧C31/C34记录重贴身份。worker35逐字钉住旧34脚本owner与C24的worker/eval/record源，运行前保存派生AST摘要。只扩展C35准入和完整profile，新增网络原输出、gain35和实际controller输入ledger；台阶亦逐行独立核`actual=float32(network)*float32(.5)`。其余物理、控制、奖励、5T时钟、终止、存档及清理逻辑由全模块AST回归锁定。原文件和原结果不修改。

reader只读已存轨迹，不加载模型或调用物理API；沿用原31流式native/C24全部几何接触门和C18评分。完整六任务贡献判定只用本轮新zero/半增益配对；镜像和台阶另外完整列出。原31仅供同任务通过标志的历史锚点，不能冒充本轮配对。报告区分任务资格、原六RL贡献、相对历史任务退化与是否自动采用（固定否）；保留所有单项及失败原因。短回合不能因累计误差较低而改善，原完整窗口与缺失惩罚不变。任务自然失败仍继续预定矩阵，工程/来源/资源错误停止，不重试；未知最终物理计数为null，可信自有回执前缀单列，不能把缺回执当零。

本文是准备范围，不是35真实结果。尚未启动35模型/物理/GUI；需源码提交、独立审查和根代理按当前资源串行安排后，才能形成新的实际验收结论。
