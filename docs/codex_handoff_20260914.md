# Codex 续接：稳定直行 → 可靠越障 → 提高速度

更新日期：2026-09-14。本文件是当前驾驶优化任务的交接入口，完整目标尚未完成。
先核对当前工作树、进程和 GitHub 状态，再根据本文件续接；较旧的本机 `task_plan.md` 含历史阶段，不能把那些阶段的 `complete` 当成本目标已完成。

## 目标与授权

用户要求改善真实 MuJoCo 驾驶体验，按稳定直行和停车、可靠越障、提高速度的顺序推进。
原始需求包括持续按键、A/D 用腿侧移、Q/E 调整航向、Space 跳跃、R 翻倒后恢复、Shift 速度换挡，以及坡道、碎石和台阶。
R 当前只是明确标注的仿真复位，物理自救尚未实现。人工试驾尚未通过，不能用自动按键记录代替人工验收。

用户已授权及时提交并上传 GitHub，明确希望由 **gpt-6-astra / ultra 定方案，实际调用 Claude Opus 编写实质代码**。
不要反复询问这些已获授权的动作，也不要把其他 Codex 子代理的输出称为 Claude。
用户要求本轮结束后转到新的 Codex 窗口继续；新窗口须继续完整目标。

## 本机路径与执行约束

- 仓库：`/home/lyh/wheel-legged-control-lab`，分支 `main`。
- 本轮工作目录：`/home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260914`。
- GitHub：`LYHrmer/wheel-legged-control-lab`。
- 每个 shell 命令使用 `rtk` 前缀，需要完整输出时用 `rtk proxy`。不创建或重建 CodeGraph 索引。
- `results/d1_budget_study/protocol.json` 的 `source_sha256` 冻结 **77** 份正式输入，包括核心 `src`、部分 runner 和配置。保持字节不变，通过新增模块/子类扩展；不得改旧模型的 sidecar 伪装兼容。
- 旧结果、正式研究、已有 release/tag 不改写。新实验写全新目录，失败和未收敛记录保留。
- 环境通常为 `PYTHONPATH=.local-deps:src:.`，`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- 大文件 HTTPS 推送曾重复失败。直接 SSH 推送已验证成功：`rtk git push git@github.com:LYHrmer/wheel-legged-control-lab.git HEAD:refs/heads/main`。不要重跑已废弃的临时分批推送序列。远端以 API/实际 refs 为准，直接 URL 推送可能不更新本地 `origin/main`。

实际 Opus 调用使用已有 wrapper：

```text
/home/lyh/wheel-legged-control-lab-work/recovery-20260912/run_opus_task.py
```

传入明确 prompt 文件、全新输出目录、预算、超时和 `--code-only`。第一次受沙箱网络限制失败后按工具规则申请网络重试，不绕过审批。读取实际 provider/model、exit code、原始响应和费用收据再报告成功；别只依据请求中的 `opus` 字样。

## 当前控制入口与已完成改进

### 课程试驾

入口为 `scripts/run_d1_course_drive.py`，使用经典 LQR/VMC 和课程地形，支持六个区域。
Shift 前进档为 .30/.40/.50 m/s，倒车档为 .20/.35/.50 m/s。Space 有现有受限跳跃；R 是仿真复位。

A/D 输入默认已改成 `commit_cycle_v1`：短按请求完整一步，松键完成当前周期，长按才继续下一步，X 请求安全取消。可显式选择 `--side-input-mode release_abort` 对照旧行为。
实测短按 1 秒后约 10.56 秒完成约 3.70 cm，交接 4 秒保留约 98.3% 位移。输入修复没有加快步态。
十场、21,900 个实际控制转移验证短按、重复、反向、取消、跳跃互斥和复位释放门禁；中止记录仍标成取消，没有冒称侧步成功。

- [操作说明](side_step_input.md)
- [公开完整证据](../results/d1_driving_stability_development/side_input_commit/README.md)
- 新核心：`scripts/d1_course_side_intent.py`，实际 Opus 实现，原稿与修正分别保留。

### 新航向强化学习任务

`scripts/d1_heading_tracking_env.py` 继承原同步任务，基座是 `wheel_leg` 的关节 PD、轮速 PI 与偏航反馈，**不是课程 GUI 的 LQR/VMC**。
新任务保留原 82 维 servo 观察，追加航向误差 sin/cos 和用户 yaw 命令，合计 85 维；动作仍为八维残差。
航向 P/D 外环固定为 2 / .4，偏航输出上限 1 rad/s，实际已执行的用户命令和 servo 命令分开记录。不能在 GUI 中再次叠加航向 P/D。
`load_heading_policy` 校验新任务配置及模型身份；旧 82 维权重不可直接加载。

三个种子 49001/49002/49003 各连续训练 65,536 步，总计 196,608 个训练转移，16k/65k checkpoint 均在 PPO 完整更新后保存并实际重载。
三份 65k 模型都完成 32 秒开发道路。零残差同样完成，航向误差更小；49003 虽速度误差更小，但多个腿残差长期饱和，不能据此直接设为默认。

- [训练、六个模型和完整评测](../results/d1_driving_stability_development/heading_learning_01/README.md)
- [总体优化说明](driving_optimization.md)

## 本轮最新失败决定下一步

固定权重的 G1 开发探针已完成 24/24 场、22,400 控制转移（112,000 物理子步），所有场景都跑满时间。
四个控制器各接受正向停车、倒车停车、左右原地转向保持、正负偏航冲击六项检查。
完整案例通过数：zero 2/6、49001 2/6、49002 0/6、49003 2/6。通过域仅为两个直行偏航冲击，不能宣称完整稳定驾驶通过。
全部正/反停车失败；zero 左右转向只有瞬态航向峰值未过，PPO 转向还普遍存在晚期速度/位移问题。
8 个受扰回合均实际施加 ±0.1 N·m·s，原始轨迹与失败门槛全部保留。

- [G1 公开记录](../results/d1_driving_stability_development/heading_g1_01/README.md)
- Runner：`scripts/probe_d1_heading_g1.py`。
- 工作协议：`heading_g1_protocol.json` rev2，SHA256 `cf5dbd51042bfafa163116f4a7fb2a990726d47c5a6000558f28aa6866a664fc`。
- 模型位于公开 `heading_learning_01/training/`；runner 的 `--models-root` 支持重定位并验证冻结 SHA。
- 扰动必须走原 `plant.step(push_torque_world_nm=...)`，原 plant 每个子步清空直接外力写入；不能把被清掉的 `xfrc_applied` 当实际施加成功。

**下一控制任务先定位零残差停车问题。** 停令从 tick 400 起，用户/servo forward 都为零，八维动作全零；不能把全部失败归因于 PPO。
zero 正停车晚段最大速度为 .06087 m/s，指定两秒累计路径 .05669 m，未满足 .03 m/s 与 .05 m 的原门槛。
基座是轮速 PI，应检查轮速积分、机身俯仰与整机速度反馈；旧课程 LQR 的距离参考干预不能直接当成这里的根因。
原门槛不得因失败而改宽；新修复使用独立版本和配对实测，先不追加无依据 PPO 训练预算。

下一窗口直接读[中文只读诊断](../results/d1_driving_stability_development/heading_stop_diagnosis_01/heading_zero_stop_readonly.md)和[Astra 单变量契约](../results/d1_driving_stability_development/heading_stop_diagnosis_01/heading_zero_stop_next_contract.md)。两个独立复算都支持骤停激振假设，尚未证明唯一根因。
契约要求实际 Opus 实现固定 .5 m/s² 的释放参考尾段，只做 zero 的四个原 G1 命令、两条件配对，最多 8,000 控制步；仍按 t=4s 已归零的原始用户命令评分，尾段参考另记。两项无停止的冲击案例应逐位不变，停止案例在 tick400 前应逐位一致。若失败保留结果，不改阈值或扫参数。

旧课程制动的完整/半重锚和动态制动→锁定候选均有坡上退化，未设为默认。
[动态候选](../results/d1_driving_stability_development/brake_dynamic_candidate/README.md)的 12 次运行共 26,400 控制步，在一个配对坡面案例中回退从 2.85 cm 增至 41.64 cm。

## 后续两阶段不能遗漏

1. 跳跃/越障：现有跳跃机身上升 60.86 mm，但四轮同一时刻最小轮底净空仅 **5.61 mm**。先增强实际起跳和落稳，再逐级验证小台阶、坡面、碎石；以净空、持续离地和落地状态验收，不以动画阶段完成或机身升高替代。
2. 提速：在稳定启停与越障能力通过后逐档验证实际速度、制动距离、卡滞/翻倒与通过率。已有 Shift 换挡只证明输入和若干平地案例，不能代替全目标。
3. 物理自救：R 当前是 simulator reset；如实现翻倒自救，必须独立证明物理恢复，不得改名冒充。

## 新 GUI 候选的交接边界

Astra ultra 已冻结 `heading_gui_contract.md`，委托实际 Opus 编写 WORK 内候选，预期为新增 `scripts/run_d1_heading_drive.py`，本轮不会仓促改变已运行评测的控制链。
第一次调用 `heading_gui_opus_01` 因网络失败，0 tokens/$0；重试输出为 `heading_gui_opus_02`。新窗口先读取最终收据和候选检查记录，不能假定请求已成功。
候选设计：每 run 显式选择 zero 或可信 PPO checkpoint；直接消费 85 维环境原始 W/S、Q/E 命令；X 只清除请求，不能称为已验证的可靠停车；R 重新分段；终止后保留窗口等待 R。
A/D/Space 在该新后端尚不支持，应明确引导原课程入口。使用 `KeyboardViewer` 的独立 model/data，不让渲染或键盘轮询增加物理步。
后续集成须独立验证逐拍轨迹、终止/复位、日志 T/T+1 和真实窗口，再邀请人工试驾。

## 新窗口启动顺序

1. 核对 `git status`、实际 HEAD、远端 main 与 CI；读取本文件和链接的最新证据。先检查仍存活的进程，不因观察超时重复启动任务。
2. 读取本轮停车只读诊断及 Astra 方案，验证支持它们的现有数据。确定一个最小控制改动，交给实际 Opus 编码，主代理独立集成和物理验证。
3. 若接入 GUI，固定模型身份且明确支持范围，验证渲染不影响控制轨迹。不能把只会当前直行命令的 PPO 当作已会侧步/跳跃/启停的万能策略。
4. 完成停车和转向门槛后继续可靠越障、提高速度。持续将可审查的完成批次上传 GitHub。

本轮本机全套 **3,314** 项测试通过，零失败、错误和跳过；Ruff 通过，77 份冻结输入 SHA 全部一致。[完整测试与交付校验](../results/d1_driving_stability_development/integration_verification_02/summary.json)保留 XML、源码 SHA、公开包 manifest 与压缩 blob 校验。该证据证明程序回归通过，不证明停车、越障或完整驾驶目标已完成。

最终提交与云端 CI 应以实际 GitHub 状态为准；GUI 调用终态先读本机 WORK 的收据及静态审阅记录。本文件不把未完成的控制能力标成已完成。
