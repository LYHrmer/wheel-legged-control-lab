"""Render the publication from actual independently verified C18 scores."""
from pathlib import Path
import json

C = Path(__file__).resolve().parent.parent
R = Path('/home/lyh/wheel-legged-control-lab')
PACKAGE = R / 'results/d1_main_route_execution_20260930_17_18'
PREFIX = '../results/d1_main_route_execution_20260930_17_18/'


def main():
    summary = json.loads((C / 'qualification_summary_18.json').read_text())
    primary = json.loads((C / 'qualification_primary_readback_18.json').read_text())
    copied = json.loads((PACKAGE / 'copy_receipt.json').read_text())
    if summary['qualification_passed'] is not True:
        raise ValueError('cannot render a successful qualification report')
    gates = summary['task_gates']
    rl = summary['primary_six_task_RL_contribution']
    branch = rl['branch_b_sse_improvement']
    cost = rl['cost_gate']
    cases = [('primary', 'flat_0p6', '平地 0.6 m/s'),
             ('primary', 'flat_1p6', '平地 1.6 m/s'),
             ('primary', 'flat_1p2_yaw', '1.2 m/s + 原转向脚本'),
             ('primary', 'bumps_0p4', '凸起 0.4 m/s'),
             ('primary', 'rough_0p35', '粗糙地面 0.35 m/s'),
             ('primary', 'ramp_0p45_complete', '完整坡道 0.45 m/s'),
             ('mirror', 'flat_1p2_yaw', '1.2 m/s + 镜像转向')]
    table = ['| 固定任务 | 零残差 | RL | RL 均速 m/s | 速度 RMS m/s | yaw RMS rad/s |',
             '|---|---|---|---:|---:|---:|']
    for arm, case, label in cases:
        policy, zero = gates[arm][case]['grouped_continue'], gates[arm][case]['zero']
        yaw = '—' if policy['hold_yaw_rms_rps'] is None else f"{policy['hold_yaw_rms_rps']:.6f}"
        table.append(f"| {label} | {'通过' if zero['task_passed'] else '失败'} | {'通过' if policy['task_passed'] else '失败'} | {policy['hold_mean_com_vx_mps']:.6f} | {policy['hold_vx_rms_mps']:.6f} | {yaw} |")
    table = '\n'.join(table)
    fast = next(row for row in primary['numeric_scores'] if row['case_id'] == 'flat_1p6'
                and row['experiment_actor'] == 'grouped_continue')
    ramp = gates['primary']['ramp_0p45_complete']['grouped_continue']
    rough = gates['primary']['rough_0p35']['grouped_continue']
    totals = summary['C17_plus_C18_actual_totals']
    dev = summary['worker_ledgers']['development']
    qualification_controls = sum(summary['worker_ledgers'][arm]['actual_controls']
                                 for arm in ('primary', 'mirror'))
    conclusion = ('达到预注册额外贡献门' if rl['RL_contribution_passed']
                  else '尚未达到预注册额外贡献门')
    content = f'''# 主路线执行 17–18：转向、完整坡道跟踪与 1.6 m/s 回归通过

**C18 在原六项任务和镜像转向共七个固定仿真场景中全部通过。** 继续实际执行[阶段 15–16](main_route_execution_20260930_15_16.md)的 grouped RL final，16 个动作通道保持；本轮没有训练新模型。七项工程资格与 RL 收益分开判断：当前策略{conclusion}，`RL_contribution_passed={str(rl['RL_contribution_passed']).lower()}`。GUI 默认资格仍为 false。

这是固定 oracle、确定性命令脚本下的验收。原任务参与过开发，改变 seed 标签不产生独立物理随机样本；不能据此声称随机初态鲁棒性、未见地形分布泛化或实机可靠性。

{table}

以上分数来自两个冷启动 worker 的[primary 完整独立读回]({PREFIX}stage18/qualification_primary_readback_18.json)和[mirror 完整独立读回]({PREFIX}stage18/qualification_mirror_readback_18.json)，另有[逐场 CSV]({PREFIX}stage18/task_results_18.csv)。七条 grouped 正式轨迹跑满，原安全、末段停止和地形门通过；原转向与镜像转向均按冻结的 yaw RMS≤0.12 rad/s 门验收。每对 zero/grouped 的 qpos、qvel、ctrl、qacc_warmstart、observation 五数组逐位相同。读回逐步核对真实控制输出、原生 5 子步、接触力与编译几何，没有再次加载策略或运行物理。

速度使用 base-body 惯性 COM 投影到机体系前向轴的速度，不是世界 X 速度，也不是所有连杆合成的整机 COM。平地 1.6 m/s 保持段均速为 **{fast['speed']['mean_com_vx_mps']:.6f} m/s**，400 个样本中 **{fast['speed']['ticks_above_1p5_mps']} 个超过 1.5 m/s**。释放到稳定为 {fast['stopping']['time_to_settle_s']:.2f} s，原生子步最大前向超行程 {fast['stopping']['max_native_forward_excursion_m']:.6f} m，均在原 4.2 s / 3.2 m 门内。释放后的限速制动由基础控制器执行，不归功于 RL。

完整坡道的 RL 均速为 {ramp['hold_mean_com_vx_mps']:.6f} m/s，速度 RMS {ramp['hold_vx_rms_mps']:.6f}，满足原 0.45±0.04 m/s 与 RMS≤0.05 门。上坡、平台、下坡均有真实正轮载荷，四个整轮碰撞几何均越过最后坡面边界。物理跨坡与速度跟踪两类检查都通过。

![保存轨迹对比]({PREFIX}stage18/figures18/tracking_17_18.png)

本轮修复针对两个已定位的基础控制问题。原 yaw 已有反馈，但在开发保持段频繁达到内部 ±0.6 rad/s 上限；只把该内部上限改为 ±1.2，gain=4、操作命令与实际轮速/力矩保护不变。原坡道的轮速积分可能在车体已经超速时继续增加，改用共同车体误差后坡道通过。共同 P 通路原本就含车体速度反馈，不能将问题描述成“原来没有闭环”。

第一次组合 C17 为 **6/7**，粗糙地面 RMS 从旧 grouped 的 0.042856 升到 0.053394 m/s，超过原 0.05 门。这个失败与所有原始记录完整保留。对四条旧/新 zero/grouped 保存轨迹的诊断发现：新共同积分波动更小，车体 1–5 Hz 波动反而更大；主要回归窗口没有 yaw 或轮扭矩限幅。这些数据不支持“共同积分高频响应过强”这一解释，因此本轮未选择直接减小 Ki 或低通车体误差。

C18 唯一候选给共同 wheel/body 误差差值加入一阶状态：`d=mean(ew)−eb`、`z_after=z_before+(d−z_before)/21`、`eI=ew−z_after`；dt=0.01 s、tau=0.20 s，reset/inactive 时 z=0。它在稳态保留共同车体误差，较快变化时保留轮速积分通路，差动部分保持。Ki=3、积分±4 Nm、P 与保护不变，退绕方向使用实际新 eI。独立 checker 和 reader 同时核公式、真实前状态及 z 的逐步连续性。

这个候选先用 4,000 控制步验证 rough/ramp，通过后再固定执行完整资格，没有扫描时间常数。最终 rough RMS 为 **{rough['hold_vx_rms_mps']:.6f} m/s**，坡道资格同时保持。该结果支持这次控制律干预的闭环效果；跨轨迹接触时序已经变化，不能把逐 tick 差值当作某通道的独立因果作用。

RL 贡献仍只按 primary 原六对计算，mirror 不并入分母。共同通过集合为 {branch['common_task_passing_count']} 项，pooled policy/zero drive SSE 比为 **{branch['pooled_sse_ratio']:.6f}**（原要求≤0.85），其中 {len(branch['pairs_no_worse_within_1p02'])} 对满足单对≤1.02（原要求至少 4 对）；成本比为 **{cost['cost_ratio']:.6f}**（原门≤1.20）。任务转换分支结果为 {str(rl['branch_a_task_conversion']['passed']).lower()}，SSE 分支为 {str(branch['passed']).lower()}。基础控制修复收益不计为训练进步；本轮也没有在新基控上对比 global/grouped 两种训练法。力矩平方仅为成本代理，不等同能耗。

| 本轮实际执行 | 控制步 | normal native | compiler native |
|---|---:|---:|---:|
| C17 三臂开发 | 8,600 | 43,000 | 6 |
| C17 组合资格（保留 rough 失败） | 24,000 | 120,000 | 4 |
| C18 唯一候选开发 | {dev['actual_controls']:,} | {dev['actual_normal_native']:,} | 2 |
| C18 固定资格 | {qualification_controls:,} | {qualification_controls*5:,} | 4 |
| 合计 | {totals['actual_controls']:,} | {totals['actual_normal_native']:,} | {totals['compiler_native_returns']} |

实际累计 PPO load {totals['policy_load_calls']} 次、torch.load {totals['torch_load_calls']} 次，actor rows {totals['actor_rows']:,}；新训练、critic、backward、optimizer 和 save 均为零。C17 的 11 项必要测试与 C18 的 10 项针对性测试通过；干净进程导入和 fatal ruff 检查通过。冻结来源前后哈希一致，全部自有 worker 已回收，原 77 份冻结源和旧记录未改。完整账见[资格汇总]({PREFIX}stage18/qualification_summary_18.json)。

实际 gpt-6-astra ultra 负责机制选择、合同、源码 GO 与终审；实际 GPT-6 Sol 编写有界控制、独立核验和数据整理模块；root 完成集成、全部实际测试、模型/物理执行、计步与发布。本轮没有新的 Claude 执行记录。

[公开包]({PREFIX}README.md)约 {copied['payload_bytes']/1e6:.1f} MB，包含新源码、合同、失败与通过的独立读回、状态、命令、图表和若干真实 control/native 块。约 {copied['local_archive_bytes']/1e9:.2f} GB 全量本地档案有[逐文件清单]({PREFIX}stage18/local_archive_inventory_17_18.json)；公开子集不足以重跑完整 native/contactForce 读回。固定权重直接引用 15–16 父包；历史脚本依赖记录中的工作目录与冻结环境，不是即下载即运行的通用启动器。

本次未切换 GUI 默认入口，也未完成自由键盘、侧移、跳跃、真实 15 mm 单台阶、噪声/时延鲁棒性或实机资格。R 仍是仿真复位，物理自救未实现。最终科学边界与下一步见 [Astra 终审]({PREFIX}stage18/final_review_17_18.md)。
'''
    report = R / 'docs/main_route_execution_20260930_17_18.md'
    with report.open('x') as stream:
        stream.write(content)
    readme = f'''# C17–18 控制修复与固定脚本资格证据

[中文报告](../../docs/main_route_execution_20260930_17_18.md) · [最终汇总](stage18/qualification_summary_18.json) · [逐场 CSV](stage18/task_results_18.csv) · [Astra 终审](stage18/final_review_17_18.md)

C18 grouped 七场景全部通过；RL 贡献判据为 `{str(rl['RL_contribution_passed']).lower()}`，默认 GUI 资格为 false。这是固定 oracle 确定性脚本，不是独立随机样本。C17 的 rough 回归与 6/7 结果保留在 stage17；stage18 包含唯一滤波候选和完整后续证据。

![保存轨迹](stage18/figures18/tracking_17_18.png)

新控制入口是 [controller18.py](stage18/controller18.py)，纯算术为 [residual18.py](stage18/residual18.py)，独立读回为 [read_eval18.py](stage18/read_eval18.py)。实际测试记录见 [pure_tests_receipt_18.json](stage18/pure_tests_receipt_18.json)。源码与旧依赖身份列于阶段计划，phase 的实际 invocation 列于 reservation/session/receipt。

唯一策略仍是父包的 [grouped final_model.zip](../d1_main_route_execution_20260930_15_16/stage15/grouped_01/final_checkpoint/final_model.zip)，SHA256 `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`。本轮没有新增训练或权重。

[publication_manifest.json](publication_manifest.json)列出公开 payload 身份；[copy_receipt.json](copy_receipt.json)保留原路径及复制校验；[本地全档清单](stage18/local_archive_inventory_17_18.json)区分已上传和未上传文件。约 {copied['payload_bytes']/1e6:.1f} MB 精选资料来自约 {copied['local_archive_bytes']/1e9:.2f} GB 的本地全量档案。只上传少量 control/native gzip 样本，其余对应 manifest 作为身份记录保留，不能据公开子集声称已重新验证全部接触力链。

复核完整读回需要本地全量 control/native 块、冻结依赖与记录中的目录布局；脚本包含历史绝对路径，尚非通用一键复现包。已有输出与 reservation 均为独占记录，不能原位重跑覆盖。图表来自保存记录，没有生成新模型或物理步。

GUI、自由键盘、实机、侧移、跳跃、15 mm 单台阶与物理自救不在本次资格范围内；R 仍为仿真复位。
'''
    with (PACKAGE / 'README.md').open('x') as stream:
        stream.write(readme)
    print(json.dumps({'report': str(report), 'readme': str(PACKAGE / 'README.md')}))


if __name__ == '__main__':
    main()
