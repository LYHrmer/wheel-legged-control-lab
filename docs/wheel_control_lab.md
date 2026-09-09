# 单轮速度闭环：辨识参数能否改善控制

[上一节](actuator_identification.md)检查模型能不能预测未见运动。本节把辨识参数用进
力矩前馈，在同一台非理想对象上测量速度跟踪误差。对象仍是 MuJoCo 固定轴悬空轮，
没有轮地接触、车身平衡或腿部运动；这里的结论不能直接推广到整机。

## 先跑一遍

在仓库根目录、已安装项目依赖的环境运行：

```bash
PYTHONPATH=src python3 scripts/evaluate_wheel_control.py \
  --output results/my_wheel_control
```

不需要 GPU 或 Torch。默认每条标定运动 3 秒，每段闭环运动 6 秒，物理步长 2 ms。
已有输出目录会报错，避免覆盖原始数据。缩短运行只用于检查接口：

```bash
PYTHONPATH=src python3 scripts/evaluate_wheel_control.py \
  --output /tmp/wheel-control-check --scenario matched \
  --duration 1 --calibration-duration 0.5 --max-delay-steps 0 --max-nfev 1
```

这个检查刻意把优化预算设得很小，通常会得到未收敛标志。脚本仍报告所得参数的控制表现，
但不会把它写成成功标定。正式记录见[结果说明](../results/wheel_control/README.md)。

## 三组到底差在哪里

| 组名 | 反馈 | 前馈 |
| --- | --- | --- |
| `pi` | 同一套 PI 与抗饱和 | 无 |
| `nominal_ff` | 同上 | 默认名义参数 |
| `identified_ff` | 同上 | 两条标定 CSV 拟合出的参数 |

`pi → nominal_ff` 的差异包含加入前馈的作用；`nominal_ff → identified_ff` 才用于观察
在同一前馈结构中更换参数的影响。只比较第一组和第三组，无法区分这两件事。

三组使用相同的初始状态、真实执行器参数、测量噪声序列和限矩。它们各自重新创建对象、
清空控制器积分与执行器延迟队列。反馈仅接收当前速度测量；仿真真值和实际施加力矩由
评测器记录，不送进控制器。负载惯量取自已知台架几何，不能用未知电机参数替代。

设定两种对象：`matched` 的摩擦形式与辨识模型相同，`mismatch` 额外带低速 Stribeck
摩擦，而前馈模型没有这一项。两者各拟合一次，各自执行全部三组对照。

标定输入是开环力矩 chirp 和双正弦。闭环参考速度另行给定：

| 工况 | 参考速度，单位 rad/s | 检查什么 |
| --- | --- | --- |
| `tracking` | $3\sin(2\pi\,0.45t)$ | 持续跟踪中的幅值误差和滞后 |
| `reversal` | $2.5\tanh(3\sin(2\pi\,0.35t))$ | 低速换向附近的摩擦误差 |
| `stress` | $18\sin(2\pi t)$ | 超出可用力矩时的误差与积分行为 |

参考加速度取上述函数在当前时刻的解析导数，不使用未来测量。反馈增益与这些参考函数
在首次闭环对照运行前固定，没有按结果挑选有利工况。默认一个参数对象、一条测量噪声
种子，没有总体置信区间。后续自己改过并看过结果的工况，应当称为开发集，不再宣称未见。

## 从方程到每一步力矩

忽略离散积分误差，台架的连续动力学写成：

$$
(J_L+I_a)\dot\omega=\tau_{\mathrm{applied}}
-b\omega-c\tanh(\omega/0.05)-\tau_{\mathrm{extra}}(\omega).
$$

$J_L=0.00189225\;\mathrm{kg\,m^2}$ 是已知轮负载惯量；$I_a,b,c$ 由日志拟合。
`matched` 中最后一项为零；`mismatch` 中它为
$0.08\exp[-(\omega/0.3)^2]\tanh(\omega/0.05)$ Nm。它们都是教学模型，不是真实 D1 参数。

参考模型前馈为：

$$
\tau_{ff,k}=(J_L+\hat I_a)\dot\omega_{r,k}
+\hat b\omega_{r,k}+\hat c\tanh(\omega_{r,k}/0.05).
$$

PI 以当前测量计算误差 $e_k=\omega_{r,k}-\omega_{m,k}$。积分状态直接存储 Nm：

$$
P_k=K_pe_k,\qquad
\Delta I_k=K_ie_k\Delta t,\qquad
I_k^*=I_{k-1}+\Delta I_k.
$$

默认 $K_p=0.12\;\mathrm{Nm\,s/rad}$、$K_i=0.6\;\mathrm{Nm/rad}$，三组不变。
这些是固定教学增益，没有做最优整定或鲁棒稳定性证明。

先计算候选总力矩 $u_k^*=P_k+I_k^*+\tau_{ff,k}$。若 $u_k^*>2$ 且
$\Delta I_k>0$，或者 $u_k^*<-2$ 且 $\Delta I_k<0$，则本步拒绝积分，
保留 $I_{k-1}$；其他情况接受 $I_k^*$。随后重新计算总力矩并限幅：

$$
u_k=P_k+I_k+\tau_{ff,k},\qquad
\tau_{cmd,k}=\operatorname{clip}(u_k,-2,2).
$$

这一顺序在 [`WheelVelocityController.compute`](../src/wheel_legged_control/wheel_control.py)
中逐项实现。候选积分被拒绝后，重新计算的总力矩可能没有超限，所以日志中的
`integration_frozen` 与 `saturated` 不必同时为真。反向积分可以被接受，用于退出饱和。

限幅后的命令还要经过对象自身的延迟队列。辨识结果包含 `delay_steps`，但这一版前馈
明确不补偿延迟；把预测偏差拟合成一个更大的延迟，并不意味着提前发力就一定正确。

## 找一行 CSV 手算

打开 `matched/tracking_identified_ff.csv`，依次核对：

1. 用 `reference_velocity_rad_s - measured_velocity_rad_s` 重算 `error_rad_s`。
2. 用上一行 `integral_torque_nm`、本行误差和步长算候选积分，第一行从零开始。
3. 从 `fit.json` 取参数，算本行 `feedforward_torque_nm`。
4. 按条件积分规则决定是否冻结；核对 `requested_torque_nm = P + I + FF`。
5. 限幅得到 `command_torque_nm`；再检查 `applied_torque_nm` 是否对应延迟后的旧命令。
6. 当前行 `next_velocity_rad_s` 应与下一行 `actual_velocity_rad_s` 完全衔接。

主跟踪误差用同一时刻的参考与仿真真值计算：

$$
\mathrm{RMSE}=\sqrt{\frac1N\sum_{k=0}^{N-1}
(\omega_{r,k}-\omega_k)^2}.
$$

不要用 $\omega_{k+1}$ 配 $\omega_{r,k}$，否则会把一部分离散延迟隐藏掉。测量噪声进入
反馈，但不直接进入主指标。所有物理步都被记录，包含起步阶段；没有删去较差区段。

`metrics.csv` 还报告力矩 RMS、峰值请求力矩、限幅比例和积分冻结比例。力矩 RMS 用于
观察控制代价，不是电功耗。超出 30 rad/s 的保护阈值会提前停止该回合，摘要保留
`completed=false` 和触发标志；部分回合的 RMSE 不能与完整回合直接排名。

## 输出与学习任务

需要带答案的纸笔练习时，先做[抗饱和、日志与 PPO 检查题](control_ppo_checks.md)。
其中专门区分了请求力矩、限幅后命令和延迟后的实际力矩。

`experiment.json` 保存完整参数、软件版本、代码 SHA 和配对协议；每个对象目录保存两条
标定 CSV、拟合候选及状态、九条逐步闭环日志和三张图。`summary.json` 汇总所有工况，
`manifest.json` 保存产物 SHA。控制器默认值或参考函数改动后，用新目录重新跑，不修改旧表格。

建议分三次完成：

- **先拆开前馈项。** 找一个正加速但参考速度为负的采样点，分别计算惯量项、黏性项和
  摩擦项。它们能否一正一负？只看最终力矩时会漏掉什么？
- **再研究饱和。** 在 `stress.png` 中找积分冻结区段，回到 CSV 验证判断。复制配置做
  一次关闭积分的对照，再做一次关闭抗饱和的实验，各自写入新目录。先预测积分和跟踪
  误差如何变化，再运行；不预设“更大前馈一定更好”。
- **最后检查模型失配。** 比较两种对象的换向误差与拟合延迟。先仅关闭摩擦前馈，再仅
  修改其幅值。预测误差降低、换向跟踪改善、饱和时间减少，是三个不同的验收指标。

完成后接着做[一次 PPO 参数更新](ppo_update_lab.md)：那一节检查冻结批次如何生成梯度，
不会训练本节单轮控制器。整机接入需要重新处理动作接口、观测、奖励、总力矩约束及独立
评测；这两项小实验目前都不改变已有 D1 控制器和策略。
