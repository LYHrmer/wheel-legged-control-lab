# C18：保留坡道直流修正，恢复轮速瞬态反馈

2026-09-30，Astra决策。C17真实组合验收grouped为6/7：原yaw RMS 0.09542139735303773、镜像0.09852257259668158，完整ramp均速0.46985660182497285/RMS 0.02120257264488338及几何通过；rough RMS 0.05339427199909047超过原0.05，zero也为0.05422323114117235。所有场次完整、记录、安全及最终停止通过。整体资格仍为false；RL贡献也为false。全部C17失败/通过、冻结源码和预算原样保留，不把两项定向通过写成整体验收完成。

## 保存数据诊断与取舍

唯一诊断只读C16/C17的zero/grouped四条rough轨迹。固定drive[175,645)、hold[245,645)及四个100tick段，读取原控制记录，重算原分数和P+common恒等式，描述接触进度及频带。root已执行完成4.326156358s，模型/物理0，输入未变。结果` saved_rough_diagnosis_18.json` SHA256=`42486219f0f318c7c479a83b65ae6f57b4b8e51f340c87be1dd3a026dca368ad`；脚本SHA256=`b232ce8a0d7ef4e92adf5dc0149351045772514fdc162d2a47ddb0b4301a8153`。

三项可证伪解释及现有证据：

1. **body积分追逐高频过强。现有数据反对。** grouped共同I标准差由旧0.408686降至新0.225889 Nm，新1–5Hz共同I能量仅0.016779 Nm²；车体速度1–5Hz能量反而增至0.002329 (m/s)²。不能以“新I高频过强”为理由再低通body误差或降低Ki。
2. **yaw上限放宽导致主要回归。作为主要解释证据不足。** 最大速度波动增幅位于首100tick；该段新旧yaw均未达到旧限幅，轮扭矩/保护和支撑分配未限幅。全hold仍有少量超过旧yaw限幅的状态，不能断言yaw改动完全无交互。
3. **即时共同坐标替换删除了有用的轮速瞬态积分。值得唯一干预检验。** C17保留了共同body误差的直流调节，rough共同I的1–5Hz响应同时减弱，而轮身相对运动和body速度波动增大。旧轮误差积分在ramp会积累错误共同偏置，因此不整体退回旧式。选择频率分离：慢共同修正采用body速度，快速轮误差保持原通路。

跨轨迹接触时序和x位置已经不同；频谱与相关是描述，以上第3项尚非因果定论。完整真实干预与原门验收负责证伪，不能用离线重算扭矩冒充新物理结果。

## 唯一固定改动

设每轮误差`ew = clipped_wheel_target − actual_wheel_omega`，共同body误差`eb=(servo_vx−pre_body_COM_vx)/0.087`。仅增加标量状态z：

```
dt = 0.01 s; tau = 0.20 s; alpha = dt/(tau+dt) = 1/21
common active:
    d = mean(ew) - eb
    z_after = z_before + alpha*(d - z_before)
    eI = ew - z_after
common inactive:
    z_after = 0
    eI = ew
reset: z = 0
```

这是对wheel/body共同坐标差的低通，不是低通body误差本身。恒定输入收敛后`mean(eI)=eb`；快变化趋向原wheel误差，差动分量保持。tau仅选0.20s，对应约0.8Hz，用于把诊断中的主要1–5Hz波动与更慢共同偏置分开；不扫描时间常数。该折中可能改变启动、停止及0.8Hz附近相位，不能预判成功。

z在每个实际控制tick更新一次，与PI是否commit无关；servo_vx不为0时沿用原common active判定，servo降到0时清z。PI保持Ki3、积分±4Nm、Kp2.2及原pre-common候选保护范围，退绕符号使用实际新eI。yaw gain4/上限1.2、动作16维和实际policy、P/common P、支撑、servo、观测、地形、物理和数值门全部保持C17。

## 执行与预算

`intervention_contract_18.md`固定开发：一个新候选worker，grouped floor600、rough1600、ramp1800，共4,000 controls；复用已存C17作为基线，不重跑旧batch。两正式任务完整通过并独立读回后，才可按`qualification_contract_18.md`冻结资格：原六zero/grouped对及mirror yaw对，24,000 controls。新阶段最多28,000 controls、140,000 normal native、6 compiler、3 PPO load、9 torch.load；训练0。

只新增C18来源和独立checker/reader，保持exact WorldUprightCourseEnv与完整native/contactForce归档链。Sol实现新control与独立核验；root负责全部测试、来源冻结、实际执行和预算；Astra给来源GO与结果审阅。任何来源/软件错误或任务失败原样保留，不原位重试，不借未用额度追加候选。若唯一候选失败，应依据新证据制定下一份有界合同，而非修改当前门或在本合同内网格搜索。

工程任务、RL贡献和GUI资格仍分别判定；最终七场景必须全过。继续使用原grouped checkpoint，不把基控改善写成新训练或裁剪算法收益。新seed只作归档身份，不声称物理随机独立。
