# 83 候选方向的只读核对

这是对 49/82 已保存轨迹与名义模型快照的标准库读回，不是正式实验、仿真、模型执行、参数搜索或候选闭环结果。工作目录为 `/home/lyh/wheel-legged-control-lab`。全部读取用 `rtk proxy`，未改旧源、旧报告或旧轨迹。以下命令运行时没有导入 NumPy、SciPy、MuJoCo 或控制器。

## 已观察数字与限制

- 82 mismatch baseline 三 seed 的高度 RMSE 为 125.236566991 / 125.228601354 / 125.227640412 mm；Delay82 为 125.293476357 / 125.287747248 / 125.286680070 mm。原拒绝结论不变。
- 以真实归档 extension <= -0.119 m 作事后描述：baseline 三场各 378/400 拍；Delay82 为 379/400、378/400、378/400 拍。首次落入该带的时间分别为 baseline 全部 0.46 s，Delay82 为 0.44/0.46/0.46 s。这个真实状态仅用于离线描述，不是候选的估计器输入。
- baseline 腿力 RMS 为 155.562931917 / 155.568010825 / 155.587536541 N；Delay82 为 152.610261853 / 152.611617096 / 152.625990141 N。baseline 腿力峰值 176.223224730 / 175.973370200 / 176.680894032 N，远低于 260 N 输入界。原 49 的 +5% 腿力 RMS 门约为 163.35 N；恢复全程直立静态支持所需的 171.675 N 是由离线已知场景质量推导的诊断值，**不得向控制器提供该真实质量或该支持力**。这表明完整高度恢复与原成本门有张力，但不是 10% RMSE 改善不可行的证明。
- 名义快照的 equilibrium=[0,137.34] N，B[2,1]=1.4207896303185807e-5、B[5,1]=0.0014194425918168142，A[5,5]=0.9943222296470948，A[2,5]=0.019943168414931017。读取 `plant.equilibrium_control` 也只会获得缓存的名义值，不能将其命名为失配对象的真实平衡力。
- 只用保存的测量值与保存的实际施加力，按 `r_v = measured_next[5]-A[5,:]@measured_now-B[5,:]@(applied-nominal_eq)`、`raw_d=r_v/B[5,1]` 读回。两端测得 |extension|<0.115 m 的 Delay82 三场各 21 个区间，raw_d 均值为 -30.736712966 / -30.823579649 / -30.947121008 N。baseline 各 22 个区间，均值 -31.099002516 / -30.813123384 / -31.680187189 N。
- 两端测得 extension<-0.119 m 时，Delay82 raw_d 均值变为 -14.343579667 / -14.949794734 / -14.851282394 N，单步最大正值可到 208.314025734 N。它混合限位反力、非线性、阻尼/输入增益失配与测量噪声；不能解释为质量估计。这支持估计器在接近机械限位时冻结的必要性，没有证明候选一定改善闭环。
- PI baseline matched 三场饱和率均 55.5666667%，mismatch 为 56.4 / 56.4 / 56.3666667%；FF 单独越界比例为 matched 50.4%、mismatch 51.6%。六条轨迹全部饱和拍中，积分项与饱和方向同号的拍数均为 0，仅由积分项造成越界的拍数也均为 0；积分绝对幅值最大约 0.10737 Nm。故没有证据将已有条件积分替换成回算 anti-windup 称为显然的修复。matched 辨识 delay=2，mismatch 辨识 delay=4 且位于搜索边界，真实合成对象 delay=2；直接用辨识 delay 做预览还存在失配风险。未提出或运行新的 PI 臂。

## 新增模块及候选数学口径

新增 `src/wheel_legged_control/offset_mpc83.py` 和 `tests/test_offset_mpc83.py`；只作过标准库 AST 文本解析，未运行测试、导入候选或数值执行。

共同骨架为 DelayMPC82。唯一干预为一个固定定义的标量垂向输入扰动观测与平衡点一致更新机制：tau=.2 s，alpha=1-exp(-dt/tau)，滤波后 clip 到 ±60 N，前后任一测得伸缩位置 |q|>=.115 m 时冻结；reset 令 d=0。估计残差一直相对固定 nominal_eq，上一拍 applied 来自控制器自身绝对力 FIFO。不得以当前新发出的力替代延迟后的 applied，也不能把 estimated_eq 代回创新式导致重复扣除已估扰动。

`estimated_eq = nominal_eq - [0,d]`，故 `B@(u-estimated_eq) = B@(u-nominal_eq)+B[:,1]*d`。预测的 pending/issued 输入、R 的输入偏置、box 的相对坐标与 warm start 一起改变坐标原点，保持原绝对输入界不变；Q/R 数值、名义 A/B、物理六态 DARE terminal、H20 与原 L-BFGS-B 均不改。warm start 按 old_eq-new_eq 平移，保持其绝对计划力不变。controller-side view 不持有 live plant，不修改真实 plant，也不读真实 mass/data/qfrc/contact。构造必须保持 82 原顺序：先名义模型构造/快照，再 env.reset 施加 domain。

可判定开发预检为同 3 个已观察 seed 490101/490102/490103 × nominal/mismatch_delay × Delay82/Offset83，每场 400 控制步：12 场、4800 controls、48000 显式积分子步、4812 次求解。原 49 六场已封存 baseline 可作固定绝对性能参照；如果 root 选择同期追加该基线，则必须将预算注册成 18 场而非 12 场。最少预检可按 60 CPU s/120 wall s 登记，82 同量级执行实测 6.28 s 仅供时长参考，不能保证本候选耗时。

正式门保持 candidate 对原 49：失配高度 RMSE 至少 -10%，每场景速度/pitch RMS/pitch 峰值/归一化 effort/轮力 RMS/腿力 RMS 均不恶化 >5%，solve P95<=20 ms，所有登记场次完整、无新增失败。对共同 Delay82 的增量效应另报，不能用其较差表现替换原绝对门。三个 seed 是已观察的开发噪声实现，nominal 重复不是独立证据，不作显著性或实机结论。通过后另行决定原完整对照；失败/partial 保留、不补 seed、不扫描 tau/限幅。

最大风险是输入增益和阻尼也失配，真实误差并非恒定 matched 输入扰动；限位冻结后可能冻结不充分或有偏的早期估计；恢复支持力可能违反固定成本门。候选只检验此机制能否在原门内形成改善，不预称解决稳态误差。

## 已执行的只读命令

第一个方向读回命令如下（输出还包括完整名义矩阵及旧拟合文件；上文只摘取相关数字）：

```bash
rtk proxy python3 - <<'PY'
import ast,csv,gzip,json,math,struct,zipfile
from pathlib import Path
root=Path('.')
p=Path('results/delay_mpc82_20261103_01/490101_mismatch_delay_benchmark_candidate/nominal_prediction_model.npz')
with zipfile.ZipFile(p) as z:
 for name in ('a.npy','b.npy','q.npy','r.npy','terminal_q.npy','equilibrium.npy'):
  data=z.read(name); maj=data[6]; start=10 if maj==1 else 12; size=int.from_bytes(data[8:start],'little'); meta=ast.literal_eval(data[start:start+size].decode()); vals=struct.unpack('<'+'d'*((len(data)-start-size)//8),data[start+size:]); shape=meta['shape']; print(name,shape, list(vals) if len(shape)<2 else [list(vals[i:i+shape[1]]) for i in range(0,len(vals),shape[1])])
for arm in ('baseline','candidate'):
 for seed in (490101,490102,490103):
  path=Path(f'results/delay_mpc82_20261103_01/{seed}_mismatch_delay_benchmark_{arm}/trajectory.csv')
  with path.open() as f: rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
  rms=lambda v:math.sqrt(math.fsum(x*x for x in v)/len(v))
  force=[r['leg_force'] for r in rows]; e=[r['extension']-r['target_extension'] for r in rows]
  print('metrics',seed,arm,'n',len(rows),'rmse_mm',rms(e)*1000,'force_rms',rms(force),'force_mean',sum(force)/len(force),'force_peak',max(force),'lower_count',sum(r['extension']<=-.119 for r in rows),'first_lower',next((r['time_s'] for r in rows if r['extension']<=-.119),None),'mean_error',sum(e)/len(e),'mean_square_fraction',(sum(e)/len(e))**2/rms(e)**2)
print('PI fits')
for obj in ('matched','mismatch'):
 print(obj,Path(f'results/wheel_control/{obj}/fit.json').read_text())
PY
```

第二个命令只计算保存测量的区间残差与 PI 现有积分方向计数：

```bash
rtk proxy python3 - <<'PY'
import csv,math,statistics
from pathlib import Path
for arm in ('baseline','candidate'):
 for seed in (490101,490102,490103):
  p=Path(f'results/delay_mpc82_20261103_01/{seed}_mismatch_delay_benchmark_{arm}/trajectory.csv')
  with p.open() as f: rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
  cats={'precontact':[],'atlimit':[]}
  for a,b in zip(rows,rows[1:]):
   residual=b['measured_5']-.9943222296470948*a['measured_5']-.0014194425918168142*(a['leg_force']-137.34)
   if abs(a['measured_2'])<.115 and abs(b['measured_2'])<.115: cats['precontact'].append(residual/.0014194425918168142)
   if a['measured_2']<-.119 and b['measured_2']<-.119: cats['atlimit'].append(residual/.0014194425918168142)
  print('MPC residual-force',arm,seed,{k:{'n':len(v),'mean':statistics.mean(v),'min':min(v),'max':max(v)} for k,v in cats.items()})
for obj in ('matched','mismatch'):
 for seed in (490101,490102,490103):
  p=Path(f'results/pi49_preflight_20261022_01/{seed}_{obj}_stress_baseline/trajectory.csv')
  with p.open() as f: rows=list(csv.DictReader(f))
  vals=lambda k:[float(r[k]) for r in rows]
  q=vals('requested_torque_nm'); ff=vals('feedforward_torque_nm'); prop=vals('proportional_torque_nm'); integ=vals('integral_torque_nm'); err=vals('error_rad_s')
  sat=[i for i,x in enumerate(q) if abs(x)>2]; wrong=[i for i in sat if integ[i]*q[i]>0]; oppose=[i for i in sat if integ[i]*q[i]<0]
  incsat=[i for i in sat if abs(ff[i]+prop[i])<=2]
  refs=vals('reference_velocity_rad_s');cmd=vals('command_torque_nm'); applied=vals('applied_torque_nm');unwind=[i for i in sat if err[i]*q[i]<0]
  print('PI',obj,seed,'saturation',len(sat)/30,'integral_range',min(integ),max(integ),'I_agree_sat',len(wrong),'I_oppose_sat',len(oppose),'saturated_only_due_I',len(incsat),'unwind_sat',len(unwind),'ff_only_sat',sum(abs(x)>2 for x in ff)/30,'max_overshoot',max(abs(x)-2 for x in q))
PY
```
