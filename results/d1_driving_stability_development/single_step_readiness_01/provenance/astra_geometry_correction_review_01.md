# 静态几何资格修正审查

审查者：实际 gpt-6-astra ultra；2026-09-22。仅只读源码、保存JSON和官方资料，并对保存数值独立做算术；未导入项目/MuJoCo、未构造模型、未调用collision/forward/step、未运行测试或消耗物理预算。本文exclusive-create。原失败证据不改写。

输入：`geometry_preflight_01/qualification.json`，SHA256 `ca62e401735e0a7c2c4fae8701d271fee86f82d8f232c9dc9b96de720132e0ed`。该记录报告native/control均0，forward2589、setConst2；机器人/solver等价、初态、唯一box、分离均通过，query-ray与static contact分类失败。这里核验的是这些已存数值及拟议修正规则，不替代root重新运行零积分资格。

结论：**同意两项修正，附以下窄范围约束。** 它们修复数值边界验证和接触特征解释，不调整plant、controller、接触参数、任务门槛或2400/12000预算。不得仅因新规则能通过旧失败而宣布资格通过；root须在新目录重做全套原静态检查并保留原失败。

## 1. 闭边界query与ray漏检

现代码 `scripts/d1_single_step_plant.py:224` 用world上下界直接作闭区间判断，正确。`scripts/qualify_d1_single_step.py:78` 的射线检查再由MuJoCo把world点变成local坐标，浮点减法可能让数学边界略在local盒外。

保存数据中，False条件18/18 ray一致；True条件仅x两侧的6个精确边/角点漏box而落到plane，其他采样点正常。6个漏检点的二维local包含残差都是 **+1.6653345369377348e-16 m**。把边界轴用一次 `nextafter(boundary, center)` 向内移动后，由保存compiled中心/半尺寸计算的残差变为 -1.1102230246251565e-16 或 -2.7755575615628914e-16 m。本审查只核验这些算术，不声称已经运行内点ray。

建议实现：

1. **完全保留query的闭区间比较**，不要在query端加epsilon、abs(local)<=half+tol、扩大box或改其center/size。
2. 精确boundary fallback只适用于当前点某个world坐标**逐值等于**compiled `center-half` 或 `center+half`，其余投影坐标处于闭范围内、query返回box顶高的情况。普通inside/outside ray不一致继续失败。
3. 验证使用实际compiled数据；构造top点 `p=(x,y,center_z+half_z)`，local为 `R.T@(p-center)`。由于本合同box identity quaternion，应同时assert该不变式。计算 `q=abs(local)-half`，报告 `max(q)`。此量应命名 **signed box containment residual**；它不是盒外多轴情况下的欧氏signed distance。如必须称signed distance，使用标准公式 `norm(max(q,0)) + min(max(q),0)`。
4. 只给这项数值交叉检查使用 `tol_roundoff = 8*eps_float64*max(1,max(abs(p)),max(abs(center)),max(half))`，单位m并保存实际值。8eps裸值对当前量级也足够，但显式量纲/scale更清楚。要求top高度及边界轴残差在此舍入容差内，不能只检查任意深在盒内也有max(q)<=tol而把它当边界。
5. 在每个精确边界轴上做一次向中心nextafter；角点需要同时处理两轴。保存perturbed点、每轴ULP位移及其strict-local-inside检查，再要求该独立内点ray确实命中compiled box、顶高一致。若一次nextafter仍不能构成strict interior或ray仍失败，保留失败，不自适应增大偏移直到通过。
6. 每个边界轴另验证向外nextafter的**原query**立刻返回0，已有±1e-8样本继续保留。fallback不能把outward-neighbor因为也位于舍入容差内而改判inside。
7. 新报告字段分开：raw_ray_agrees=false、boundary_roundoff_explained、compiled_containment_residual、interior_ray_agrees、qualification_passed。不要把原ray命中plane的ID/高度覆写为box，也不要将其描述为精确边界ray已经通过。

这满足“query与真实compiled几何的ray或距离结果核验”：边界由compiled几何包含关系交叉验证，ray在最邻近可表示内点独立确认，没有改query定义。

## 2. 以位置选面，再检查法向锥

现 `scripts/d1_single_step_geometry.py:249` 把每个abs(normal_component)>1e-6的轴直接视为support face。因此顶面中心附近的0.001切向分量被误认为距离约0.18m或0.64m以外的侧面；front中部同样被误认为top/bottom。这是分类算法问题。

官方说明：positive margin的多接触路径可通过小角度旋转两几何重复求碰撞。来源：[MuJoCo multiple contacts](https://mujoco.readthedocs.io/en/latest/computation/#multiple-contacts)。已核对与实际metadata版本一致的[MuJoCo 3.12.0 engine_collision_convex.c](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_convex.c#L839-L894)：角度为0.001，两个geom反向旋转；每个切向轴分别尝试后恢复，新增点还继承初始contact.dist。故不要把附加点的微小切向法向分量解释为新的远处box面，也不要要求所有附加点的contact位置与该dist构成精确未扰动表面关系。

最小正确算法：

- 保存输入normal原始值，不改frame、不归一化回写原始normal、不投影替换力，不修改normal_load。现有norm/orthonormal检查保留。
- 令 `tau=max(actual_contact_margin,abs(contact.dist))+1e-7 m`。使用actual `contact.includemargin`（若使用geom margin须先证明与此一致）；当前样本均.001。对于每个面 `(axis,sign)`，只有 `abs(local_position[axis]-sign*half[axis])<=tau` 才作为position candidate。仍要求所有轴位于扩展盒内，且至少有一个候选面。
- 候选面法向为 `sign*e_axis`。将单位原normal投影到它们生成的**非负法向锥**，不是整个线性span；单top面时投影只保留正z，向下法向必须失败。没有同轴正反两面时：`projected += outward_face_normal*max(dot(normal,outward_face_normal),0)`；residual=`norm(normal-projected)`。box有旋转时先在local frame计算，当前合同应直接确认无旋转。
- 固定 `normal_cone_residual<=0.0021` 可接受，定义为dimensionless unit-vector距离的**解释容差**。该数值远小于错误面/反向法向的残差，容许当前多接触小扰动。它不是物理门槛、不是MuJoCo给出的严格数值误差保证；尤其不能用“两次切向轴角度相加”推导，因为源码是逐轴独立尝试。若说明其来源，应写“与±0.001rad几何扰动一致的保守固定诊断余量”，不要写成证明。
- candidate数为1/2/3时可标face/edge/corner **neighborhood within declared position tolerance**；同时保存candidate face名、面偏差、tau、投影系数、cone residual、是否有效和原normal。near-edge的pure top法向可能有两个位置候选，因此该标签不单独证明同时由两面承载；任务的真实载荷仍用原native contact force。
- 若深穿透使同一轴两相反面同时落入tau，不能因法向锥扩成整个空间而随便确认normal合法。标记ambiguous/deep-overlap，保留contact与物理失败证据，不伪造唯一feature。当前低深度21条box样本均无此歧义。

独立重算保存的全部box contact（包括同姿态其它轮，不只被预检选中的FL轮）：

| 合成姿态 | box contact数 | 位置候选 | 最大normal cone residual |
|---|---:|---|---:|
| top | 8 | 全部仅top | 0.0010015818331200997 |
| front | 9 | 全部仅front | 0.0010353578937705712 |
| front_top_edge | 4 | 全部front+top | 0.0009999985771347126 |

全部在候选位置范围内、无正反面歧义，且residual<0.0021。这支持root提出的固定规则；不是对将来轨迹中任何法向都有效的保证。未来超过界限继续归档/失败，不再调整容差以刷通过。

## 3. root本次修正的必要验证范围

仅新零积分和纯数学检查，无物理：

- 原False/True模型等价、初态、box唯一性、分离、所有普通query/ray检查不删除。
- 六个已失败边/角点：raw miss保留；compiled residual验证与nextafter内点ray通过；每边向外nextafter query=0。
- 顶/前/前顶边全部实际静态contact通过新的位置+cone解释，plane法向检查保持原严格定义；所有静态力都仍不算readiness载荷。
- 纯反例：顶面中心配45°侧向normal失败；顶面向下normal失败；front面向内normal失败；正确edge正锥内normal通过，锥外normal失败；距离面远处的点不能靠normal方向获面资格；0.0021上下邻近残差正确分界；原frame/normal/force bitwise保留；deep-opposing-face返回ambiguous。
- 两个修正只修改interpretation与qualification源码，新manifest包含变更。原失败目录及qualification.json保持原SHA，新结果写geometry_preflight_02或其它exclusive目录。

据此可以继续实现和重做静态资格；本报告不授权绕过其它尚未通过的记录/scorer/计步前检，也没有执行或增加任何物理。
