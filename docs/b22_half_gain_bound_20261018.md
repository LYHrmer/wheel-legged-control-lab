# B22 gain0.5：贡献门必要条件已经不可满足（2026-10-18）

本记录为 **analytic necessary-condition failure**。在保留35已完成五组zero/half-gain配对、37已完成ramp zero参考和原六任务评分门的条件下，缺失的ramp B即使取得最有利的合法结果，当前固定gain0.5候选也无法通过原RL贡献门。

这不是完整矩阵验收：35与37仍为 `INCOMPLETE`，整轮 `actual_counts=null`不变；原完整任务和RL贡献资格字段继续为null。本分析没有补造ramp B、镜像yaw或其他缺失结果，也没有调用要求12条完整记录的 `score_pairs` 来制造完整判定。

## Branch B：最多3组不劣化，小于要求4组

冻结原门要求至少4个共同task-pass配对、pooled SSE≤zero的0.85倍，并且至少4组满足 `S_policy ≤ 1.02 × S_zero`。三个条件必须同时成立。

从既有独立readback中的实际drive-window SSE重新计算：

| 原任务 | half-gain / zero SSE | 满足≤1.02 |
|---|---:|---|
| flat0.6 | 1.111028343655 | 否 |
| flat1.6 | 1.074186626656 | 否 |
| yaw1.2 | 1.015013581831 | 是 |
| bumps0.4 | 1.046487162848 | 否 |
| rough0.35 | 1.006683465609 | 是 |

这五组均有完整1600拍、完整drive窗口，两边task均通过。已知只有2组满足不劣化门，另3组已经超过1.02。原六任务只剩1组ramp B未知；它至多增加1个合格配对。因此上界为 **2+1=3<4**。即使缺失ramp的pooled SSE或torque cost很好，也无法挽回这个独立的必要条件。

true15mm step与mirrored-yaw属于扩展任务验证，不计入原六任务贡献门，不能增加上述分母或替换三个劣化配对。

## Branch A：六个zero均通过，没有任务转换机会

Branch A要求至少一个真实 `zero task_fail → policy task_pass` 转换，并满足无新任务退化、共同前提和cost门。

35前五个zero全部task-pass；37的完整ramp zero（seed181206）也task-pass。因此这套固定六zero参考的失败数为0，任何保留这些参考的合法补齐至多有 **0个转换**，小于要求的至少1个。缺失ramp B不能把已经通过的zero参考变成失败。

所以两条贡献分支的必要条件都已失败，完整cost门或未知ramp B的好结果均不能使这一个固定候选通过原贡献门。

## 配对与来源

本次独立核对五组readback SHA及各自seed、指令drive窗口、完整时域；重读十个初始NPZ，重新计算六字段dtype/shape/bytes SHA、完整controller reset JSON SHA及compiled robot/course SHA，zero/B五组全部精确配对。十个session均包含原35 release输入闭包，实际gain0.5、controller combined，B加载证明均指向同一完整policy state `59440de0c5beb102cec0d292838fb9b125eca7eb56bbffc06d1eaec46ec7aa48`，原checkpoint为 `7bd58b5d6114745b12b4284be3f861f973a5d31d9d596365a5b54cb5e0d76691`。

37的ramp zero有完整1800拍和独立readback，沿用35物理/控制/重置代码的派生，仅更换C37准入标识。它用于固定ramp zero参考；未知B仍必须满足真实相同初态配对，才能算合法补齐。

原评分源码 `rl16_heldout_score_08.py` SHA为 `9ca55bff625125aac63e598c06f2177cb88518a7268febd0abb48bbccfe5bc1f`；branch B常量和两条分支逻辑均按此原源码读取。全部绑定SHA与逐pair数据见 [机器报告](../reports/b22_half_gain_bound_20261018.json)。没有重新扫描旧590原始native归档，没有执行推理、物理或测试。

## 下一步的证据边界

这个上界只约束保留上述既有证据的 **固定gain0.5候选**，不能推广成所有缩放增益或未来新训练策略都会失败。38的两个profile仍可验证NumPy后端兼容性；它不会修复原贡献门，也不需要为证明这个上界重新采集整个矩阵。

后续若考虑训练新候选，先闭合完整checkpoint、Adam和父代累计训练来源。C22训练源码实际调用 `load_warm_start20(...parentpaths..., stage=...)`，并以 `reset_num_timesteps=True`学习；32768是当前stage预算，不能据此断言全部历史只训练了32k，更不能仅用“训练太少”作为继续同类训练的理由。本分析没有创建新训练版本或执行预算。
