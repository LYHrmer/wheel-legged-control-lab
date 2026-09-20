# 原内层 yaw 反馈恢复资格

决策：保留一个受控试验假设——纯 raw turn pulse 内恢复原 `servo + 4*(servo-body_yaw)`，不加入新的 cap，也不叠加 center 或 leg damper。首先执行本目录固定 208 保存态的零物理资格，不能由保存态 shadow 声称修复。

证据分两层：

- 已确定：baseline 和 turn-damper 左右四条轨迹均为 50/50 pulse 拍被 ±0.6 截断；原未截断请求约 ±3.54..4.40。因此原 body-rate 反馈在这段对实际 wheel target 的局部导数为 0。去掉该 clip 恢复的是随误差变化的原反馈公式，不是选择下一固定 cap。
- 尚待证伪：恢复误差反馈是否能把更多有效牵引传给机身。已存阻尼试验虽降低中心纵向 u RMS 约 21.8%、增加 pulse body yaw 约 26.8%，heading peak 仍约 12.4°；native 世界 x/y 力矩继续大幅抵消。侧向接触约束和滑移没有被排除。

`fixed_contract.md` 固定唯一公式、样本、保护与最多一次 3200/16000 后续探针。`sampling_note.md` 明确新启用的 −6／−4 反馈斜率及 10 ms 风险。`qualify_authority.py` 从保存姿态重建横向坐标和 body yaw，独立计算原/恢复 PI、antiwindup、额定保护及精确力矩余量 yaw 区间；该区间只用于诊断。脚本还报告原几何刚腿 yaw 模态的质量/控制斜率投影和真实加载点侧向速度残差，防止将其误称为 loaded 闭环稳定性模型。

预计首次 wheel 请求大于 12 N·m，rated saturation 必须如实保留；最终原 ±30 rad/s target 与 12 N·m 保护仍在。额定饱和期间实际扭矩的速度斜率可能暂时为 0，但未饱和区的原 wheel-spin 负反馈并未被直接替换公式取消。

交付时作者只完成 AST/compile，未执行资格 helper。root 阅读后运行：

```bash
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps:/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/turn_actuation_plan_01/qualify_authority.py --work /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920 --output /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/turn_actuation_plan_01/report.json
```

全部积分、forward/inverse dynamics、collision 入口被设为立即报错；只执行 kinematics/comPos/comVel、crb/fullM、Jacobian 与物体速度查询。208 个 scratch 时刻保持 0。原 native 力只在自己的 solved-cache 采样时刻聚合，不混用端点速度算功。

旧 1.0 cap 失败保留，但它来自旧 hfield plant 且依然保留 clip，不能等同于当前 native-plane 上恢复反馈公式已被否定。反过来，它也提醒不能把更大轮速或更大轮矩当作 body yaw 已实现。

若本唯一反馈恢复候选仍失败，停止 cap/effort 家族的串行试探，按真实接触与腿工作空间证据转入新的几何/机动模式合同。原 raw 命令和 5° 等全部任务门槛始终保留，可靠转向之前不推进跳跃台阶或提速。

交付校验和不包含 root 后续生成的资格报告；该报告应另行归档。
