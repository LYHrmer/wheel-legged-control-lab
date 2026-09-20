# 下一机械机制的零物理资格

判断：直接以 body yaw error 替换 wheel PI 差动误差已经因取消直接 wheel-spin 阻尼而 NO-GO。下一项只考察**原 plane-zero 控制器的纯转向脉冲腿纵向阻尼**，固定 `b=126.4374005337902`，不叠加失败的中心速度补偿。

这是假设资格，尚不是修复。原 baseline 左转 pulse 的机身平均 yaw rate 为 0.13286 rad/s；中心补偿后为 0.24239 rad/s，但加载接触的完整腿运动分解由 0.31291 增至 0.55358 rad/s。候选 native pulse 的纵向力贡献 yaw moment 为 +11.18191 N·m，侧向贡献 −9.81334 N·m；最后 25 拍分别 +13.95534 / −14.89597 N·m。此时尚未在 tick 250 关断，故不能把 pulse 内失败归因于关断。上述分解是存档几何/力矩事实，不是各机制的因果份额。

阻尼的候选理由是直接抑制反向腿相对纵向速度，保留原轮速 PI 对 wheel spin 的负反馈。它不直接消除侧向接触阻力。真实轮轴侧向约束要求腿侧向活动或接触侧滑；完全冻结腿的无侧滑 yaw 不能被当作可实现目标。`qualify_turn_damper.py` 因此把这个几何边界与力矩、速度、采样风险一起报告。

`fixed_method_contract.md` 已先于数值检查固定输入和方法。helper 交付时只经过 AST/compile，作者没有执行它。由 root 完整阅读后运行：

```bash
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps:/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/turn_next_mechanics_plan_01/qualify_turn_damper.py --work /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920 --output /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/turn_next_mechanics_plan_01/report.json
```

成功报告的 `algebra_checks_passed` 仅表示恒等式、来源和数值闭合。它不自动授权物理、不宣称原门槛通过。自由惯量 `dt·λ` 不能代替 loaded 离散闭环；同 b 已有停车试验也是范围有限的旁证。root 综合数值与实际开发风险决定是否启动合同中唯一的 3200/16000 证伪试验，不要求未知全状态的稳定性证明。

代表点的线性解只作机械可实现性诊断，不写入控制器。完整加载接触点的残差、原 wheel target 与代表点所需轮速的差均保留，以防把单点滚动近似误称为全接触无滑移。不会用 native 力与端点速度混算功。

交付校验和只覆盖 helper、固定方法合同、本说明及静态检查收据；root 生成的实际资格报告需另行归档。
