# C32 一次保存数据诊断合同

冻结 C31 的 256 个事件，按完整周期 0–31 / 32–47 / 48–63 切成 128 / 64 / 64 行。训练之外只报告，不选模型、不重拟合。目标为保存的 GAE value_target 与周期内无自举 gamma=1 reward 累加 MC return；独立核对原 reward/GAE 公式。

四个固定估计器分别为方向×macro 的 8 格均值 P、原 54 维加截距的 OLS L0、同空间 ridge L1、8 格截距加 obs[6:54] 的 ridge H1。ridge 目标统一为平均残差平方加 0.01 倍受罚系数平方和；全局/8 格截距不罚。L0/L1/H1 各一次双目标 numpy lstsq（rcond=1e-12），合计 3 调用、6 目标拟合；P 另计 2 均值拟合。无调参或重复求解。

计数：四方法×双目标×256=2048 标量预测；JSON 保存的最终 value 仿射参数再算 256 次，合计 2304。最终 value 曾看过全部周期，不能称为其未见验证集。只报告 MSE/MAE/RMSE/偏差/EV/相对 P 的 MSE、求解器返回秩和奇异值，并保存逐行预测。EV≥0.8 且 MSE/P≤0.5 在 validation/test 各自同时满足，仅为预先固定的描述性“充分拟合”标签。

root 唯一执行，一次最多 300 秒；Astra 冻结合同并在 root 纯检查后审阅 source GO。0 actor、B22、torch、physics、native integration、optimizer、新采样。输入哈希见 JSON。任何完整性/数值/预算失败立即停止，无自动重试。旧记录、77 冻结控制和 B22 均不改。

诊断不能证明物理提速，也不构成任务完成。后续重点为占 5.47 秒的 body shift 与 recenter 时序，另写有限合同，先左右固定可行性，再考虑 RL；现有安全与提速门槛不放宽。
