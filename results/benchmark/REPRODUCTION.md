# 复现说明与版本容差

本目录的 `metrics.csv` / `metrics.md` / `randomized_audit.*` 是在**第一个提交** `aa93def`（2026-08-29）就入库的，当时 `pyproject.toml` 的约束是 `mujoco>=3.1,<4`，**没有记录实际使用的 MuJoCo 版本**。本文补上这项缺口，并记录一次实测复现的差异。原表**不改写**。

## 这张表是什么

6 状态平面教学模型（`src/wheel_legged_control/assets/wheel_legged_planar.xml`，3 关节 `base_x` / `body_pitch` / `leg_extension`，2 执行器 `wheel_force` / `leg_force`）上，三种外环在三个固定场景的对照：

- `LQR`、`MPC` 是 `src/wheel_legged_control/controllers.py` 的 `LQRController` / `LinearMPCController`
- `LQR + PPO` 是 2 维有界残差叠在 LQR 输出上（`env.py` 的 `WheelLeggedResidualEnv`，`baseline="lqr"`），**智能体从不直接给力**
- 场景：`nominal`、`push`、`mismatch_delay`，默认 seed 11

`randomized_audit.*` 是另一组：20 个配对随机种子的均值 ± 标准差。

## 2026-10-11 实测复现

重跑命令（在仓库根目录）：

```bash
PYTHONPATH=src MPLBACKEND=Agg python3 -m wheel_legged_control.experiments \
  --output results/my_benchmark --seed 11
```

环境：**MuJoCo 3.12.0**、NumPy 2.2.6、Python 3.10.12、Linux x86_64。未安装 `[rl]` 可选依赖，因此只产出 LQR / MPC 两行（残差行需要 `stable-baselines3`，见下）。

固定场景表的逐项差异：

| 场景 / 外环 | 指标 | 库内（约 2026-08） | 重算（MuJoCo 3.12.0） | 差异 |
|---|---|---:|---:|---:|
| nominal / LQR | 速度 RMSE | 0.5436128 | 0.5374909 | −1.13 % |
| nominal / LQR | 高度 RMSE | 20.3109 mm | 19.7203 mm | −2.91 % |
| nominal / MPC | 速度 RMSE | 0.5399265 | 0.5339407 | −1.11 % |
| nominal / MPC | 高度 RMSE | 20.3177 mm | 19.7273 mm | −2.91 % |
| push / LQR | 速度 RMSE | 0.6720185 | 0.6673156 | −0.70 % |
| push / LQR | 高度 RMSE | 19.7528 mm | 19.1696 mm | −2.95 % |
| push / MPC | 速度 RMSE | 0.6680701 | 0.6634831 | −0.69 % |
| push / MPC | 高度 RMSE | 19.7740 mm | 19.1902 mm | −2.95 % |
| mismatch_delay / LQR | 速度 RMSE | 0.5428561 | 0.5358307 | −1.29 % |
| mismatch_delay / LQR | 高度 RMSE | 125.23419 mm | 125.23416 mm | −2.6e-5 % |
| mismatch_delay / MPC | 速度 RMSE | 0.5381366 | 0.5312466 | −1.28 % |
| mismatch_delay / MPC | 高度 RMSE | 125.23438 mm | 125.23435 mm | −2.6e-5 % |

**俯仰 RMSE 六项全部逐位相同**：2.6631843 / 2.2355434 / 2.6798492（LQR 的 mismatch / nominal / push）与 2.5860889 / 2.1689709 / 2.6153177（MPC 同序）。

此外 `success`、`normalized_effort_rms`、`max_abs_pitch_deg` 三列**逐位不变**（最大差 0），`mean_reward` 最大差 2.28e-3。求解时间是墙钟测量，每次都不同，不做比较。

## 怎么读这些差异

- seed 固定为 11，**不是随机性**。
- **俯仰逐位相同、速度与高度系统性偏移**，与接触/积分细节的版本差异一致，不像是控制器逻辑改变。本仓库另有一处独立记录提到在 MuJoCo 3.12 上发现并修复过 `qM` 的潜伏缺陷（见[研究日志](../../docs/research_log.md)的 C36–C38 一节），说明该项目对 MuJoCo 小版本差异是敏感的。
- `mismatch_delay` 的高度 RMSE 几乎不动（2.6e-5 %），因为该场景下高度误差被失配本身主导（125 mm 量级），接触细节的影响被淹没。
- **定性结论不受影响**：MPC 与 LQR 在速度 RMSE 上仍相差约 0.005–0.006（标准差 0.16 量级），即统计上无差别；两个经典控制器在 `mismatch_delay` 下高度误差仍同时劣化到 125 mm。

所以读这张表时：**百分之几的差异是版本容差，不要当成控制器变化**；要比较控制器，必须在同一 MuJoCo 版本下比较。

## 已知缺口：输出不带版本记录

`experiments.py` 不写任何版本记录，所以本目录的历史产出没有 provenance 文件 —— 这正是上面那张差异表需要手工补的原因。仓库里其实有 `src/wheel_legged_control/provenance.py` 的 `capture_git_provenance` 可用，但 `experiments.py` 从未调用它。

**此项未修复**：`src/wheel_legged_control/experiments.py` 属于 77 份只读冻结文件（见[独立项目的保护回执](https://github.com/LYHrmer/ddt-d1-rl-lab)的 `protected_guard_final03_01.json`），冻结 SHA-256 为 `5c8a7a273fd3219c549c005a7d6996de66600a4fcf54c409b88c4d3cd49a7ed4`，不在本轮改动范围内。重跑时请手工记录 MuJoCo 版本，或先解除该文件的冻结约束再改。

## 残差行需要可选依赖

`--policy` 默认指向 `results/residual_ppo/model.zip`（已在库中），但加载它需要 `[rl]` 可选依赖：

```bash
pip install -e '.[rl]'
```

**这个依赖不是可选的。** `main()` 用 `args.policy.exists()` 把残差行设计成可选，但紧接着无条件 `from stable_baselines3 import PPO`，而权重文件就在库中 —— 所以只装基础依赖时命令会直接抛 `ModuleNotFoundError`，而不是降级到两行。

**此项同样未修复**，原因与上一节相同（`experiments.py` 在 77 份冻结文件内）。规避办法：装 `[rl]`，或用 `--policy /nonexistent` 让存在性检查落空而跳过导入。
