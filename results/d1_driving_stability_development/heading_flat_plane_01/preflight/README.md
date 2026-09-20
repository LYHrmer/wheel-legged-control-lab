# Flat-plane 非物理前检

2026-09-20，gpt-6-astra / ultra。结果：`report.json` 的 `passed=true`。本目录及 `tests/test_d1_flat_plane_env.py` 是本次新增输出；没有修改 plane 模块、runner、冻结源或旧结果。

| 检查 | 结果 |
|---|---|
| 专项单元测试 | 60 项通过，见 `pytest.xml` |
| 新物理积分 | 0；测试和审计均将 `mj_step`、`mj_step1`、`mj_step2` 替换为立即失败的 guard |
| 初始化、reset、重复 prepare | 所有主/measurement/snapshot data 时间为 0；命令在每次 reset 仅消费一次 |
| 原机器人、执行器、接触材质和 solver | 188 项 compiled model/option 字段相同，详列 `compiled_invariant_fields` |
| 冻结输入 | 77 份 SHA256 全部符合原 budget protocol；原 G1 rev2 protocol SHA 也匹配 |
| 保存姿态 | 9 个转向 + 28 个停车，共 37 帧；207 个 active wheel-floor contact |
| Plane 接触法向 | 所有接触归一为 ground→wheel 方向后均为 +z；最大水平分量 0；无接触帧 0 |
| MuJoCo | 3.12.0 |

测试覆盖有限解析地面查询（含边界、域外、bool、非有限值）、严格 plane 身份校验、原 spawn/关节初态/零速度/零 warmstart、新 plant 的 provider/loop 绑定、85 维观察和 8 维动作、metadata 实际 terrain 与请求配置分离及复制隔离。原 controller/source/observation/reward 身份保持，新的 task/config 同时包含 plane 身份。原 85 维 heading sidecar 分别由 heading config 检查和 generic task_schema 检查拒绝；模型路径不存在，PPO.load 同样被封锁。非零/非法残差在 parent step 前拒绝；合法零残差仅在 spy parent 上验证单次委派，没有执行真实 step。

旧 hfield 与 plane 的差异限于 floor 几何表示及相关 asset/派生缓存；不要求 floor 的 hfield/plane size、dataid、包围盒等相同。Robot geometry 全字段、body/joint/dof/actuator/material 数组、floor 接触材质和全部可读 solver option 都比较。模型构造和 reset 允许原实现调用 `mj_forward`，时间仍为 0。保存姿态另建 MjData，仅调用 `mj_fwdPosition`，逐帧断言 qpos/qvel 不变且 audit time=0。

37 帧来自旧 hfield 的已存轨迹；`source_endpoint_tick` 保留历史索引，不能与审计的零时钟混淆。接触 frame 第一行为 geom1→geom2 法向，若 ground 为 geom2 则取反，得到 ground→wheel。`efc_address>=0` 仅标记该碰撞位置缓存中的 active constraint；审计没有为历史姿态求解或重构接触力。没有接触的帧会单独记为 `not_observed_no_active_contact`，不会凭空通过法向检查。本批恰好没有这种帧。

本前检不建立 plane 上的新轨迹，也不证明停车、转向或任何原任务门槛通过。5600 控制步的物理合同及 runner 收据审查由 root 独立负责。后续若改变被测 module、test 或输入，现有报告对应的版本就不再覆盖该改变，必须重新归档相应非物理检查。

被测 module `scripts/d1_flat_plane_env.py` SHA256：

```text
145c58e4201a657b76e1608263d3a4cea57eb1c0f39530fb47faf32b19026a51
```

`report.json` 记录 96 个唯一输入路径的 SHA256，包含 77 份冻结源、URDF/mesh、新增 module/test/helper、合同、原 protocol、旧诊断报告、8 份 states.npz 及本次 JUnit 收据；执行末尾再次验证全部输入未变化。`checksums.json` 保护本目录输出（不包含它自身）。

复算时使用新的输出路径，避免覆盖本次收据。以下命令从仓库根目录执行；如果 `/tmp/flat_plane_preflight_reproduction.json` 已存在，请选择另一个新路径：

```bash
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps:/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/test_d1_flat_plane_env.py -q -p no:cacheprovider --junitxml=/tmp/flat_plane_preflight_reproduction_pytest.xml
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps:/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920/flat_plane_preflight_01/audit_preflight.py --repo /home/lyh/wheel-legged-control-lab --work /home/lyh/wheel-legged-control-lab-work/recovery-20260912/stability_20260920 --pytest-junit /tmp/flat_plane_preflight_reproduction_pytest.xml --output /tmp/flat_plane_preflight_reproduction.json
```

复算 helper 拒绝覆盖 report；pytest 自身不保证 JUnit 输出独占，因此也应为重跑选择新的 JUnit 路径。没有参数扫描、训练、settle 或任何物理 smoke test。
