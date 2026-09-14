# A/D 短按提交完整侧步：有限开发验收

课程入口默认 `commit_cycle_v1`。空闲时短按 A/D 请求一整步，长按在完成边界重复；松键只停止
后续重复，本步继续。反向只在下一周期开始时生效。X、失焦、输入超时或恢复状态走明确的安全取消；
取消后要真实释放 A、D 才能重新请求。原 `--side-input-mode release_abort` 对照保留。

方案由 GPT-6-astra / ultra 冻结，纯意图核心实际调用 Claude Opus 生成（实际返回模型
`claude-opus-5`，费用 $0.32844125）。原稿逻辑测试 32 通过、1 失败；本地修正非空字符串的验证边界、
文案例外和注解格式后，纯核心 33 项通过。薄集成最终针对性测试 89 项通过。
原直接调用 drive、Fast/Conservative 两个物理侧步控制器、步长和步态参数均未改。

九场固定矩阵累计 **19,900 个真实控制转移、107 项门槛通过**。矩阵显式选择新模式，全部通过后
才切换默认值；随后通过默认参数再次运行 2,000 步短按 A，并在活动中发 Space。
该请求被明确记录为阻挡，直到回合结束没有延迟跳跃。合计 **10 场、21,900 个控制转移**。

| 场景 | 结果 |
| --- | --- |
| A / D 短按 1 秒 | 各 10.56 秒完成一周期；净侧移 3.703 / 3.704 cm |
| 交接轮式控制后 4 秒 | 保留 98.30% / 98.29%；最大回退 0.917 / 0.913 mm |
| 长按进入第二周期后松键 | 恰好完成两周期，3.703 / 3.589 cm，没有第三周期 |
| A 开始后改按 D | 第一周期保持原方向，完成后才开始反向周期；3.703 / 3.564 cm |
| 支撑 shift / 实际 swing 时按 X | 均保留 `cancelled`，四触点且无待落腿后才交接，没有按住 A 自动重启 |
| 非平地、跳跃、显式 reset | 坡道启动 guard 仍拒绝；跳跃期间侧步短按丢弃；reset 后不以合成零方向解除释放门禁 |

两次取消不是成功侧移，也不保证回到起点：最终沿原侧步方向分别约 -2.88 / -2.86 cm，保留了支撑
重心移动的真实位置。一次活动周期被显式 R 仿真复位中断，按独立 segment 保存，没有跨复位拼接位移。

这些是自动注入 GLFW 等价事件/held 采样后，经过实际课程 runner 和 MuJoCo 的有限检查；
**没有人工操作 GUI**。当前快步仍约 11 秒一周期，初期身体可能先向相反方向移动。
结果不证明更快步态、跨地形横移、完整 G1 驾驶验收或跳跃目标完成。保守步态的新输入路径未在此矩阵实测。

## 复核资料

- [冻结契约](provenance/side_intent_contract.md)、[核心与接缝独立审阅](provenance/side_intent_independent_review.json)
- [Opus 原稿](provenance/d1_course_side_intent.original.py)、[调用回执精简副本](provenance/opus_receipt.reduced.json)
- [原稿失败测试](provenance/original_logic_tests.xml)、[候选通过测试](provenance/candidate_logic_tests.xml)、[最终 89 项集成测试](summary/integration_tests.xml)
- [九场矩阵](summary/matrix_summary.json)、[默认入口与 Space 互斥验证](summary/default_summary.json)、[本地集成与范围](summary/local_integration.json)
- [物理验证 runner](helpers/run_side_intent_validation.py)、[默认入口验证 runner](helpers/run_default_side_validation.py)
- [逐文件原始记录核对](original_record_audit.json)、[无损恢复校验](bundle_verification.json)
- [独立物理抽查](summary/independent_physics_spot_audit.json)、[抽查 helper](helpers/audit_side_intent_spot.py)、[原始数据 SHA 绑定](summary/independent_physics_audit_binding.json)：仅独立复算 A/D 短按与 swing 取消三场，0 个新物理步骤，全部通过。

## 完整原始证据与共享对象

`evidence_index.json` 包含 130 项原始记录，合计 **909,627,364 字节**；其中 116 项是十场完整原始
运行文件。每场的 qpos/qvel、实际 16 路力矩、输入事件、逐拍 telemetry、物理侧步 trace、协议、
manifest、model.mjb 和 source.tar.gz 均保留。恢复后，九场在 `runs/matrix/`，默认验证在 `runs/default/`。

新增本地 blob 为 **24,127,242 字节**。另有三个相同对象复用相邻
[course_braking](../course_braking/README.md) 的 immutable blob，以旧 evidence_index 的完整 SHA256
绑定；没有修改旧归档，也没有再次上传相同的约 80 MB 编译模型。需要同时保留两个相邻目录。

```bash
python results/d1_driving_stability_development/side_input_commit/restore_evidence.py --verify-only

python results/d1_driving_stability_development/side_input_commit/restore_evidence.py \
  --output /tmp/side_input_commit_restored
```

输出目录必须是新的，完整恢复需要约 0.91 GB。校验会检查共享 index、blob 的压缩/解压 SHA，以及
每个原始文件的字节数和 SHA。仅 source.tar.gz 的 gzip 时间字段被规范化以去重，原字段另存，恢复时逐字节还原；
其余文件只做无损压缩或原样存储。`readable_copies.json` 把便于阅读的副本绑定到相同原始记录。
