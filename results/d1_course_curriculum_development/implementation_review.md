# Claude Opus 课程核心采用与独立审阅

有效候选原文见 [opus_curriculum_original.py](opus_curriculum_original.py)，调用 session 为 `7590a2c7-cc43-4c8c-a56b-3a3b9d0fdc29`。调用及费用摘要见 [contribution.json](contribution.json)，原始请求/响应保存在本机工作目录；采用位置为仓库 `scripts/d1_course_curriculum.py`。Opus 编写了 wrapper、独立随机数流、按实际转移计数的 reset 课程调度、完整/部分回合记录的主体。以下改动由 GPT-6-astra ultra 独立核对本机源码后完成，不能称候选原样通过。

1. 修正严重的地形索引误判：原候选认为 `terrain_index==0` 是平地并直接返回原配置。实际四条训练路均为非平地；改为 `level==0` 返回合法默认平地，其余所有 index 都按 level/3 缩放四项幅值。
2. 在任何 reset 副作用之前校验 schedule 类型、与回合相同的物理时长，以及 external command source 冲突。无效 schedule 不再提前结束当前回合、消耗 RNG 或替换物理环境。
3. 回合时长拒绝字符串与 Python/NumPy 布尔值；要求有限、至少一个 10 ms tick、严格符合 10 ms 格点，与底层契约一致。
4. 畸形/非有限 action 在进入物理 step 之前拒绝，保持实际计数和当前回合；有效 action 不改数值，继续由底层做物理动作映射与限幅。底层物理 step 异常原样抛出，并要求显式 reset，不能用可能已部分推进的状态重试或虚增成功转移数。
5. 修正 imports 与代码格式，保持变更仅涉及新增文件。

新增 `tests/test_d1_course_curriculum.py`，36 项定向测试全部通过，覆盖 16 个实际 terrain level/index 组合、非法 level/index、回合边界晋级、提前失败按实际转移累计、三条件采样流配对和复现、错误 reset 不消费 RNG/状态、显式 schedule、完整/部分/零步回合记录、负向净位移、异常 step 与关闭语义，以及真实 MuJoCo wrapper 与直接同配置环境的逐拍 observation、reward、termination 和 exposure 完全一致。

已运行命令：

```text
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps:/home/lyh/wheel-legged-control-lab/src:/home/lyh/wheel-legged-control-lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B -m pytest tests/test_d1_course_curriculum.py -q
rtk proxy env PYTHONPATH=/home/lyh/wheel-legged-control-lab/.local-deps python3 -B -m ruff check scripts/d1_course_curriculum.py tests/test_d1_course_curriculum.py scripts/run_d1_course_curriculum.py
```

这项源码审阅与实际训练分别记录。根 agent 随后已完成三组 smoke、四级探针和三组各 16384 步、32 s 回合的开发训练；实际曝光与模型独立检查见本目录 README 和 evidence_review.json。

`opus_core_01` 的退出码虽然为零，正文却含伪工具调用和错误配置，没有可用目标实现；它未作为代码采用。原始失败响应保留。
