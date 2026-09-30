# C23 v03：剩余键盘段的按钮标签修正

审阅：实际 gpt-6-astra ultra，`/root/astra_mainplan19`。本文件在唯一 script GUI 600 controls 完成、实际键盘段尚未开始时制定。已执行的原 headless、v02 GUI、原来源、入口、manifest、GO、reader、截图与错误日志全部保留。无任何物理重跑或新增预算。

实际 `script_gui_02/frame_drive.png` 的 STOP/RESET 标签在右缘重叠、被剪裁；本审阅者已直接看图，截图 SHA256 为 `69c33cb953da1936099e4a7f7c686d36c7b2969597ee02189d65671fe0178c14`。矩形和点击命中区域正确，标签调用 `mjr_text` 时归一化坐标与最后矩形 viewport 不匹配。

只新增 `run_gui23_v03.py`，将两个标签分别交由各自按钮矩形的 `mjr_overlay(font, TOPLEFT, button_rect, label, '', context)` 绘制。必要的 actor B/zero HUD 标识同属纯显示文本；除此之外，模型、控制数学、输入/准备消费时序、I/O 每五控制发布、记录和计步都须保持与 v02 相同。原输入命中矩形、键盘表、seed、1200 上限、publisher pause 和停止窗全部不动。

`reader23_v03.py` 唯一差异为精确允许第三 worker 文件名及对应错误文本。实际来源仍须与 session argv 和新 GO worker 相等，15 个模块 origin 逐项精确核验；所有读回数学、模型/native 计数、物理安全、GUI 性能与原末100停止门不变。新 manifest/入口独立版本化，不覆盖 v02 冻结文件。

最后 GO 仅授权原 keyboard_gui 一次、不超过1200 controls；执行前必须将实际 v02 script GUI 的完整同轨与性能读回、其 reader 宿主成功回执作为绑定前置。完成后必须对原输入表全部路径、一次真实 R reset/第二次拒绝、物理安全、RTF≥.8、FPS≥8、poll/snapshot p95≤250 ms、最后100点 raw/servo 全零及原 .04/.05/.02 停止门逐项验收，并看实际键盘 PNG 确认两个完整标签位于各自可点击按钮内。前两脚本的原停止失败继续公开。

这次键盘段可继承已完成 v02 headless/GUI 数值同轨证明，范围仅限经静态确认的纯显示变更；不得声称 v03 再次实测完成脚本同轨。总 GUI 预算仍最多2400 controls/12000正常native/+6compiler；已经完成1200/6000/+4，剩余最多1200/6000/+2，模型剩余最多1 PPO load、3 torch loads、32 probe rows与1200控制推理，无训练、无优化、无追加物理尝试。本文件不预写键盘、GitHub或CI成功。
