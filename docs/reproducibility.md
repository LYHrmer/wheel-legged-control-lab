# 下载记录与复现实验

先区分两件事：复算已有日志，是检查记录的算术；重新运行仿真，是检查给定源码和环境是否
产生相同的运动。哈希一致也只证明文件没有变，不能证明模型等于真实 D1。

## 代码仓库和原始数据各放什么

Git 保存源码、学习文档、协议与摘要，也包含策略 checkpoint、演示视频和回归测试必需的原始文件。
新的逐步 CSV、NPZ、编译模型 MJB、源码快照放在同仓库的
[v0.6.0 Release 附件](https://github.com/LYHrmer/wheel-legged-control-lab/releases/tag/v0.6.0)。
历史已跟踪的数据仍保留，不改写 Git 历史。

整包上传两次因连接中断失败，Release 改用 128 MiB 分片。下载以下几类附件：

| 文件 | 用途 |
|---|---|
| `result_artifacts.tar.gz.partNNNN` | 原压缩包的有序字节分片，最后一片可能较小 |
| `parts_manifest.json` | 每片大小与 SHA-256，绑定原压缩包和下面两份元数据 |
| `archive_manifest.json` | 每个成员的路径、字节数、SHA-256，以及打包时的 Git HEAD |
| `SHA256SUMS` | 重组后的完整压缩包与 archive_manifest 的 SHA-256 |

清单里的 `packaging_git_head` 是发布打包版本，不是每次训练时的源码版本。
每个实验自己的 `source.json` / `source.tar.gz` 才记录当时的源码。
部分实验在尚未提交的工作树中完成，但保存了完整源码快照与前后 SHA 检查；
不能把发布 commit 倒填成它们的训练 commit。

0.6.0 补充了有限控制参考越界的失败终止记录，旧程序在相同位置可能直接抛异常。
新回合通过 `terminal_reference_handling` 元数据标明这一点；
正常轨迹没有因此改变，旧失败记录也没有重新标成完成。

## 下载后先校验

以下在仓库根目录运行，需要 GitHub CLI；也可以从 Release 网页手动下载全部分片与三份元数据。
目录必须新建，避免混入上一次下载。

```bash
mkdir downloaded-artifacts
gh release download v0.6.0 --repo LYHrmer/wheel-legged-control-lab \
  --dir downloaded-artifacts --pattern 'result_artifacts.tar.gz.part*' \
  --pattern parts_manifest.json \
  --pattern archive_manifest.json --pattern SHA256SUMS

python scripts/transfer_result_artifacts.py join \
  --directory downloaded-artifacts --output joined-artifacts

python scripts/verify_result_artifacts.py --directory joined-artifacts
```

重组器检查分片及整包哈希，只写新的输出目录，不解包、不加载模型。完成后得到原来的
`result_artifacts.tar.gz`、`archive_manifest.json`、`SHA256SUMS`，压缩包字节没有变化。
随后校验器只读这三个文件，检查压缩包和逐成员的哈希、大小、路径与重复项；
不接受符号链接、路径越界或不在清单中的成员。SHA 文件与附件来自同一发布渠道，
这不是独立签名认证。

通过后解到单独的新目录，保留 Git 中的原始文件：

```bash
mkdir restored-artifacts
tar --extract --gzip --file joined-artifacts/result_artifacts.tar.gz \
  --directory restored-artifacts --keep-old-files --no-same-owner --no-same-permissions
```

得到 `restored-artifacts/results/...`。不要直接解到正在运行实验的 `results/` 中。
同时保留分片、重组包和解包目录约需 7 GB，建议至少留 8 GB 空间。
准确解包大小由清单的 `total_input_bytes` 给出。下载中断时可只补缺失或损坏的分片，
不要修改清单里的校验值。

## 从日志复算

安装 NumPy 后，六模型的算术检查不需要 MuJoCo 或加载 checkpoint：

```bash
python scripts/audit_d1_locomotion.py \
  --matrix-root restored-artifacts/results --output results/my_arithmetic_check.json

python scripts/audit_d1_ppo_math.py \
  restored-artifacts/results/d1_v3_locomotion_ppo/wheel_leg_seed24000/updates \
  --output results/my_ppo_math_check.json
```

前者核对原始轨迹、命令和完整评测矩阵；后者重算 GAE、Gaussian 概率及裁剪相关量。
它们分别回答不同的问题，见[主线学习文档](locomotion_lab.md)。

## 重新运行时保留哪些条件

每个训练或评测目录的 `protocol.json` 记录参数、依赖版本、种子和道路。
当前代码的新实验命令见 [README](../README.md)；逐字节重做旧实验应先恢复对应源码
快照到隔离目录，不能只使用当前 main。

需要同时保留：控制器五参数及 schema、状态来源、历史长度、道路与命令、控制与物理周期、
初始 seed、依赖版本。默认增益下的旧三参数 sidecar 仍可按兼容规则加载；
改变低层增益或历史长度不能通过修改 sidecar 绕过检查。

本次主线记录使用 MuJoCo 3.12.0。MJB 是编译后的模型，不保证跨 MuJoCo 版本兼容；
渲染旧记录应使用它的版本。状态回放直接使用保存的模型与 qpos/qvel，不重新采样轨迹。
只加载自己生成或信任的 SB3 checkpoint，其序列化格式可以执行 Python。

## 加入自己的实验

所有新主线入口要求新的输出目录，失败也保留。历史脚本未必有同样的保护，
学习文档统一使用 `results/my_...`，运行前仍要检查目录是否存在。

先预测结果，再改一个参数。报告完整时长、终止原因和跟踪误差，保留对应的零残差对照。
提前摔倒的 2 s 轨迹不能凭较小 RMSE 赢过完整 60 s 轨迹。固定地图和三个训练 seed
也不足以证明可以直接迁移到真机。
