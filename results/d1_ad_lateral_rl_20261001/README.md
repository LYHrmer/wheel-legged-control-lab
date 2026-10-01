# C31 侧移 RL 研究快照

训练与十场仿真物理资格通过，四组预注册提速门均未通过。RL 相对固定参考仅缩短 0.10%–0.41% 的周期，不能作为更快 A/D 控制器交付。

正式说明见 [实验结果](../../docs/ad_lateral_rl_20261001.md)。`continuation31/final_review31.json` 保存实际 Astra ultra 终审；训练、评估与保存诊断报告及唯一最终侧移权重均在本目录。权重独立于 B22，GUI 未接入。

`publication_manifest31.json` 列出1612份逐字节复制文件及1854份完整训练/评估库存。约2.07 GB原始归档留在本机，其中controls/native/states和超过10 MiB的训练worker_receipt没有上传。pytest临时current链接仅作排除记录，实际fixture目录按规则保留。公开子集不足以独立重跑完整读回，也不是便携运行包。

`publication_tools/`保留初次打包拒绝pytest别名的记录与修订后成功记录；原实验记录未修改。
