# C18-C精确来源GO

2026-09-30，Astra：**GO，固定qualification_spec_18.json的primary和mirror两个worker，共24,000 controls。** 开发候选的rough与ramp已通过真实执行和完整独立读回。必须先由已审freeze18.py成功核验并写入初态五数组配对等全部前置，生成plan_go_C18.json，才可启动worker；任何freeze失败不绕过。此GO不预判最终七场景结果。

## 已完成的开发证据

逐项读取`development_readback_18.json`与worker、host和reader host receipts。唯一开发worker完整4,000 controls、20,000 normal native、2 compiler，PPO load1/torch.load3、predict API4,001/actor rows4,032、critic/train等0。host用时93.969738002s，reader用时54.644227807s；首异常、cleanup、warning、源变更和存活自有进程均无，归档/模型/计步闭合。

rough hold均速0.3500110834189388 m/s、RMS 0.04191901909011142≤0.05，原记录、安全、地形通道、完整1600步与最终停止均通过；C17同策略RMS为0.05339427199909047失败。ramp均速0.4710231949423562位于0.45±0.04，RMS 0.023834166751352408≤0.05，完整1800步与全部门通过。up/deck/down真实正轮载荷计数2179/1427/2168；四轮碰撞几何最小x分别8.14283447、8.14391296、7.70301802、7.74067897，均超过实际坡道最远边界6.89826798 m。floor独立门和完整controller/native/contactForce链通过。

root已报告C17与C18对应rough/ramp的qpos、qvel、ctrl、qacc_warmstart、observation逐位相同。freeze18的资格路径将重新核这五数组并把结果及两侧初态身份写入C plan，作为运行前的实际硬前置。该比较不产生模型或物理步。

全部19个运行/纯检查源码当前仍与B的测试receipt hash相同，未改运行逻辑，不重复已通过的10项纯测试。新z状态、独立checker和reader连续性、exact WorldUprightCourseEnv及完整5T证据链均沿用B冻结来源。

## 绑定身份

| 文件 | SHA256 |
|---|---|
| qualification_spec_18.json | 2858d04964ca69f7414dfb7c0c48ee4c62b7989d4d742fe617157534509d92ad |
| qualification_contract_18.md | b05038140660c06b6e550f9e4c669f896da2b96b6659327a0eef4dacd7d3434d |
| plan_go_B18.json | 4ab9645b3f07b9db269c6713b502e716f7986c5ef3265a3c24cab7d7685a7630 |
| development_readback_18.json | 4a7b06b8f6142c093992404cba617486d03684d4355118337630a5025a61478c |
| development_readback_18_host_receipt.json | 4fe01ab063fcac0f42a05da28b98e1c0d40c1888512188ca0f16da913cead87f |
| development_filtered_01/host_receipt.json | 98799dfae6d843f1dc62be50e5773bee6ed8b90eac178c0abd5006dc3a1ee370 |
| development_filtered_01/worker_receipt.json | afcccd548af4712639aae0b1f73d4c03262d2a1fb5b3c412e23a9a63bbebecdf |
| pure_tests_receipt_18.json | 60897dc5fbdf832d9aee49a212249846f67ae0397dcfb485ae79cd6306709ea9 |
| freeze18.py | d284b1e398c0561549092f83df3715e3ddee01eda3d7aa45d3617022f0cbc1fe |
| residual18.py | a21d6ad7e6507530b2b441305ddbebe0267727cb12e6ed8ad592bca5d1b208c4 |
| controller18.py | b8f3b1d59880ca0bff7d7d4791ba823a101d396dd9f614e25414b83bd2133aba |
| verify_control18.py | afcbfecbd1b6d176fe26d0559adc67820e0867a0af68cc61bd5ce2378cf62fb3 |
| read_eval18.py | 09c579ae6503b536ff22809cde49ee464bab15bf1254eb122ec875c2ac990f75 |
| host18.py | c9d75008ef01548f1a9c71159e59072b434173fcf7de4d38c3d4734539fc9a58 |
| worker18.py | 4129eabbb3ae6c0a717808c629ac769b3b6153a003c5d360935bd63eedc02d3c |
| eval18.py | 8658bdad7941e0e6474f5e11594abff57734344d967b003bd47f1086f1bef6ec |

其他已测试源码和完整历史闭包由B plan与测试receipt绑定；freeze18资格阶段直接增加完整B归档和独立reader receipts。grouped ZIP仍为`1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`。本表不代替计划的全来源核验。

## 固定执行和完成条件

两个worker均为combined，tau0.20/alpha1/21、yaw gain4/limit1.2，16RL通道继续实际执行，不训练或选checkpoint。primary=floor181190+原六任务seeds181201–181206，每项zero→grouped，20,200 controls；mirror=floor181390+反号yaw181403，zero→grouped，3,800 controls。命令、地形、时间窗口和所有原数值门不变。

C总24,000 controls/120,000 normal native/+4 compiler、2 PPO/6 torch、12,602 predict API/12,664 actor rows。primary soft/close/hard1200/1320/1380s，mirror600/720/780s；每个独立正式reader295+5≤300s。C18开发+资格合计上界28,000/140,000/+6，沿用原合同，不借失败/剩余额度重试或扩展。

只有grouped七场景全部通过，且两worker来源/计步/配对/完整独立物理读回闭合，才能报告转向、完整坡道和六任务回归完成。C17的rough失败不抹去；B18通过不能替代尚未执行的最终资格。RL贡献只用primary原六对按旧门另判，基控修复收益不记作训练收益；seed不构成独立物理随机样本，不扩张到GUI默认/自由驾驶/实机资格。

Astra本次仅审阅已保存文件、核来源身份及写本GO，没有新增测试、模型或物理执行。现在停止修改C阶段冻结输入。
