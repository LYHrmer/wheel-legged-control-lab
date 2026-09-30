# C18-B精确来源GO

2026-09-30，Astra：**GO，仅执行development_spec_18.json唯一development worker，最多4,000 controls。** 不授权在本GO内运行资格阶段、其他参数、重试或训练。C17的6/7失败结果保持原样；本次只检验一个机制相关候选，尚无新物理成功结论。

完整审阅新residual/controller、独立checker及reader增量、runtime移植、freeze18和测试来源。新z为共同wheel/body误差差值的一阶状态，tau0.20s、alpha1/21；active用z_after=z_before+(mean(ew)−eb−z_before)/21及eI=ew−z_after，inactive/reset清零。I状态、原yaw、RL16、P、限幅与实际native执行链保持合同定义。reader以实际pre-state和servo重算，并检查z从零开始的逐tick连续性，不调用live控制算术。

root实际纯检查全部通过：control 9项测试；独立进程checker一项包含完整记录接受及z_after/alpha篡改拒绝；clean floor reader导入和fatal ruff通过。新控制测试与禁止physics导入的checker测试分进程执行。`changed_sources=[]`，模型/control/native/optimizer均0；当前19个源码hash全部仍与测试receipt一致。首次物理前发现的checker字符串数值转换问题已修复并被该真实测试覆盖。host18旧freeze入口已删除，实际冻结仅用freeze18.py。

## 精确身份

| 文件 | SHA256 |
|---|---|
| plan_18.md | 56019932e5372c2cdc2465d076f731218954382821b7fc43ff0500508a32ff27 |
| intervention_contract_18.md | e90ff9a58d9556d4a442959f67069304fadee99ab8a8eb12aa636d8a20d0429c |
| development_spec_18.json | 725bbfc97a354dfadc474db168b6bb963d0cb3d711472da5ebcfc5c9feb049a8 |
| pure_tests_receipt_18.json | 60897dc5fbdf832d9aee49a212249846f67ae0397dcfb485ae79cd6306709ea9 |
| saved_rough_diagnosis_18.json | 42486219f0f318c7c479a83b65ae6f57b4b8e51f340c87be1dd3a026dca368ad |
| diagnosis_receipt_18.json | 11ecb9555baea1c36f7afa10b4d3bc2a6b34ab6cc374ccea2e841204ff9edcf8 |
| residual18.py | a21d6ad7e6507530b2b441305ddbebe0267727cb12e6ed8ad592bca5d1b208c4 |
| controller18.py | b8f3b1d59880ca0bff7d7d4791ba823a101d396dd9f614e25414b83bd2133aba |
| verify_control18.py | afcbfecbd1b6d176fe26d0559adc67820e0867a0af68cc61bd5ce2378cf62fb3 |
| read_eval18.py | 09c579ae6503b536ff22809cde49ee464bab15bf1254eb122ec875c2ac990f75 |
| score18.py | 7d8ae73496223d2e4807ae4205c410032784ba512663fa64d71c19c59c072a3a |
| recipes18.py | 673a67fd89d6700db78a97b20865c459104fbfe11ed48f4e8619912248467caa |
| worker18.py | 4129eabbb3ae6c0a717808c629ac769b3b6153a003c5d360935bd63eedc02d3c |
| eval18.py | 8658bdad7941e0e6474f5e11594abff57734344d967b003bd47f1086f1bef6ec |
| host18.py | c9d75008ef01548f1a9c71159e59072b434173fcf7de4d38c3d4734539fc9a58 |
| freeze18.py | d284b1e398c0561549092f83df3715e3ddee01eda3d7aa45d3617022f0cbc1fe |
| readback_host18.py | 8a46a516af7ef8893c1e3982131a9ea88e38a463f6c90650807a4eeac868489b |

其余floor桥/clean reader、两测试文件、纯检查host及诊断脚本的完整路径/hash列于已绑定的pure_tests_receipt_18.json；freeze18还直接绑定全部19源、完整C17档案与失败summary、四rough诊断输入及历史模型/源码闭包，不以本简表代替来源检查。唯一policy ZIP仍为grouped final，SHA256 `1fcfe833d8a7cfdbbc224c9007a35e44c035674ce948b51557434b77838476fb`。

## 执行边界

root可先用freeze18.py --phase development独占生成plan_go_B18.json，再按host18的--plan/--arm执行唯一development。顺序floor600(seed181090)、rough1600(seed181105)、ramp1800(seed181106)，后两者均grouped。floor独立全链通过才执行正式场次。最大4,000 controls、20,000 normal native、2 compiler、1 PPO/3 torch、4,001 predict API/4,032 actor rows；训练等全0。

host soft/close/hard600/720/780s，回收额外最多5s；独立正式读回295+5≤300s。实际发生的尝试均计账，不改原时间和任务门，不换参数或原位重试。只有rough原RMS≤0.05与ramp原均速0.45±0.04/RMS≤0.05及几何、安全、最终停止、来源/物理读回均通过，并与C17基线初态逐位配对，才能审阅后续固定24,000-control资格GO。

本次Astra只读审阅及写本文件；所有真实测试和后续模型/物理均由root拥有并执行。现在停止修改本GO及将被B计划冻结的输入。
