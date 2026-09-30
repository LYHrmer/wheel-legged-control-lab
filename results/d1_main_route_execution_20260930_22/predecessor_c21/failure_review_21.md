# C21 静态资格失败审阅

实际模型角色：gpt-6-astra ultra，负责合同、科学判断与源码审阅；本审阅只读已保存数据与源码，未执行模型、物理或测试。结论：C21不得启动机器人worker；它在完整有限几何前置资格处失败。该失败不涉及RL策略表现，也不否定B课程假设。C21的source GO从未签出，A1复用桥通过不等于全部执行资格通过。

root唯一正式预检完成160条训练+16条评估、184800次纯servo advance，约16.17s，0模型/物理。160训练全部通过；12评估通过；四条新yaw失败。原始回执 `passed=false` 与全部失败行、路径数组均保留。不得删除四失败行、重命名成通过，或将已过训练子集当作整个合同资格。

| 失败命令 | 实际servo净航向 rad | 名义末Y m | 具体失败 |
|---|---:|---:|---|
| dev_yaw_left | +.240 | −3.5195282339891882 | 超flat车道上界−3.6 |
| dev_yaw_right | −.240 | −5.880471766010819 | 膨胀max|Y|6.352635585782303；首坏点845 |
| final_yaw_left | +.224 | −3.580249485242919 | 超flat车道上界−3.6 |
| final_yaw_right | −.224 | −5.819750514757085 | 膨胀max|Y|6.291914334528569；首坏点870 |

净航向及末Y由root只读保存NPZ计算，未新增servo或物理；膨胀半径.4721638197714839m。这里的名义轨迹不是机器人轨迹。C21机制是原120/160/120与130/160/110窗口的raw净yaw本来就分别±.240/±.224，实际servo净航向与之相等。它与C20 B source18的“raw净0但slew后净−.10416”是两个不同问题，不得混称servo积分误差。

源语义审阅支持条件性reset推导：生产env显式采用oracle；oracle reset丢弃seed；plant.reset固定nominal joints、默认单位四元数、零qvel/warmstart，唯一出生点由schedule提供；loop依次重置plant/controller/provider，env再prepare(0)。CoursePlant与FullDriveLoop未覆盖该reset。几何推导用已保存各地形qpos姿态及动态数组同态作佐证，只平移robot geoms，不平移world。哈希保证这条推理所针对的版本，不是单凭哈希建立形式证明。未来reset仍未观察，必须在任何新动作前以实际geometry、五数组及完整控制reset核验。

后续已具体限定为新C22：四条yaw唯一改为100/200/100（dev430/530/730/830，final440/540/740/840，镜像反号），保留速度、幅值、drive/hold/release/horizon、出生点和评分门。只做新四条6400次纯servo资格，精确复用旧160训练+12评估来源；不重复184800全集，不搜索候选。C22成立需新四条实际资格通过及全部runtime/reader/source测试闭合；本审阅不提前授予GO。B仍从共同C15父模型开始唯一32768，A20_1复用；失败B20不恢复，C21没有robot预算消耗或退款。既定三worker物理上限不增加。

C21累计184800与C22候选6400分别记账；新有限命令通过只能证明名义几何可执行，不能写成RL收益、可靠性、15mm、GUI或自由操控资格。最终科学结论仍须新B全量读回、A20/B22首1024配对及固定开发/封存评估。

| 绑定文件（相对C21） | SHA256 | bytes |
|---|---|---:|
| `spec21.json` | `889b6819190b23983a3e3ea587aeff6f46df90b938b73c34a28805482166fff0` | 25618 |
| `geometry21.py` | `166734144ff69582fe1e7eef97be8fd49a008f133d10b50aaddc9aab5a9575ef` | 24120 |
| `run_preflight21.py` | `ad81c6798c9343156364e0d8f5d24540b9f308b76192595be9dbabf8b0c55d7c` | 5746 |
| `a_reuse_21.json` | `81d12d9579901eb017d2f5da2b279d4cf0021efb1402b65c1ad0257418d11954` | 1549654 |
| `finite_geometry_preflight_21/finite_preflight21.json` | `6d8ce10aca5977952a1c117e86af5c43cf0a2af8b0c8fafd678e212e951392ae` | 424440 |
| `finite_geometry_preflight_21/all_nominal_traces21.npz` | `541fd624cbc0aee2fd42bdac2830e963ec8fcf8642d6c243a6c3ab79435b504f` | 1339529 |
