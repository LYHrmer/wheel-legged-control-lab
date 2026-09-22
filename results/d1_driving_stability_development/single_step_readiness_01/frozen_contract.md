# 唯一下一合同：真实单低台阶的滚越任务资格（下一终端执行）

选择：先建立 **native plane + 单个 15 mm box** 的真实碰撞任务，完成零积分几何前检及一次固定零残差低速滚越 readiness；不做第三次 flat-jump 增训。目的为低台阶滚越 RL 提供可信任务和失败基线。原地四轮同时净空 20 mm/腾空 20 ms 不是低台阶滚越的必要前提；跳跃需求仍未完成，不删除或改判旧跳跃门槛。本轮仅冻结合同，不执行它。

## 1. 新 plant，几何与查询须一致

只增加独立模块/测试/runner及新 plant/task 身份，不改 frozen77、原 native-plane、旧 GUI 或旧记录。使用原 URDF/机器人惯量、碰撞几何、关节、执行器、solver、摩擦/接触 margin；地形只有原生无限 plane z=0 和至多一个静态 box，没有 hfield，也不导入全 course。新增 box 必须在 MjSpec compile 前写入；编译后按名字重新绑定所有 ID，不能仅替换 model 而继续使用旧缓存。

尺寸取原 `src/wheel_legged_control/d1/terrain.py::_add_stairs` 的第一阶：全长 0.36 m、全宽 1.24 m、高 0.015 m，即半尺寸 (0.18,0.62,0.0075)。设初始 base 水平位置为 (x0,y0)，heading=0，box 中心为 (x0+0.70,y0,0.0075)，无旋转。此处是一个有限长的低台阶/台块，不称为完整楼梯。面摩擦用原名义0.9，box接触维数/材质等沿原真实box helper，不调碰撞参数。

水平地面高度查询必须对应实际几何：box水平投影闭矩形内返回0.015，其余返回0；边界规则显式记录，侧壁是几何不连续，不能伪造连续坡度。校验query与真实compiled box/plane ray或距离结果；query只是几何量，不能当载荷/穿越证据。当前零残差命令保持原世界z参考0.455m，使用真实地面高度与原控制器的世界z减地面高度语义；不要为隐藏地形切换自行加高度前馈。新元数据明示这一语义。

## 2. 先做0积分前检；失败不花物理预算

同时构造同版本的 obstacle_enabled=False/True 两条件，分别只有plane、plane+该box。逐名字比机器人惯量/关节/执行器/几何和solver/材质；允许差异仅为该world box及其身份/ID。核验初始robot与box分离、与旧flat相同初态和dt。对固定的plane、box顶面、前立面、边棱合成接触姿态进行collision/kinematics检查，禁mj_step/step1/step2；如调用mj_forward，单列次数，不能把其静态求解当历史动态载荷。

contact frame需按geom顺序解释，验证方向/正交性和对应box特征。真实立面水平法向及边棱法向是正确几何，不能沿用flat-plane的“所有法向竖直”判据。原轮平面净空函数不能拿来冒充box距离；新增针对compiled box的轮/全机器人碰撞几何界限与接触分类。装饰visual geom不进入物理通过判定。前检、纯计数/归档测试和source manifest通过后，才允许以下一次有界物理。

## 3. 固定两场，共最多2400控制/12000 native

顺序 plane-only，再 plane+box；各一次1200控制步/6000 native，control/native dt=0.01/0.002 s。同seed77301、原nominal reset、oracle synchronized、同一已验证 StopTurnCompositionController、每拍zero8、无policy加载/学习、无外力。直接使用原组合核心并绑定当前raw命令，不复制/更改增益、stop latch或yaw条件。新增plant env不能调用flat-only validator来忽略box。

raw yaw=0、世界z参考=0.455m；forward严格为：k=0..199取0，200..999取+0.2m/s，1000..1199取0。无跳跃指令、无release/ramp。stop latch从1000起按原实际消费forward非零→零激活；authority纯turn门始终不激活。这个baseline只为暴露真实接触/通过与失败，不进行经典控制调参。

保存每次实际native调用的qpos/qvel/ctrl、完整接触geom/body/frame/位置/法向载荷和终端收据；endpoint/controller字段按T/T+1保存，5T子步，部分失败不填充、不重试、不扩容。两条件比较初态和首次box物理接触前的共同状态/扭矩/原始命令前缀；geom ID、地形配置不同应按名字映射并单列，不能误称整个模型逐位相同。若接触之前已有不可解释动力学分岔，应停在记录/plant语义诊断，不调控制补偿。

## 4. 把记录有效、接触资格和任务通过分开

记录有效要求有限状态、真实native时序/额定保护、无未知wrench/地形、原动作/命令/source身份正确。box接触资格要求实际轮-box正载荷接触被识别，并能解释对应前立面/顶面/边棱；未接触只能记曝光不足，不能声称已验证真实滚越接触。

机械通过要求真实碰撞几何沿道路前进越过box：至少存在轮-box实际载荷接触，四轮及所有机器人collision geom最终均在box远侧（各geom真实/保守世界x最小值 > x0+0.88 +原contact margin），不是仅base中心越线。轮-box接触允许；任何非轮机器人-box/plane接触计失败。原倾覆/位置/速度/额定力矩保护保留，全程|heading|<=5°、|roll/pitch|<=10°、|y-y0|<=0.1m。最后100控制interval以实际端点1101..1200检查|body vx|<=0.03m/s、|COM vz|<=0.03m/s、原world-z高度RMSE<=0.015m；对应最后500 native每轮正法向载荷占比>=0.95。碰撞几何只证明采样时刻范围，不能夸为未采样连续时间无穿透。

滚越不要求四轮同时腾空，不因没有20ms flight判失败。任何失败照实归档；记录有效的zero失败可作为下一独立RL任务合同的基线，不要求先用经典控制把zero修到通过。若几何/记录本身无效则先修该具体模块，不进入训练。下一RL训练预算此时尚未授权自动启动；需基于本唯一readiness的真实机制再冻结，一次只选一个任务/动作假设。不得更换高度、速度、控制增益或尝试更多场来刷通过。
