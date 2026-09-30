# C24 静态实现边界

固定真实场景为原生水平 plane 与唯一箱体 `terrain_single_15mm_box`。箱体中心 `(-3.1,0,.0075)`、半尺寸 `(.18,.62,.0075)`，顶面真实高度 15 mm，前后缘 -3.28/-2.92 m。初态 `(-3.8,0,.455)`、单位姿态、零速度。命令仅在控制 tick `[200,1000)` 请求前向 0.20 m/s，其余为零；yaw/lateral/jump 始终为零。

继承 C18 的 0.5 m/s² servo 后，名义 release 位移 1.561 m、停车尾段后位移 1.600 m。实际 reset 后使用全部碰撞形状的真实编译尺寸与姿态检查初始前缘间隙及固定姿态平移下的后缘空间，并检查实际 box/plane oracle 查询。该检查不是物理越障或停车通过结论。实际 qualification 必须完整记录与独立读回。

`SingleStepPlant24` 直接继承冻结 `scripts/d1_single_step_plant.py` 的真实单箱体模型构建器，只新增由这两个真实原生碰撞几何建立的 `CourseGroundMap`。没有先造 92 几何场景再替换模型，也没有 fake-flat oracle。`SingleStepWorldEnv24` 显式构建该生产 plant、冻结 C18 combined controller 与 world-upright loop，继承同一 reset/step/99D/16D/servo/奖励路径。

旧台阶的 8D controller 与世界高度命令没有被冒充复用。本次控制语义为 C18 的局部真实地面净空命令 .455 m；台阶上参考世界高度相应增加 .015 m。末窗已经全体机器人离开箱体，因此保留旧 `.455 m` 世界 base-z RMSE 门有明确意义。

机器人源码来自同一冻结 URDF。实际新编译 binding 与旧 C22 保存 binding 直接比较质量、关节、驱动映射及排除 world 后全部 robot geom 类型/尺寸/碰撞掩码。旧 binding 未记录的惯量来源明确标为同 URDF 的源码推导，不声称旧数值曾被保存。实际新 solver、阻尼、armature、摩擦参数另保存并核对共同构建器固定值。

每 endpoint 的全体 body COM 与碰撞形状由保存 compiled tree、qpos、qvel 纯算术重建；记录时再交叉核实际同步 measurement cache 的 xipos/geom pose 与每个 body 的原生只读世界速度。新增 `mj_objectVelocity` 查询数单列，18 bodies、两条满轨迹时为 `2×1201×17=40834`；不作为 normal-native 积分次数，也不声称这是全系统所有只读调用总数。重建不新增模型、data、forward 或 kinematics API 调用。

保留旧严格任务门：完整 1200 controls；至少一次真实轮—箱体正 native 法向载荷；无任何非轮地面候选接触；世界 roll/pitch ≤10°、heading 变化 ≤5°、侧向偏移 ≤0.1 m；净空 ≥.28 m；endpoint 275–875（含两端，601 点）均速 ≥.18 m/s 且对 .20 RMS ≤.05；所有机器人碰撞形状 `min_x > -2.92 + 自身margin` 且包含四轮；末 100 最大 |body vx| 和全系统 COM |vz| 各 ≤.03、base-z RMSE ≤.015；末 500 native 各轮正载荷比例 ≥.95；末 100 mean |yaw rate| ≤.05、净空 std ≤.02。

唯一 worker 顺序 zero→B，每条至多1200，共至多2400 controls /12000正常native，冷模型与 nominal cache 共预计2 compiler native。B22 ZIP只加载一次，torch.load 3次；保存32行probe与至多1200真实策略决策，总actor行至多1232，无训练/优化/critic前向。两条轨迹五初态数组及完整控制重置状态直接配对。

本次源实现与测试文件均仅编写，未由作者执行测试、模型、物理或策略。实际所有计数、SOURCE GO、唯一运行、独立读回由 root 所有。单个固定场景通过只建立该场景下工程资格；不证明已训练真实台阶技能、RL优于零残差或统计泛化。
