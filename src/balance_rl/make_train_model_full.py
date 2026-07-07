"""
生成"带机械臂"的训练模型 scene_train_full.xml
========================================
基于 combined/scene.xml(完整机器人,固定基座), 改造成 Isaac 训练用:
  1. 把 worldbody 里的整机(base几何+两腿+arm_mount)包进 chassis body + free joint
  2. chassis 底部加配重 box(降重心, 让平衡更易训) —— 用户决策
  3. 加地面 plane
  4. chassis 抬高到 0.18m(轮子贴地)

机械臂关节的"锁定到收拢姿态"不在这里做, 在 RL 环境配置(ArticulationCfg)里
用高刚度 + init_state 实现, 更干净。这里只管动力学骨架。

收拢姿态(第2步会用): arm_joint2=2.0, arm_joint3=2.5, 其余0 → 整机质心最低0.309m
"""

SRC = "/home/lbc/桌面/机器人作业/models/combined/scene.xml"
DST = "/home/lbc/桌面/机器人作业/models/combined/scene_train_full.xml"

with open(SRC) as f:
    lines = f.readlines()

i_world = next(i for i, l in enumerate(lines) if "<worldbody>" in l)
i_world_end = next(i for i, l in enumerate(lines) if "</worldbody>" in l)
inner = lines[i_world+1:i_world_end]

# arm_mount 是空 body(无质量) → 转USD会报负质量。给它加微小 inertial。
inner = [
    l.replace(
        '<body name="arm_mount" pos="-0.036 -0.18 0.287">',
        '<body name="arm_mount" pos="-0.036 -0.18 0.287">\n'
        '        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>',
    )
    for l in inner
]

# 不加地面! Isaac 环境的 TerrainImporterCfg 已经会建地面。
# ★加回 freejoint! 不加的话 MJCF importer 默认把 chassis 焊到 world(rootJoint固定),
#   导致 Isaac 里 chassis 浮不起来(纯重力 z 纹丝不动)。
#   加 freejoint → importer 不焊死, 转USD后会多一个 worldBody root,
#   再用 fix_usd_root.py 删掉多余 root 即可。这是双根 vs 焊死两难的正解。
# 不挂外露配重盒, 改为给车身 base_link 一个真实 inertial。
chassis_open = (
    '    <body name="chassis" pos="0 0 0.18">\n'
    '      <freejoint name="chassis_free"/>\n'
    '      <inertial pos="-0.036 -0.18 0.05" mass="8.0" '
    'diaginertia="0.08 0.08 0.08"/>\n'             # 车身8kg,重心压到z=0.05(轮轴下方):降重心治平衡难+训练数值崩
)
chassis_close = '    </body>\n'

inner_indented = ["  " + l if l.strip() else l for l in inner]
new_lines = (
    lines[:i_world+1] + [chassis_open] + inner_indented
    + [chassis_close] + lines[i_world_end:]
)

with open(DST, "w") as f:
    f.writelines(new_lines)

import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(DST)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print(f"✅ 生成 {DST}")
print(f"   关节{m.njnt}(无freejoint,浮动由Isaac处理) 连杆{m.nbody} 总质量{m.body_mass.sum():.1f}kg")
print(f"   注: 此MJCF单根chassis, 转USD后由Isaac ArticulationCfg设为浮动基座")
