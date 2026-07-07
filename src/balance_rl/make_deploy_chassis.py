"""
生成 MuJoCo 部署用底盘模型 deploy_chassis.xml
========================================
要求: 和 Isaac 训练模型物理等价(浮动基座/髋锁住/轮力矩/有地面)。

在纯底盘 two_wheel_robot/robot.xml 基础上加:
  1. <option> 重力(默认已有, 显式写)
  2. worldbody 里加地面 plane
  3. 把 base几何+两腿 包进 chassis body + freejoint, 抬高到 z=0.18
  4. 加执行器:
       - 两轮: motor 力矩执行器(策略输出施加处)
       - 两髋: position 执行器, 高刚度锁在 0
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(_ROOT, "models", "two_wheel_robot", "robot.xml")
DST = os.path.join(_ROOT, "models", "two_wheel_robot", "deploy_chassis.xml")

with open(SRC) as f:
    txt = f.read()

# 1) 在 </asset> 后插入 option(重力) —— robot.xml 无 option, 加在 compiler 后
# 用 implicitfast 积分器 + 小时间步, 提高接触数值稳定性, 防止轮子高速接触时爆炸
txt = txt.replace(
    '<compiler angle="radian"/>',
    '<compiler angle="radian"/>\n'
    '  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>',
)

# 2) worldbody 内容包进 chassis + freejoint, 并加地面
#    原 worldbody 第一行是 base 的 geom, 之后是 left_link / right_link
old_world_open = "<worldbody>\n    <geom type=\"mesh\" rgba=\"1 1 1 1\" mesh=\"base_link\"/>"
new_world_open = (
    '<worldbody>\n'
    '    <geom name="ground" type="plane" size="0 0 0.1" rgba="0.55 0.65 0.55 1"/>\n'
    '    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>\n'
    '    <body name="chassis" pos="0 0 0.18">\n'
    '      <freejoint name="chassis_free"/>\n'
    '      <geom type="mesh" rgba="1 1 1 1" mesh="base_link"/>'
)
txt = txt.replace(old_world_open, new_world_open)

# 3) 在 </worldbody> 前补上 chassis body 的闭合标签
txt = txt.replace("</worldbody>", "  </body>\n  </worldbody>")

# 3b) 给轮关节加微小 armature + damping(数值稳定, 防高速接触爆炸)
#     armature 给关节加等效转动惯量, 显著提高积分稳定性
txt = txt.replace(
    '<joint name="left_wheel_joint" pos="0 0 0" axis="0 0 1"/>',
    '<joint name="left_wheel_joint" pos="0 0 0" axis="0 0 1" armature="0.01" damping="0.05"/>',
)
txt = txt.replace(
    '<joint name="right_wheel_joint" pos="0 0 0" axis="0 0 1"/>',
    '<joint name="right_wheel_joint" pos="0 0 0" axis="0 0 1" armature="0.01" damping="0.05"/>',
)

# 4) 在 </mujoco> 前加执行器
actuators = """
  <actuator>
    <motor name="left_wheel"  joint="left_wheel_joint"  ctrlrange="-20 20" gear="1"/>
    <motor name="right_wheel" joint="right_wheel_joint" ctrlrange="-20 20" gear="1"/>
    <position name="left_hip"  joint="left_hip_joint"  kp="200" kv="10"/>
    <position name="right_hip" joint="right_hip_joint" kp="200" kv="10"/>
  </actuator>
"""
txt = txt.replace("</mujoco>", actuators + "</mujoco>")

with open(DST, "w") as f:
    f.write(txt)

# 验证编译
import mujoco
m = mujoco.MjModel.from_xml_path(DST)
print(f"✅ 生成 {DST}")
print(f"   关节 {m.njnt}(含1 free) | 执行器 {m.nu} | 连杆 {m.nbody}")
print("   执行器:", [m.actuator(i).name for i in range(m.nu)])
