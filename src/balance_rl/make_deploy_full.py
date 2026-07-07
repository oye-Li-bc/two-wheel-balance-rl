"""
生成完整机器人 MuJoCo 部署模型 deploy_full.xml
========================================
用于部署带速度指令的策略(sim2sim验证 + final_demo阶段A)。
基于 scene_train_full.xml(已有配重+机械臂收拢姿态), 加:
  1. free joint(MuJoCo部署要浮动, 和Isaac训练等价)
  2. 地面 plane + option重力
  3. 执行器: 两轮motor(策略施力矩) + 两髋position锁 + 机械臂position锁收拢姿态
  4. 轮关节armature+damping(数值稳定, 同纯底盘部署经验)

注: scene_train_full.xml 是给Isaac转USD用的(无freejoint), 这里给MuJoCo部署版加回freejoint。
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(_ROOT, "models", "combined", "scene_train_full.xml")
DST = os.path.join(_ROOT, "models", "combined", "deploy_full.xml")

with open(SRC) as f:
    txt = f.read()

# 1) 加 option(重力+小时间步+稳定积分器)
txt = txt.replace(
    '<compiler angle="radian"',
    '<option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>\n  <compiler angle="radian"',
    1,
)

# 2) chassis body 加 freejoint(MuJoCo部署要浮动)
txt = txt.replace(
    '<body name="chassis" pos="0 0 0.18">',
    '<body name="chassis" pos="0 0 0.18">\n      <freejoint name="chassis_free"/>',
    1,
)

# 3) 轮关节加 armature+damping(数值稳定)
for side in ["left", "right"]:
    txt = txt.replace(
        f'<joint name="{side}_wheel_joint" pos="0 0 0" axis="0 0 1"/>',
        f'<joint name="{side}_wheel_joint" pos="0 0 0" axis="0 0 1" armature="0.01" damping="0.05"/>',
    )

# 4) worldbody 内首行加地面
txt = txt.replace(
    "<worldbody>",
    '<worldbody>\n    <geom name="ground" type="plane" size="0 0 0.1" rgba="0.55 0.65 0.55 1"/>\n'
    '    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>',
    1,
)

# 5) 替换 actuator 段: 原scene.xml的机械臂general执行器 → 我们要的部署执行器
#    先删掉原 <actuator>...</actuator>, 再加新的
import re
txt = re.sub(r"<actuator>.*?</actuator>", "", txt, flags=re.DOTALL)
ARM_TUCK = {"arm_joint1":0.0,"arm_joint2":2.0,"arm_joint3":2.5,
            "arm_joint4":0.0,"arm_joint5":0.0,"arm_joint6":0.0}
arm_acts = "\n".join(
    f'    <position name="{j}" joint="{j}" kp="500" kv="50"/>' for j in ARM_TUCK
)
actuators = f"""
  <actuator>
    <motor name="left_wheel"  joint="left_wheel_joint"  ctrlrange="-20 20" gear="1"/>
    <motor name="right_wheel" joint="right_wheel_joint" ctrlrange="-20 20" gear="1"/>
    <position name="left_hip"  joint="left_hip_joint"  kp="200" kv="10"/>
    <position name="right_hip" joint="right_hip_joint" kp="200" kv="10"/>
{arm_acts}
  </actuator>
"""
txt = txt.replace("</mujoco>", actuators + "</mujoco>")

with open(DST, "w") as f:
    f.write(txt)

import mujoco
m = mujoco.MjModel.from_xml_path(DST)
print(f"✅ 生成 {DST}")
print(f"   关节{m.njnt}(含free) 执行器{m.nu}")
print("   执行器:", [m.actuator(i).name for i in range(m.nu)])
