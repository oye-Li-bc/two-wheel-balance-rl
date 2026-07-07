"""
把 Isaac 训练的平衡策略部署到 MuJoCo (sim-to-sim)
========================================
原理: 策略就是个小网络 13->64->64->2 (ELU)。
  每个控制步: 从 MuJoCo 算出13维观测 -> numpy过网络 -> 2维轮力矩 -> 施加。
  髋关节锁住(和训练一致)。

★ 最关键: 13维观测必须和 Isaac 训练时逐项、同坐标系、同顺序对齐:
  [0:3]  projected_gravity  重力在机体系的投影(单位向量方向)
  [3:6]  base_lin_vel       机体系线速度
  [6:9]  base_ang_vel       机体系角速度
  [9:11] wheel_vel          左右轮关节角速度
  [11:13] last_action       上一步策略输出(缩放前的[-1,1])

用模型: scene_dynamics.xml (带 free joint 的动力学底盘)
注意: Isaac用PhysX, MuJoCo用自己引擎, 存在sim-to-sim gap, 策略未必完美迁移。
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # src/balance_rl -> src -> 仓库根
MODEL = os.path.join(_ROOT, "models", "two_wheel_robot", "deploy_chassis.xml")
POLICY = os.path.join(_HERE, "policy_actor.npz")

ACTION_SCALE = 20.0   # 和训练 JointEffortActionCfg.scale 一致
# 控制周期须=训练的0.02s。本模型 timestep=0.002, 故 DECIMATION=10 (0.002×10=0.02s=50Hz),
# 和 Isaac(dt=1/200×decim4) 对齐。之前误用4→125Hz, 已修。
DECIMATION = 10       # 控制频率 = 物理/10 (和训练一致 50Hz)


# ---------- 策略网络 (纯 numpy) ----------
class Policy:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.w0, self.b0 = d["w0"], d["b0"]
        self.w2, self.b2 = d["w2"], d["b2"]
        self.w4, self.b4 = d["w4"], d["b4"]

    @staticmethod
    def elu(x):
        # 数值稳定: 负半轴 exp 输入裁到 >=-30, 防止 exp 上溢/inf
        return np.where(x > 0, x, np.exp(np.clip(x, -30, 0)) - 1.0)

    def __call__(self, obs):
        obs = np.clip(obs, -100.0, 100.0)   # 观测裁剪, 防异常大值喂入网络
        x = self.elu(self.w0 @ obs + self.b0)
        x = self.elu(self.w2 @ x + self.b2)
        return self.w4 @ x + self.b4   # 输出层无激活


def quat_rotate_inverse(q, v):
    """把世界系向量 v 旋到机体系 (q 为 [w,x,y,z] 机体姿态)。
    等价于 R(q)^T @ v。"""
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])
    return R.T @ v


def main():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    policy = Policy(POLICY)

    # 索引: free joint 的 qpos[0:3]=位置, qpos[3:7]=四元数(wxyz); qvel[0:3]=线速度(世界), qvel[3:6]=角速度(世界)
    # 轮关节 dof 地址(取角速度用)
    lw_dof = m.joint("left_wheel_joint").dofadr[0]
    rw_dof = m.joint("right_wheel_joint").dofadr[0]
    # 执行器索引
    act_lw = m.actuator("left_wheel").id
    act_rw = m.actuator("right_wheel").id
    act_lh = m.actuator("left_hip").id
    act_rh = m.actuator("right_hip").id

    GRAV = np.array([0, 0, -1.0])   # 重力方向(单位向量, 和Isaac projected_gravity一致)
    last_action = np.zeros(2)

    def get_obs():
        quat = d.qpos[3:7].copy()                       # 机体姿态 wxyz
        lin_vel_w = d.qvel[0:3].copy()                  # 世界系线速度
        ang_vel_w = d.qvel[3:6].copy()                  # 世界系角速度
        proj_grav = quat_rotate_inverse(quat, GRAV)     # 重力投影到机体系
        lin_vel_b = quat_rotate_inverse(quat, lin_vel_w)
        ang_vel_b = quat_rotate_inverse(quat, ang_vel_w)
        wheel_vel = np.array([d.qvel[lw_dof], d.qvel[rw_dof]])
        return np.concatenate([proj_grav, lin_vel_b, ang_vel_b, wheel_vel, last_action])

    mujoco.mj_resetData(m, d)
    # 髋关节锁在0(position执行器目标=0)
    d.ctrl[act_lh] = 0.0
    d.ctrl[act_rh] = 0.0

    print("部署 Isaac 平衡策略到 MuJoCo... 关闭窗口结束。")
    print("观测13维已按训练顺序对齐(重力投影/线速/角速/轮速/上次动作, 均机体系)。")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        step = 0
        while viewer.is_running():
            if step % DECIMATION == 0:                  # 控制频率=物理/4(同训练)
                obs = get_obs()
                action = policy(obs)                    # 网络输出[-1,1]附近
                last_action = action.copy()
                tau = np.clip(action, -1, 1) * ACTION_SCALE
                d.ctrl[act_lw] = tau[0]
                d.ctrl[act_rw] = tau[1]
                d.ctrl[act_lh] = 0.0                    # 持续锁髋
                d.ctrl[act_rh] = 0.0
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)
            step += 1


if __name__ == "__main__":
    main()
