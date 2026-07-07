"""
带速度指令策略 → MuJoCo 部署 (完整机器人, sim2sim)
========================================
观测16维(比纯底盘多3维速度指令):
  [0:3]  projected_gravity   (机体系)
  [3:6]  base_lin_vel        (机体系)
  [6:9]  base_ang_vel        (机体系)
  [9:12] velocity_command    ★速度指令 [vx, vy, yaw]
  [12:14] wheel_vel
  [14:16] last_action

机械臂用position执行器锁在收拢姿态; 髋锁0; 轮由策略给力矩。
可改 CMD 测试不同速度指令。
"""

import numpy as np
import mujoco
import mujoco.viewer
import time

MODEL = "/home/lbc/桌面/机器人作业/models/combined/deploy_full.xml"
POLICY = "/home/lbc/桌面/机器人作业/scripts/balance_rl/policy_cmd_actor.npz"

ACTION_SCALE = 20.0
DECIMATION = 4
ARM_TUCK = {"arm_joint1": 0.0, "arm_joint2": 2.0, "arm_joint3": 2.5,
            "arm_joint4": 0.0, "arm_joint5": 0.0, "arm_joint6": 0.0}

# ★速度指令 [前进vx(m/s), 侧移vy, 转向yaw(rad/s)] —— 改这里测试
CMD = np.array([0.4, 0.0, 0.0])    # 往前 0.4 m/s


class Policy:
    def __init__(self, p):
        d = np.load(p)
        self.W = [d["w0"], d["w2"], d["w4"]]
        self.b = [d["b0"], d["b2"], d["b4"]]

    @staticmethod
    def elu(x):
        return np.where(x > 0, x, np.exp(np.clip(x, -30, 0)) - 1.0)

    def __call__(self, obs):
        x = np.clip(obs, -50, 50)
        x = self.elu(self.W[0] @ x + self.b[0])
        x = self.elu(self.W[1] @ x + self.b[1])
        return self.W[2] @ x + self.b[2]


def quat_rot_inv(q, v):
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])
    return R.T @ v


def main():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    policy = Policy(POLICY)

    lw = m.joint("left_wheel_joint").dofadr[0]
    rw = m.joint("right_wheel_joint").dofadr[0]
    act = {m.actuator(i).name: i for i in range(m.nu)}
    last_action = np.zeros(2)
    GRAV = np.array([0, 0, -1.0])

    def get_obs():
        q = d.qpos[3:7]
        pg = quat_rot_inv(q, GRAV)
        lb = quat_rot_inv(q, d.qvel[0:3])
        ab = quat_rot_inv(q, d.qvel[3:6])
        wv = np.array([d.qvel[lw], d.qvel[rw]])
        return np.concatenate([pg, lb, ab, CMD, wv, last_action])

    mujoco.mj_resetData(m, d)
    # ★初始高度 z=0.03(轮子触地的正确高度, 不是0.18悬空) —— sim2sim关键修复
    d.qpos[2] = 0.03
    d.qpos[3:7] = [1, 0, 0, 0]
    # 机械臂position目标=收拢姿态, 髋=0
    for j, v in ARM_TUCK.items():
        d.ctrl[act[j]] = v
    d.ctrl[act["left_hip"]] = 0.0
    d.ctrl[act["right_hip"]] = 0.0
    mujoco.mj_forward(m, d)

    print(f"部署带速度指令策略到MuJoCo, 指令={CMD}")
    print("关闭窗口结束。")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        step = 0
        while viewer.is_running():
            if step % DECIMATION == 0:
                a = policy(get_obs())
                last_action = a.copy()
                tau = np.clip(a, -1, 1) * ACTION_SCALE
                d.ctrl[act["left_wheel"]] = tau[0]
                d.ctrl[act["right_wheel"]] = tau[1]
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)
            step += 1


if __name__ == "__main__":
    main()
