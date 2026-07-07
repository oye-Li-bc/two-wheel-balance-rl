"""
无窗口量化验证: 平衡策略到底站得多稳
========================================
和 deploy_mujoco.py 用完全一样的模型/策略/观测构造, 但:
  - 不开窗口(headless), 能在没显示器/批处理下跑
  - 每次 reset 给车身一点随机初始倾斜(和训练 EventCfg 一致: roll/pitch ±0.1 rad)
  - 跑 N 次试验, 每次最长 EPISODE_S 秒, 统计:
      * 存活时间(倾角超过 FALL_TILT 判定倒下, 提前结束)
      * 平均/最大倾角(度)
      * 水平位置漂移(m) —— 平衡车理想是原地站住, 漂移越小越好
  - 打印逐次结果 + 汇总, 这些数字直接进报告

用法:
  python3 verify_balance.py
  python3 verify_balance.py --trials 20 --episode 20
"""

import argparse
import os
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # src/balance_rl -> src -> 仓库根
MODEL = os.path.join(_ROOT, "models", "two_wheel_robot", "deploy_chassis.xml")
POLICY = os.path.join(_HERE, "policy_actor.npz")

ACTION_SCALE = 20.0     # 和训练 JointEffortActionCfg.scale 一致
# ★控制周期必须和训练一致 = 0.02s(50Hz)。
#   Isaac: sim.dt=1/200=0.005 × decimation4 = 0.02s。
#   MuJoCo: 本模型 timestep=0.002, 所以要 DECIMATION=10 才是 0.02s(0.002×10)。
#   之前误用 DECIMATION=4 → 控制周期0.008s(125Hz), 策略按20ms训的却每8ms重算,
#   闭环动力学错位, 是 sim-to-sim 早倒的主因之一。
DECIMATION = 10
FALL_TILT = 40.0        # 倾角超过此值(度)判定倒下; 和训练终止条件 limit_angle=0.7rad(≈40°)一致


# ---------- 策略网络 (纯 numpy, 与 deploy_mujoco.py 相同) ----------
class Policy:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.w0, self.b0 = d["w0"], d["b0"]
        self.w2, self.b2 = d["w2"], d["b2"]
        self.w4, self.b4 = d["w4"], d["b4"]

    @staticmethod
    def elu(x):
        return np.where(x > 0, x, np.exp(np.clip(x, -30, 0)) - 1.0)

    def __call__(self, obs):
        obs = np.clip(obs, -100.0, 100.0)
        x = self.elu(self.w0 @ obs + self.b0)
        x = self.elu(self.w2 @ x + self.b2)
        return self.w4 @ x + self.b4


def quat_rotate_inverse(q, v):
    """世界系向量 v 旋到机体系 (q=[w,x,y,z])。等价 R(q)^T @ v。"""
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])
    return R.T @ v


def euler_from_quat(q):
    """四元数(wxyz)转 roll/pitch(弧度), 只为算倾角。"""
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    return roll, pitch


def small_tilt_quat(roll, pitch):
    """由小的 roll/pitch 生成初始姿态四元数(wxyz), 模拟训练时的随机扰动。"""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    # yaw=0
    w = cr*cp
    x = sr*cp
    y = cr*sp
    z = -sr*sp
    return np.array([w, x, y, z])


def run_trial(m, d, policy, episode_s, rng):
    """跑一回合, 返回 (存活秒数, 平均倾角度, 最大倾角度, 水平漂移m, 是否摔倒)。"""
    mujoco.mj_resetData(m, d)

    lw_dof = m.joint("left_wheel_joint").dofadr[0]
    rw_dof = m.joint("right_wheel_joint").dofadr[0]
    act_lw = m.actuator("left_wheel").id
    act_rw = m.actuator("right_wheel").id
    act_lh = m.actuator("left_hip").id
    act_rh = m.actuator("right_hip").id

    # 初始随机倾斜(和训练 EventCfg pose_range roll/pitch ±0.1 一致)
    roll0 = rng.uniform(-0.1, 0.1)
    pitch0 = rng.uniform(-0.1, 0.1)
    d.qpos[3:7] = small_tilt_quat(roll0, pitch0)
    mujoco.mj_forward(m, d)

    start_xy = d.qpos[0:2].copy()
    GRAV = np.array([0, 0, -1.0])
    last_action = np.zeros(2)

    dt = m.opt.timestep
    max_steps = int(episode_s / dt)
    tilts = []
    fell = False
    alive_steps = 0

    for step in range(max_steps):
        if step % DECIMATION == 0:
            quat = d.qpos[3:7].copy()
            lin_vel_b = quat_rotate_inverse(quat, d.qvel[0:3].copy())
            ang_vel_b = quat_rotate_inverse(quat, d.qvel[3:6].copy())
            proj_grav = quat_rotate_inverse(quat, GRAV)
            wheel_vel = np.array([d.qvel[lw_dof], d.qvel[rw_dof]])
            obs = np.concatenate([proj_grav, lin_vel_b, ang_vel_b, wheel_vel, last_action])
            action = policy(obs)
            last_action = action.copy()
            tau = np.clip(action, -1, 1) * ACTION_SCALE
            d.ctrl[act_lw] = tau[0]
            d.ctrl[act_rw] = tau[1]
        d.ctrl[act_lh] = 0.0
        d.ctrl[act_rh] = 0.0
        mujoco.mj_step(m, d)

        # 倾角 = 机体z轴与世界竖直的夹角。用重力投影的水平分量算最直接:
        # proj_grav 直立时≈[0,0,-1], 倾角 = arccos(-proj_grav_z)
        pg = quat_rotate_inverse(d.qpos[3:7].copy(), GRAV)
        tilt_deg = np.degrees(np.arccos(np.clip(-pg[2], -1, 1)))
        tilts.append(tilt_deg)
        alive_steps += 1
        if tilt_deg > FALL_TILT:
            fell = True
            break

    alive_s = alive_steps * dt
    drift = float(np.linalg.norm(d.qpos[0:2] - start_xy))
    return alive_s, float(np.mean(tilts)), float(np.max(tilts)), drift, fell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--episode", type=float, default=20.0, help="每回合最长秒数(=训练episode_length_s)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    policy = Policy(POLICY)
    rng = np.random.default_rng(args.seed)

    print(f"模型: {MODEL.split('/')[-1]}  策略: {POLICY.split('/')[-1]}")
    print(f"每回合最长 {args.episode}s, 摔倒阈值 {FALL_TILT}°, 共 {args.trials} 次试验\n")
    print(f"{'#':>3} {'存活(s)':>8} {'平均倾角°':>10} {'最大倾角°':>10} {'漂移(m)':>9}  结果")
    print("-" * 55)

    results = []
    for i in range(args.trials):
        alive, mean_t, max_t, drift, fell = run_trial(m, d, policy, args.episode, rng)
        results.append((alive, mean_t, max_t, drift, fell))
        tag = "摔倒" if fell else "站住✓"
        print(f"{i+1:>3} {alive:>8.2f} {mean_t:>10.2f} {max_t:>10.2f} {drift:>9.3f}  {tag}")

    arr = np.array([[r[0], r[1], r[2], r[3]] for r in results])
    n_survive = sum(1 for r in results if not r[4])
    print("-" * 55)
    print(f"存活率: {n_survive}/{args.trials} = {100*n_survive/args.trials:.0f}%")
    print(f"平均存活时间: {arr[:,0].mean():.2f}s  (满分 {args.episode}s)")
    print(f"全程平均倾角: {arr[:,1].mean():.2f}°   全程最大倾角: {arr[:,2].max():.2f}°")
    print(f"平均水平漂移: {arr[:,3].mean():.3f} m")


if __name__ == "__main__":
    main()
