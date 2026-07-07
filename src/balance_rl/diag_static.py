"""
静态诊断: 只生成不训练, 一次性查清三件事
========================================
1. 机械臂关节是否真锁死(打印刚度+零动作下臂关节位置漂移多少)
2. 车放下后是否站得住(零动作下root高度/倾角随时间变化)
3. 哪个量先发散成NaN/大数(逐步检查root速度/角速度)
回答用户回放观察: "瘫软在地+机械臂甩着+瞬移消失"到底是哪个环节。
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app

import torch, os, sys
sys.path.append(os.path.dirname(__file__))
from two_wheel_cmd_env_cfg import TwoWheelCmdEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

cfg = TwoWheelCmdEnvCfg(); cfg.scene.num_envs = 4; cfg.sim.device = "cuda:0"
env = ManagerBasedRLEnv(cfg)
robot = env.scene["robot"]

# ---- 1. 关节配置: 刚度/阻尼 ----
names = robot.joint_names
stiff = robot.data.default_joint_stiffness[0]
damp = robot.data.default_joint_damping[0]
print("\n===== 关节刚度/阻尼(查机械臂是否锁死) =====")
for i, nm in enumerate(names):
    print(f"  {nm:28s} stiffness={stiff[i]:8.1f} damping={damp[i]:7.1f}")

# 记录臂关节初始位置
arm_ids = [i for i, nm in enumerate(names) if nm.startswith("arm_joint")]
q0 = robot.data.joint_pos[0, arm_ids].clone()

# ---- 2&3. 零动作下放置, 看站立/发散 ----
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
print("\n===== 零动作放置(查站立稳定性+发散) =====")
print(f"动作维度={env.action_manager.total_action_dim} (应为2=仅两轮)")
print(f"初始 root_z={robot.data.root_pos_w[0,2].item():.3f}")
for i in range(150):
    env.step(act)
    if i % 25 == 0 or i == 149:
        z = robot.data.root_pos_w[0, 2].item()
        # 倾角: projected_gravity z分量, -1=完全直立
        pg = robot.data.projected_gravity_b[0]
        rv = robot.data.root_lin_vel_w[0]
        av = robot.data.root_ang_vel_w[0]
        armq = robot.data.joint_pos[0, arm_ids]
        arm_drift = (armq - q0).abs().max().item()
        nan = torch.isnan(robot.data.root_pos_w).any().item()
        print(f"step{i:3d} z={z:6.3f} grav_z={pg[2]:6.3f} "
              f"|v|={rv.norm():7.2f} |w|={av.norm():7.2f} "
              f"臂漂移={arm_drift:.4f} NaN={nan}")

print("\n判读:")
print("  机械臂锁死? → 臂漂移应≈0(高刚度锁住)。若漂移大=没使能")
print("  车站得住? → grav_z应接近-1且z稳定。若grav_z往0跑+z降=在倒")
print("  发散? → |v|/|w|应有限。若爆到1e3+或NaN=PhysX发散")
app.close()
