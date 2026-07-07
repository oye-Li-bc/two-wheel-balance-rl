"""在Isaac里读策略控制下车站立稳定后的真实状态, 作为MuJoCo部署的对照基准。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless=True
app = AppLauncher(args).app

import torch, os, sys
sys.path.append(os.path.dirname(__file__))
from two_wheel_cmd_env_cfg import TwoWheelCmdEnvCfg
from ppo_cfg import TwoWheelPPORunnerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

cfg = TwoWheelCmdEnvCfg(); cfg.scene.num_envs=4; cfg.sim.device="cuda:0"
env = ManagerBasedRLEnv(cfg); env = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(env, TwoWheelPPORunnerCfg().to_dict(), log_dir=None, device="cuda:0")
runner.load("/home/lbc/桌面/机器人作业/scripts/balance_rl/logs/two_wheel_cmd/model_final.pt")
policy = runner.get_inference_policy(device="cuda:0")

obs,_ = env.get_observations()
for i in range(300):  # 跑稳
    with torch.inference_mode():
        obs,_,_,_ = env.step(policy(obs))

robot = env.unwrapped.scene["robot"]
print("\n===== Isaac站立稳定后(env0) =====")
print("root高度z =", robot.data.root_pos_w[0,2].item())
print("root四元数 =", robot.data.root_quat_w[0].cpu().numpy().round(3))
print("projected_gravity =", robot.data.projected_gravity_b[0].cpu().numpy().round(3))
print("关节名顺序 =", robot.data.joint_names)
print("观测向量(16) =", obs[0].cpu().numpy().round(3))
app.close()
