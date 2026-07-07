"""测chassis是否真浮动: 纯重力下看z是否下降。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless=True
app = AppLauncher(args).app
import torch, os, sys
sys.path.append(os.path.dirname(__file__))
from two_wheel_cmd_env_cfg import TwoWheelCmdEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
cfg = TwoWheelCmdEnvCfg(); cfg.scene.num_envs=4; cfg.sim.device="cuda:0"
env = ManagerBasedRLEnv(cfg)
robot = env.scene["robot"]
print("初始 root_z =", robot.data.root_pos_w[0,2].item())
# 纯重力, 给零动作
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for i in range(200):
    env.step(act)
    if i%50==0:
        print(f"step{i} root_z={robot.data.root_pos_w[0,2].item():.3f}")
print("末 root_z =", robot.data.root_pos_w[0,2].item())
print("→ 若z持续下降=浮动正常; 若z不变=仍焊死")
app.close()
