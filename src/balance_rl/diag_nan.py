"""定位NaN来源: 跑一步, 检查obs/reward每项是否有NaN/Inf。"""
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
cfg = TwoWheelCmdEnvCfg(); cfg.scene.num_envs=64; cfg.sim.device="cuda:0"
env = ManagerBasedRLEnv(cfg)
obs,_ = env.reset()
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for i in range(30):
    obs, rew, term, trunc, info = env.step(act)
    o=obs["policy"]
    bad_o = torch.isnan(o).any() or torch.isinf(o).any()
    bad_r = torch.isnan(rew).any() or torch.isinf(rew).any()
    if bad_o or bad_r or i<3 or i==29:
        print(f"step{i}: obs范围[{o.min():.1f},{o.max():.1f}] obsNaN={bad_o.item()} rewNaN={bad_r.item()} rew范围[{rew.min():.1f},{rew.max():.1f}]")
    if bad_o:
        # 哪一维NaN
        for d in range(o.shape[1]):
            if torch.isnan(o[:,d]).any() or torch.isinf(o[:,d]).any():
                print(f"  ★obs维度{d} 有NaN/Inf")
        break
print("各奖励项:")
for k,v in info.get("log",{}).items():
    if "Reward" in k or "reward" in k: print(f"  {k}: {v}")
app.close()
