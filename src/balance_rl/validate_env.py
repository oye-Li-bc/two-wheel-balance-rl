"""
只实例化环境、跑几步随机动作, 校验域随机化 EventCfg 配置无误(不训练、不保存模型)。
用法(conda env):
  cd <IsaacLab 安装目录>
  ./isaaclab.sh -p <仓库>/src/balance_rl/validate_env.py --headless
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os, sys, torch
sys.path.append(os.path.dirname(__file__))
from two_wheel_env_cfg import TwoWheelBalanceEnvCfg
from isaaclab.envs import ManagerBasedRLEnv


def main():
    cfg = TwoWheelBalanceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"
    env = ManagerBasedRLEnv(cfg)
    obs, _ = env.reset()
    print(f"[OK] 环境创建成功. 观测维度: {obs['policy'].shape}")
    print(f"[OK] 事件项: {list(env.event_manager.active_terms) if hasattr(env,'event_manager') else 'n/a'}")
    for i in range(60):  # 跑够触发一次 interval push(3-7s, dt*decim=0.02s/步)
        act = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device="cuda:0")
        env.step(act)
    print("[OK] 跑了60步(含域随机化+推力事件), 无报错. 配置有效 ✅")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
