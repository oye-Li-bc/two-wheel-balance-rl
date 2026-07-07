"""
验证 RL 环境配置能否正确加载、跑通几步
========================================
用少量环境(16个)跑随机动作, 检查:
  - 环境能否创建(配置/API 名字对不对)
  - 观测维度、动作维度
  - 奖励能否计算
  - reset/step 能否正常运行

运行:
  cd /home/lbc/IsaacLab-2.1.0
  ./isaaclab.sh -p /home/lbc/桌面/机器人作业/scripts/balance_rl/check_env.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import sys, os
sys.path.append(os.path.dirname(__file__))
from two_wheel_env_cfg import TwoWheelBalanceEnvCfg
from isaaclab.envs import ManagerBasedRLEnv


def main():
    cfg = TwoWheelBalanceEnvCfg()
    cfg.scene.num_envs = 16          # 验证用少量环境
    cfg.sim.device = "cuda:0"

    env = ManagerBasedRLEnv(cfg)
    print("\n" + "=" * 55)
    print("✅ 环境创建成功")
    print("观测空间:", env.observation_space)
    print("动作空间:", env.action_space)
    print("=" * 55 + "\n")

    obs, _ = env.reset()
    print("reset 后观测 shape:", obs["policy"].shape)

    for i in range(50):
        action = torch.randn(env.num_envs, env.action_manager.total_action_dim,
                             device=env.device)
        obs, rew, term, trunc, info = env.step(action)
        if i % 10 == 0:
            print(f"step {i:2d}  奖励均值={rew.mean().item():.3f}  "
                  f"终止数={term.sum().item()}  截断数={trunc.sum().item()}")

    print("\n✅ 跑通 50 步, 环境工作正常")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
