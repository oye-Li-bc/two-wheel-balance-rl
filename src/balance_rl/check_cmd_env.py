"""
验证带速度指令的新环境能否加载、跑通
========================================
同时验证: 第1步的完整USD能在Isaac加载, 第2步的环境配置正确。
检查: 观测维度(应16维=13+3指令)、动作2维、command/reward各项、跑50步不崩。

运行:
  cd <IsaacLab 安装目录>
  ./isaaclab.sh -p <仓库>/src/balance_rl/check_cmd_env.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch, sys, os
sys.path.append(os.path.dirname(__file__))
from two_wheel_cmd_env_cfg import TwoWheelCmdEnvCfg
from isaaclab.envs import ManagerBasedRLEnv


def main():
    cfg = TwoWheelCmdEnvCfg()
    cfg.scene.num_envs = 16
    cfg.sim.device = "cuda:0"
    env = ManagerBasedRLEnv(cfg)

    print("\n" + "=" * 55)
    print("✅ 带指令环境创建成功")
    print("观测空间:", env.observation_space)
    print("动作空间:", env.action_space)
    print("=" * 55 + "\n")

    obs, _ = env.reset()
    print("reset 后观测 shape:", obs["policy"].shape, "(期望最后一维=16)")

    for i in range(50):
        action = torch.randn(env.num_envs, env.action_manager.total_action_dim,
                             device=env.device)
        obs, rew, term, trunc, info = env.step(action)
        if i % 10 == 0:
            cmd = env.command_manager.get_command("base_velocity")[0]
            print(f"step {i:2d} 奖励={rew.mean().item():.3f} 终止={term.sum().item()} "
                  f"指令样例={cmd.cpu().numpy().round(2)}")

    print("\n✅ 跑通50步, 带速度指令环境正常")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
