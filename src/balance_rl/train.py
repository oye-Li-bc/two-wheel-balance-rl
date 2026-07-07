"""
两轮平衡 PPO 训练入口
========================================
做的事:
  1. 启动 Isaac
  2. 把我们的环境包成 RSL-RL 能用的接口
  3. 跑 PPO 训练, 自动保存模型到 logs/

运行(训练, 无渲染更快):
  cd <IsaacLab 安装目录>
  ./isaaclab.sh -p <仓库>/src/balance_rl/train.py --headless

运行(带画面看训练, 较慢):
  ./isaaclab.sh -p .../train.py --num_envs 64
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096, help="并行环境数")
parser.add_argument("--max_iterations", type=int, default=300, help="训练轮数")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os, sys
import torch
import gymnasium as gym
sys.path.append(os.path.dirname(__file__))

from two_wheel_env_cfg import TwoWheelBalanceEnvCfg
from ppo_cfg import TwoWheelPPORunnerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    # 环境配置
    env_cfg = TwoWheelBalanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    # PPO 配置
    agent_cfg = TwoWheelPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations

    # 日志目录 (与本脚本同级的 logs/)
    log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs",
        agent_cfg.experiment_name,
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] 日志/模型保存到: {log_dir}")

    # 创建环境 + 包装成 RSL-RL 接口
    env = ManagerBasedRLEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    # PPO Runner 并训练
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    print(f"[INFO] 开始训练: {args.num_envs} 环境 × {args.max_iterations} 轮")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)

    # 保存最终模型
    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"[INFO] ✅ 训练完成, 模型已保存")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
