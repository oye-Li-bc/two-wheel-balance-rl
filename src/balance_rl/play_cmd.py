"""
回放训练好的平衡策略 —— 亲眼看车站住
========================================
加载训好的 PPO 策略, 开画面, 用策略实时控制车保持平衡。
和训练的区别: 用确定性动作(不加探索噪声), 少量环境, 开渲染。

运行(看画面):
  cd /home/lbc/IsaacLab-2.1.0
  ./isaaclab.sh -p /home/lbc/桌面/机器人作业/scripts/balance_rl/play.py --num_envs 16
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str,
                    default="/home/lbc/桌面/机器人作业/scripts/balance_rl/logs/two_wheel_cmd/model_final.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# 回放要看画面, 强制非 headless
args.headless = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os, sys, torch
sys.path.append(os.path.dirname(__file__))
from two_wheel_cmd_env_cfg import TwoWheelCmdEnvCfg
from ppo_cfg import TwoWheelPPORunnerCfg
from isaaclab.envs import ManagerBasedRLEnv
from nan_guard import NanGuard   # ★复用训练NaN清洗包装(独立模块,无顶层副作用)
from rsl_rl.runners import OnPolicyRunner


def main():
    env_cfg = TwoWheelCmdEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    env = ManagerBasedRLEnv(env_cfg)
    env = NanGuard(env)   # ★和训练一致(清NaN), 否则发散环境的车会瞬移干扰观察

    # 加载策略
    agent_cfg = TwoWheelPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")   # 确定性策略
    print(f"[INFO] 已加载策略: {args.checkpoint}")
    print("[INFO] 看车保持平衡... 关闭窗口结束。")

    obs, _ = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            action = policy(obs)
            obs, _, _, _ = env.step(action)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
