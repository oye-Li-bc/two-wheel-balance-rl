"""
NaN 清洗环境包装 —— 训练/回放共用
========================================
★这是根治"std>=0崩溃"的真正修复(不是调参, 是堵毒源)。

现象: 训练 iter0 健康跑过, iter1 的 update 报 RuntimeError: std>=0。
真因链: 浮动基座的车一放就倒(fell_over~343/存活0.16s) → 翻倒的 free body
  在 PhysX 里数值发散, root 速度/四元数蹦出 NaN/Inf → 喂进网络 → NaN 梯度
  → log_std 变 NaN → exp(NaN)=NaN 过不了 torch.normal 的 std>=0 检查。
关键认知: 观测项的 clip=(-10,10) 挡不住 NaN —— torch.clamp(NaN) 仍是 NaN,
  所以之前调重心/lr/clip/奖励权重全没用, 毒源根本没被堵。
修复: 在 obs/reward 进入 RSL-RL 之前用 nan_to_num 清洗, 把毒源彻底堵死。

配合另外两层(都在 ppo_cfg.py):
  - noise_std_type="log": std=exp(log_std) 恒正, 不会被梯度推成负数
  - empirical_normalization=True: 观测在线归一化, 防 critic/梯度爆炸
"""

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class NanGuard(RslRlVecEnvWrapper):
    @staticmethod
    def _clean_obs(t):
        # 先洗nan/inf, 再硬clamp幅度。★关键修正: nan_to_num只挡nan/±inf,
        # 放行"有限但巨大"的值(如发散环境的1e17), 必须再加clamp才真正堵死毒源。
        t = torch.nan_to_num(t, nan=0.0, posinf=100.0, neginf=-100.0)
        return torch.clamp(t, -100.0, 100.0)

    @staticmethod
    def _clean_rew(t):
        # 奖励夹到[-50,50]: 防发散环境的ang_vel平方(1e16级)毒化critic目标→value_loss爆炸。
        t = torch.nan_to_num(t, nan=0.0, posinf=50.0, neginf=-50.0)
        return torch.clamp(t, -50.0, 50.0)

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        return self._clean_obs(obs), self._clean_rew(rew), dones, extras

    def reset(self):
        obs, extras = super().reset()
        return self._clean_obs(obs), extras

    def get_observations(self):
        obs, extras = super().get_observations()
        return self._clean_obs(obs), extras
