"""
PPO 训练超参数配置 (RSL-RL)
========================================
PPO = 近端策略优化, 目前最常用的连续控制 RL 算法。
两轮平衡是个简单任务, 用小网络 + 适中训练量即可。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TwoWheelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24            # 每个环境每轮收集24步经验
    max_iterations = 300              # 训练300轮(平衡简单, 通常几十轮就站住)
    save_interval = 50                # 每50轮存一次模型
    experiment_name = "two_wheel_balance"
    # ★防"第一次update梯度爆炸成NaN→log_std=NaN→std>=0崩溃"的真正杠杆:
    #   关掉时obs_normalizer=Identity,原始观测直接喂网络,4维动作+浮动基座下
    #   首轮暴烈状态让critic回归目标巨大→梯度inf→权重NaN。开启后观测在线归一化,
    #   是Isaac Lab几乎所有locomotion任务的默认。之前一直是False,这是关键差异。
    empirical_normalization = True

    # 策略网络(Actor-Critic): 输入观测 -> 输出动作
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,           # 初始探索噪声
        # ★根治"std>=0崩溃": scalar参数化下std是裸参数,PPO一步梯度就能把它推成负数
        #   → torch.normal报错。log参数化存log_std,用时exp(),std数学上恒>0,不可能崩。
        #   4维动作(轮+髋)梯度比2维大,才暴露此问题。这是标准做法(SB3/原版PPO同款)。
        noise_std_type="log",
        actor_hidden_dims=[64, 64],   # Actor 两层64(小任务够用)
        critic_hidden_dims=[64, 64],  # Critic 两层64
        activation="elu",
    )

    # PPO 算法超参
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,               # PPO裁剪范围
        entropy_coef=0.005,           # 鼓励探索
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,                   # 折扣因子
        lam=0.95,                     # GAE lambda
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
