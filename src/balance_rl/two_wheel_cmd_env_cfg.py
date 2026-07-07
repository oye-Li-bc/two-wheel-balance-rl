"""
两轮平衡车 + 机械臂 RL 环境配置(带速度指令) —— Manager-Based
========================================
和 two_wheel_env_cfg.py(纯底盘原地平衡)的区别:
  1. 机器人 = 完整机器人(底盘+机械臂), 机械臂锁在收拢姿态当配重
  2. 加速度指令(command): 策略学会跟踪给定的前进速度 → 演示时可控移动
  3. 观测 13→16维: 多了 3维速度指令(lin_x, lin_y, ang_z)
  4. 奖励多 track_vel 项: 实际速度接近指令就加分

用模型: combined/scene_train_full.usd(浮动基座+配重降重心)
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

USD_PATH = "/home/lbc/桌面/机器人作业/models/combined/scene_train_full.usd"

# 机械臂收拢姿态(质心最低): joint2=2.0, joint3=2.5, 其余0
ARM_TUCK = {
    "arm_joint1": 0.0, "arm_joint2": 2.0, "arm_joint3": 2.5,
    "arm_joint4": 0.0, "arm_joint5": 0.0, "arm_joint6": 0.0,
}

# ---------- 1. 机器人配置 ----------
ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            fix_root_link=False,        # ★强制浮动基座(平衡必须!否则chassis焊死在空中)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.08),
        joint_pos={**ARM_TUCK, ".*_hip_joint": 0.0, ".*_wheel_joint": 0.0,
                   "arm_gripper_left_finger": -0.004,   # 夹爪半开(范围[-0.008,0]内)
                   "arm_gripper_right_finger": 0.004},  # 范围[0,0.008]内
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 髋关节: 高刚度锁死在0(当刚体)。★退回2维方案: 纯底盘2维平衡已验证成功,
        # 当初上4维(髋也RL控)是为治"移动失败",但那真凶是焊死bug(已修),非RL难。
        # 4维梯度大+物理易发散→NaN污染奖励。锁髋回到验证可行的2维。
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint"], stiffness=200.0, damping=20.0,
            effort_limit_sim=50.0,
        ),
        # 轮关节: 力矩控制(解耦方案: 轮控前进)
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"], stiffness=0.0, damping=0.0,
            effort_limit_sim=20.0,
        ),
        # 机械臂: 超高刚度+足够力矩上限锁在收拢姿态。★诊断发现stiffness=500时臂漂移5.2rad
        # (没设effort_limit→默认力矩上限掐住刚度,臂被重力拉垮甩动→拖乱车身质心→训练发散毒源)。
        # 拉到2000+显式effort_limit=200,把臂真正钉死当刚体配重。
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint.*"], stiffness=2000.0, damping=100.0,
            effort_limit_sim=200.0,
        ),
        # 夹爪: 锁住
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_finger"], stiffness=100.0, damping=10.0,
        ),
    },
)


# ---------- 2. 场景配置 ----------
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
    )
    robot: ArticulationCfg = ROBOT_CFG
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )


# ---------- 3. 速度指令配置 ----------
import isaaclab.envs.mdp as mdp
from isaaclab.managers import CommandTermCfg


@configclass
class CommandsCfg:
    """速度指令: 随机给目标前进速度, 策略学跟踪。
    两轮车只能前进(x) + 转向(yaw), 不能侧移(y), 所以 lin_vel_y 范围=0。"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),     # 每4~6秒换一次目标速度
        rel_standing_envs=0.2,                # 20%环境指令为0(学站定)
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.8),            # 前进速度范围 m/s
            lin_vel_y=(0.0, 0.0),             # 不侧移
            ang_vel_z=(-1.0, 1.0),            # 转向 rad/s
        ),
    )


# ---------- 4. 动作 / 观测配置 ----------
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import SceneEntityCfg


@configclass
class ActionsCfg:
    """动作(2维): 仅两轮力矩。★退回验证可行的2维方案,髋锁死不参与动作。
    纯底盘2维平衡已成功过,这是最可能一次跑通的路。"""
    wheel_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*_wheel_joint"], scale=10.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # clip=(-50,50) 防极端观测值导致训练数值发散(std<0崩溃)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, clip=(-10.0, 10.0))
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, clip=(-10.0, 10.0))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, clip=(-10.0, 10.0))
        velocity_command = ObservationTermCfg(           # ★新增: 速度指令(3维)
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        wheel_vel = ObservationTermCfg(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
            clip=(-10.0, 10.0),
        )
        # ★退回2维: 髋已锁死, 策略无需知道髋状态, 删掉hip_pos/hip_vel两项观测
        last_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------- 5. 奖励 / 终止 / 事件 ----------
from isaaclab.managers import RewardTermCfg, TerminationTermCfg, EventTermCfg
import torch
from isaaclab.assets import RigidObject


def ang_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """惩罚yaw自旋(z) —— 但注意: 有转向指令时不该全惩罚, 这里只做小权重稳定项。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def hip_deviation_l2(env, asset_cfg=SceneEntityCfg("robot", joint_names=[".*_hip_joint"])):
    """惩罚髋关节偏离中位(0) —— 让髋只做小幅平衡调整, 不乱摆到极限。"""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return torch.sum(torch.square(asset.data.joint_pos[:, ids]), dim=1)


def ang_vel_xy_l2_clamped(env, asset_cfg=SceneEntityCfg("robot")):
    """★防爆版 ang_vel_xy_l2: roll/pitch角速度平方, 但每分量先clamp到[-20,20]再平方。
    根治"训练几十轮后此项暴涨1e15→nan"的真凶: 发散环境角速度1e7平方后炸,在总reward
    被夹住前已污染critic。在源头先clamp角速度本身(20rad/s远超正常平衡所需), 平方上限400,
    彻底封死爆炸路径。物理意义不变(仍惩罚翻倒方向自转), 只是给个合理天花板。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    w = torch.clamp(asset.data.root_ang_vel_b[:, :2], -20.0, 20.0)
    return torch.sum(torch.square(w), dim=1)


@configclass
class RewardsCfg:
    # 活着
    alive = RewardTermCfg(func=mdp.is_alive, weight=0.5)
    # 直立(权重调温和: 大权重让回报值域大, critic难拟合→value loss涨→std崩)
    upright = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0)
    # 速度跟踪
    track_vel = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp, weight=1.5,
        params={"std": 0.5, "command_name": "base_velocity"},
    )
    # 转向跟踪
    track_yaw = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp, weight=0.5,
        params={"std": 0.5, "command_name": "base_velocity"},
    )
    # 翻倒方向角速度惩罚: ★用自带clamp的版本(见下方ang_vel_xy_l2_clamped)。
    # 真凶就是它: mdp.ang_vel_xy_l2是角速度平方,发散环境角速度1e7→平方1e15,
    # 在被总reward-clamp夹住前已先污染critic的returns/advantage→梯度爆→nan。
    # 必须在"源头单项"clamp,夹总reward挡不住。权重小(-0.01)对平衡贡献小,但平方最易爆。
    ang_vel = RewardTermCfg(func=ang_vel_xy_l2_clamped, weight=-0.01)
    # 力矩惩罚(仅轮, 髋已锁死不再是动作)
    effort = RewardTermCfg(
        func=mdp.joint_torques_l2, weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fell_over = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": 0.7},
    )


@configclass
class EventCfg:
    reset_root = EventTermCfg(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"roll": (-0.1, 0.1), "pitch": (-0.1, 0.1)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ---------- 6. 总环境配置 ----------
from isaaclab.envs import ManagerBasedRLEnvCfg


@configclass
class TwoWheelCmdEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 4.0, 2.5)
