"""
两轮平衡车 RL 环境配置 (Manager-Based)
========================================
任务: 只学前后平衡(髋关节锁住, 仅用两轮力矩保持车身直立)

本文件分 5 块:
  1. 机器人配置 ROBOT_CFG
  2. 场景配置   TwoWheelSceneCfg
  3. 动作/观测  ActionsCfg / ObservationsCfg
  4. 奖励/终止/事件 RewardsCfg / TerminationsCfg / EventCfg
  5. 总环境配置 TwoWheelBalanceEnvCfg
"""

import math
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# 仓库根目录 (src/balance_rl/ -> src -> 仓库根)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 底盘 USD(浮动基座、力矩驱动)
USD_PATH = os.path.join(_ROOT, "models", "two_wheel_robot", "two_wheel.usd")

# ---------- 1. 机器人配置 ----------
# 初始 z=0.18m: 让轮子刚好接触地面(轮轴高度≈轮半径), 车身大致直立。
# 髋关节: 高刚度 -> 锁死不动(只学平衡, 不动升降)。
# 轮关节: stiffness=damping=0 -> 纯力矩控制, 由 RL 输出力矩驱动。
ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        # 让刚体接触/求解更稳的物理参数
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),                 # 轮子贴地、车身直立
        joint_pos={".*": 0.0},                # 所有关节零位
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 髋关节: 锁住(高刚度位置伺服, 保持在0)
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            stiffness=200.0,
            damping=10.0,
        ),
        # 轮关节: 力矩控制(无伺服, RL给力矩)
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=20.0,            # 单轮最大力矩 20 N·m
        ),
    },
)


# ---------- 2. 场景配置 ----------
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg


@configclass
class TwoWheelSceneCfg(InteractiveSceneCfg):
    """场景: 地面 + 机器人 + 光。num_envs 个并行环境由训练时指定。"""

    # 地面
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )

    # 机器人(每个并行环境一个)
    robot: ArticulationCfg = ROBOT_CFG

    # 光照
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )


# ---------- 3. 动作 / 观测配置 ----------
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import SceneEntityCfg


@configclass
class ActionsCfg:
    """动作: 只给两个轮关节施加力矩(髋关节不在动作里=被锁住)。"""
    wheel_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*_wheel_joint"],
        scale=20.0,                  # 策略输出[-1,1] × 20 = 力矩(N·m)
    )


@configclass
class ObservationsCfg:
    """观测: 策略能看到的状态。"""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # 重力在机体系的投影 -> 编码车身朝向(直立时≈[0,0,-1], 倒了就变)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        # 车身线速度、角速度(机体系) -> "正在往哪倒/动"
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        # 轮关节速度 -> 轮子转多快
        wheel_vel = ObservationTermCfg(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
        )
        # 上一次动作 -> 帮助策略平滑
        last_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------- 4. 奖励 / 终止 / 事件配置 ----------
from isaaclab.managers import RewardTermCfg, TerminationTermCfg, EventTermCfg
import torch
from isaaclab.assets import RigidObject


def ang_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """惩罚绕竖直轴(z, yaw)的角速度 —— 防止车原地自旋。
    Isaac 2.1.0 只有 ang_vel_xy_l2(管翻倒), 没有 z 版本, 这里自定义补上。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def lin_vel_xy_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """惩罚机体系水平(x前后 + y左右)线速度 —— 让车"原地"站住, 别直立着前溜。
    ★ 这是修"直立前溜10m"问题的关键: 原奖励只惩罚竖直速度 lin_vel_z_l2,
      从没管水平速度, 于是"压着一点前倾角一路匀速滑行"对策略是合法最优解。
      加上这项后, 前溜要扣分, 策略只能选择真正静止平衡。
    Isaac 2.1.0 内置只有 lin_vel_z_l2, 没有 xy 版本, 这里仿 ang_vel_z_l2 自定义补上。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


@configclass
class RewardsCfg:
    """奖励: 直立为主, 惩罚晃动/平移/自旋/大力矩。"""
    # 活着就加分(鼓励别倒)
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)
    # 直立: 惩罚重力投影的水平分量(车身越直, x/y分量越小, 惩罚越小)
    upright = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.0)
    # 惩罚车身翻倒方向角速度(xy) —— 权重调小, 防止初期大角速度爆炸把训练冲垮
    ang_vel = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.01)
    # 惩罚 yaw 自旋(z) —— 修"一直转圈", 权重温和, 别盖过平衡主信号
    yaw_spin = RewardTermCfg(func=ang_vel_z_l2, weight=-0.05)
    # 惩罚竖直方向移动
    lin_vel = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.1)
    # ★惩罚水平前后/左右移动 —— 修"直立前溜"的关键项, 逼策略原地站住。
    #   权重 -0.2: 足够压住"匀速滑行"这个诱人的最优解, 但不能太大——太大(试过-0.5)会
    #   和下面的推力扰动打架: 被推了策略想挪轮子自救, 一动就被重罚, 导致训练在"好↔崩"
    #   之间剧烈震荡学不出稳定行为。-0.2 给恢复动作留了余地。
    lin_vel_xy = RewardTermCfg(func=lin_vel_xy_l2, weight=-0.2)
    # 惩罚力矩(平滑、省力)
    effort = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
    )


@configclass
class TerminationsCfg:
    """终止: 超时 或 倒下(车身过度倾斜)。"""
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # 倒下: 重力在机体z的投影 > -0.5 (直立时≈-1, 倒了趋向0/正)
    fell_over = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},          # 倾斜超 ~40° 判定倒下
    )


@configclass
class EventCfg:
    """事件: 域随机化(Domain Randomization) —— 让策略别只会在 PhysX 里站, 能迁到 MuJoCo。
    核心思想: 训练时把动力学参数随机抖动 + 周期性踹一脚, 逼策略学"对参数不敏感、
    还会主动挪轮子自救"的鲁棒平衡器。这样换到 MuJoCo(又一组不同参数)也扛得住。"""

    # --- startup: 仿真启动时随机化一次(每个并行环境一套不同参数) ---
    # 随机化机器人材质摩擦 —— 覆盖 PhysX↔MuJoCo 的接触摩擦差异
    randomize_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    # 随机缩放所有连杆质量 ±15% —— 覆盖惯量/质量估计误差
    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    # 随机化轮关节 armature(转动惯量) —— 直接对上 MuJoCo 轮子 armature=0.01、Isaac=0 的已知差异
    randomize_wheel_joints = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"]),
            "armature_distribution_params": (0.005, 0.02),
            "friction_distribution_params": (0.0, 0.08),
            "operation": "abs",
        },
    )

    # --- reset: 每回合开始, 给车身随机初始倾斜 + 小初速度 ---
    reset_root = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"roll": (-0.12, 0.12), "pitch": (-0.12, 0.12)},
            "velocity_range": {
                "x": (-0.2, 0.2), "y": (-0.15, 0.15),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # --- interval: 回合进行中每隔几秒踹一脚(直接设置车身速度=模拟外部冲击) ---
    # ★教策略"主动挪轮子接住重心"的关键。但推力别太猛: 试过 ±0.6m/s 每3-7s, 对这么小的
    #   两轮车太暴力→回报方差爆炸→PPO在"好↔崩"间震荡。这里 ±0.3m/s、每5-10s, 温和得多。
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2)}},
    )


# ---------- 5. 总环境配置 ----------
from isaaclab.envs import ManagerBasedRLEnvCfg


@configclass
class TwoWheelBalanceEnvCfg(ManagerBasedRLEnvCfg):
    """把所有部分拼起来 + 仿真参数。"""
    scene: TwoWheelSceneCfg = TwoWheelSceneCfg(num_envs=4096, env_spacing=2.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4                   # 控制频率 = 物理频率/4
        self.episode_length_s = 20.0          # 每回合最长 20 秒
        self.sim.dt = 1.0 / 200.0             # 物理步长 200Hz
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.0, 3.0, 2.0)
