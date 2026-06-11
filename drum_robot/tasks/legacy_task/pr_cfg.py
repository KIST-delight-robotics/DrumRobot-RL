# drum_robot/tasks/legacy_task/pr_cfg.py

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

import math # pi

USD_PATH = "/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/drum_robot/assets/drum_robot/usd/drum_robot.usd"
PRIM_PATH = "/World/envs/env_.*/Robot"   # env_.*를 써야 수백 개의 환경에 복제됩니다.

# sim
SIM_DT = 1.0 / 120.0
DEVICE = "cuda"

@configclass
class DrumRobotEnvCfg(DirectRLEnvCfg):

    decimation = 2  # 정책(Policy) 업데이트 한 번당 시뮬레이션 스텝 수
    episode_length_s = 5.0  # 에피소드 최대 길이 (초)
    action_space = 9       # 로봇 제어 차원
    observation_space = 30  # 관측 차원
    state_space = 0
    action_scale = 0.03

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT, 
        device=DEVICE, 
        render_interval=decimation
    )

    # robot (s)
    robot_waist_joint_to_link_z: float = 0.0755
    robot_waist_joint_offset_z: float = 1.0

    robot_cfg: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,  # 0 -> 1 진동 줄이기?
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, robot_waist_joint_offset_z+robot_waist_joint_to_link_z), 
            joint_pos={"(.*)": 0.0}, 
        ),
        prim_path=PRIM_PATH,
        actuators={
            "drum_joints": ImplicitActuatorCfg(
                joint_names_expr=["waist_joint",
            "right_shoulder_1",
            "left_shoulder_1",
            "right_shoulder_2",
            "right_elbow",
            "left_shoulder_2",
            "left_elbow",
            "right_wrist",
            "left_wrist",], # 필요시 모든 관절을 제어 대상으로 설정 joint_names_expr=[".*"]
                stiffness=200.0,           # 벨로시티 제어 시 일반적으로 0
                damping=10.0,           # 댐핑값은 로봇 무게에 맞춰 조절
            ),
        },
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)

    joint_range = {
        "waist_joint":          (-30*math.pi/180, 30*math.pi/180),
        "left_shoulder_1":      (60*math.pi/180, 120*math.pi/180),
        "left_shoulder_2":      (-40*math.pi/180, 50*math.pi/180),
        "left_elbow":           (60*math.pi/180, 120*math.pi/180),
        "right_shoulder_1":     (60*math.pi/180, 120*math.pi/180),
        "right_shoulder_2":     (-40*math.pi/180, 50*math.pi/180),
        "right_elbow":          (60*math.pi/180, 120*math.pi/180),
        "left_wrist":           (0*math.pi/180, 60*math.pi/180),
        "right_wrist":          (0*math.pi/180, 60*math.pi/180),
    }

    joint_limit = {
        "waist_joint":          (-90*math.pi/180, 90*math.pi/180),
        "left_shoulder_1":      (30*math.pi/180, 180*math.pi/180),
        "left_shoulder_2":      (-60*math.pi/180, 90*math.pi/180),
        "left_elbow":           (0*math.pi/180, 140*math.pi/180),
        "right_shoulder_1":     (0*math.pi/180, 150*math.pi/180),
        "right_shoulder_2":     (-60*math.pi/180, 90*math.pi/180),
        "right_elbow":          (0*math.pi/180, 140*math.pi/180),
        "left_wrist":           (-20*math.pi/180, 90*math.pi/180),
        "right_wrist":          (-20*math.pi/180, 90*math.pi/180),
    }

    joint_usd_dir = {
        "waist_joint":          +1,
        "left_shoulder_1":      -1,
        "left_shoulder_2":      +1,
        "left_elbow":           +1,
        "right_shoulder_1":     -1,
        "right_shoulder_2":     -1,
        "right_elbow":          -1,
        "left_wrist":           -1,
        "right_wrist":          +1,
    }

    # --- Reward weights ---
    rew_w_dist: float = 0.3          # distance shaping
    rew_w_exp: float = 3.0           # exp shaping
    rew_w_action: float = 0.002      # action L2 penalty
    rew_w_joint_vel: float = 0.0005  # joint velocity penalty
    rew_w_limit: float = 0.2         # joint limit barrier penalty
    rew_w_died: float = 1.0          # 죽으면 -5

    # --- Reward shaping params ---
    exp_k: float = 4.0              # exp(-k * dist)
    limit_margin: float = 10.0 * math.pi / 180.0   # [rad] limit 근처 마진

    # 성공 보너스
    success_dist_thr: float = 0.1    # [m] 각 손이 타겟에 이 거리 이내면 성공 후보
    success_hold_steps: int = 10     # 연속 유지 스텝 수 (policy dt=1/60 (decimation*SIM_DT))
    success_bonus: float = 2.0       # 성공 시 보너스

    # 랜덤 타겟 min/max (Lxyz, Rxyz)
    target_min: tuple[float, float, float, float, float, float] = (-0.20, 0.35, -0.35, 0.15, 0.35, -0.35) # 기준은 허리 조인트 기준
    target_max: tuple[float, float, float, float, float, float] = (-0.15, 0.40, -0.30, 0.20, 0.40, -0.30)

    target_center = (0.0, 0.4, -0.3, 0.0, 0.4, -0.3)    # 정규화 (L, R)
    target_half_range = (0.5, 0.6, 0.3, 0.5, 0.6, 0.3)

    tip_offset_left = (0.385, 0.0, -0.023)   # [m] in left_wrist frame
    tip_offset_right = (0.385, 0.0, -0.026)  # [m] in right_wrist frame

    # 마커 시각화
    enable_target_markers: bool = False
    enable_tip_markers: bool = False

    target_marker_radius: float = 0.03
    target_marker_color: tuple[float, float, float] = (0.0, 0.0, 1.0)

    tip_marker_radius: float = 0.015
    tip_marker_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
