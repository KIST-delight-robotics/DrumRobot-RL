# drum_robot/tasks/legacy_task/ds_cfg.py

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

    """ 기본 환경 설정 """
    decimation = 2  # 정책(Policy) 업데이트 한 번당 시뮬레이션 스텝 수
    episode_length_s = 5.0  # 에피소드 최대 길이 (초)
    action_space = 9       # 로봇 제어 차원
    observation_space = 58  # 관측 차원
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
                solver_velocity_iteration_count=1,  # 0 -> 1 진동 줄이기
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

    """ 태스크 및 로봇 파라미터 """
    # 관절 제한 범위
    joint_limit = {
        "waist_joint":          (-90*math.pi/180,    90*math.pi/180),
        "left_shoulder_1":      ( 30*math.pi/180,   180*math.pi/180),
        "left_shoulder_2":      (-60*math.pi/180,    90*math.pi/180),
        "left_elbow":           (  0*math.pi/180,   140*math.pi/180),
        "right_shoulder_1":     (  0*math.pi/180,   150*math.pi/180),
        "right_shoulder_2":     (-60*math.pi/180,    90*math.pi/180),
        "right_elbow":          (  0*math.pi/180,   140*math.pi/180),
        "left_wrist":           (-10*math.pi/180,    90*math.pi/180),
        "right_wrist":          (-10*math.pi/180,    90*math.pi/180),
    }

    # 로봇 좌표계와 USD 파일의 방향 차이
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

    # 악기의 x, y, z 좌표 (허리 조인트 기준)
    instruments = {
        "snare":  (-0.100,  0.361,  -0.480),
        "floor":  ( 0.232,  0.359,  -0.485),
        "mid":    ( 0.216,  0.597,  -0.378),
        "high":   (-0.069,  0.607,  -0.321),
        "hihat":  (-0.292,  0.493,  -0.224),
        "ride":   ( 0.326,  0.644,  -0.146),
        "crash_r":( 0.485,  0.424,  -0.249),
        "crash_l":(-0.184,  0.669,  -0.147),
    }

    # 양 팔이 타격 가능한 드럼 조합
    drum_set = [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (5, 1),
        (1, 2),
        (3, 2),
        (4, 2),
        (5, 2),
        (8, 2),
        (6, 2),
        (1, 3),
        (4, 3),
        (5, 3),
        (8, 3),
        (1, 4),
        (5, 4),
        (8, 4),
        (1, 5),
        (1, 6),
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
        (8, 6),
        (1, 7),
        (2, 7),
        (3, 7),
        (4, 7),
        (5, 7),
        (8, 7),
        (1, 8),
        (4, 8),
        (5, 8),
    ]

    # 초기 관절각 범위
    init_joint_range = {
        "waist_joint":          (-30*math.pi/180,   30*math.pi/180),
        "left_shoulder_1":      ( 60*math.pi/180,   120*math.pi/180),
        "left_shoulder_2":      (-40*math.pi/180,   50*math.pi/180),
        "left_elbow":           ( 60*math.pi/180,   120*math.pi/180),
        "right_shoulder_1":     ( 60*math.pi/180,   120*math.pi/180),
        "right_shoulder_2":     (-40*math.pi/180,   50*math.pi/180),
        "right_elbow":          ( 60*math.pi/180,   120*math.pi/180),
        "left_wrist":           ( 20*math.pi/180,   50*math.pi/180),
        "right_wrist":          ( 20*math.pi/180,   50*math.pi/180),
    }

    # 관측값 정규화 파라미터
    joint_vel_scale: float = 5.0
    tip_vel_scale: float = 5.0

    # wrist link to tip
    tip_offset_left = (0.385, 0.0, -0.023)   # [m]
    tip_offset_right = (0.385, 0.0, -0.026)  # [m]

    # 타격 상태 파라미터
    idle_xy_radius: float = 0.29

    lift_xy_radius: float = 0.24
    lift_z_min_above_drum: float = 0.2

    lift_fail_xy: float = 0.3
    # lift_fail_vel: float = 0.05

    descend_impact_xy_radius: float = 0.1
    descend_min_impact_velocity: float = 0.5

    descend_fail_xy: float = 0.25
    descend_miss_z_margin: float = 0.05
    # descend_fail_vel: float = 0.05

    return_xy_radius: float = 0.15
    return_z_min_above_drum: float = 0.1

    return_fail_xy: float = 0.25
    # return_fail_vel: float = 0.05
    
    """ 보상 설정 """
    # 하이퍼 파라미터
    exp_k_xy = 80.0
    exp_k_z = 120.0

    lift_z_max = 0.2
    lift_up_vel = 0.05

    descend_down_vel = 0.5

    return_z_max = 0.1
    return_up_vel = 0.3

    limit_margin = 0.08

    # 가중치
    w_idle_xy = 0.01

    w_lift_xy = 0.05
    w_lift_z = 0.02
    w_lift_v = 0.02

    w_desc_xy = 0.1
    w_desc_z = 0.05
    w_desc_v = 0.05

    w_return_xy = 0.1
    w_return_z = 0.2
    w_return_v = 0.1

    w_first_success = 15.00
    w_episode_success = 15.00

    w_success_motion = 0.02
    w_action = 0.002
    w_joint_vel = 0.001
    w_limit = 0.2

    w_first_fail = 5.0

    """ 시각화 설정 """
    enable_tip_markers: bool = False
    enable_drums: bool = False

    color_L: tuple[float, float, float] = (0.0, 0.0, 1.0)
    color_R: tuple[float, float, float] = (1.0, 0.0, 0.0)

    tip_marker_radius: float = 0.015

    drum_radius: float = 0.1
    drum_height: float = 0.01

    


