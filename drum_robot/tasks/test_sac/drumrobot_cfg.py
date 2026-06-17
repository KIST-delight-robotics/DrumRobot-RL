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
    observation_space = 94  # 관측 차원
    state_space = 0
    action_scale = math.pi

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
    inst_noise_scale: float = 0.02

    # 관측값 정규화 파라미터
    joint_vel_scale: float = 5.0

    # wrist link to tip
    tip_offset_left = (0.385, 0.0, -0.023)   # [m]
    tip_offset_right = (0.385, 0.0, -0.026)  # [m]

    # 타격 관측
    max_lookahead_time: float = 1.0    # 최대 관측 범위
    num_hits: int = 3      # 최대 관측 타격 개수

    # 타격 판정
    drum_xy_radius = 0.13
    drum_z_range = 0.07
    min_impact_velocity = 0.2
    rearm_height = 0.18
    hit_window_step = 10

    """ 시각화 설정 """
    enable_visualization: bool = False
