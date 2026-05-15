# source/extensions/drum_robot/drum_robot/tasks/test_sac/drumrobot_cfg.py

from __future__ import annotations

import gymnasium as gym

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
    # action_space = 9       # 로봇 제어 차원
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(9,))   # 행동 공간이 제한되지 않았다면, 랜덤 탐험을 수행하기 때문에 샘플링된 랜덤 행동이 제한되지 않고 값이 거의 무한대에 가까워진다고 함
    observation_space = 288  # 관측 차원
    state_space = 0
    action_scale = 0.05

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

    # state 들어가는 악보 길이
    rds_observation_length: int = 30

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

    # 타격 판정
    drum_xy_radius: float = 0.15
    drum_z_margin: float = 0.10
    min_impact_velocity: float = 0.5
    rearm_height: float = 0.15
    hit_window_step: int = 5

    """ 보상 설정 """
    # 하이퍼 파라미터
    limit_margin = 0.08
    alpha = 1.0

    # 팁이 가면 안되는 범위
    x_limit = 0.5
    y_limit_l = 0.2
    y_limit_h = 0.8
    z_limit = -0.6

    # 팁의 대기 위치
    idle_tip_pos = (-0.100, 0.361, -0.380)  # 스네어 위치에서 z +10cm

    # 가중치
    w_success = 0.5
    w_wrong = 0.1
    w_miss = 0.1
    w_time_error = 0.1

    w_progress = 10.0
    w_proximity = 2.0

    w_action = 0.0005
    w_joint_vel = 0.0003
    w_limit = 0.2
    w_tip_limit = 2.0

    """ 시각화 설정 """
    enable_visualization: bool = False

    color_L: tuple[float, float, float] = (0.0, 0.0, 1.0)
    color_R: tuple[float, float, float] = (1.0, 0.0, 0.0)

    tip_marker_radius: float = 0.015

    drum_radius: float = 0.1
    drum_height: float = 0.01

    


