from __future__ import annotations

import numpy as np
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# 너가 성공시킨 경로 그대로
USD_PATH = "/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/drum_robot/assets/drum_robot/usd/drum_robot.usd"
PRIM_PATH = "/World/DrumRobot"

# sim
SIM_DT = 1.0 / 120.0
DEVICE = "cuda"

# implicit PD
KP = 250.0
KD = 10.0


@configclass
class DrumRobotSceneCfg(InteractiveSceneCfg):
    """Scene 구성 (로봇, 땅, 조명)"""

    # 로봇
    robot: ArticulationCfg = ArticulationCfg(
        prim_path=PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "pos": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=KP,
                damping=KD,
            )
        },
    )


@configclass
class DrumRobotEnvCfg(DirectRLEnvCfg):
    """DirectRLEnv 필수 cfg"""

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=SIM_DT, device=DEVICE)

    # Scene
    scene: DrumRobotSceneCfg = DrumRobotSceneCfg(num_envs=1, env_spacing=2.0)

    # RL basic
    decimation: int = 1  # policy step마다 physics step 몇 번? 지금은 1:1
    episode_length_s: float = 10.0

    # validate 통과용 필수: gym spaces
    # StepA: action 1개(waist position target), obs 2개(q, qd)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    # StepA control target joint
    waist_joint_name: str = "waist_joint"

    # action scaling (rad)
    # action in [-1, 1] -> target in [-max_rad, +max_rad]
    max_waist_rad: float = 0.8

    # reward target
    target_waist_rad: float = 0.0
