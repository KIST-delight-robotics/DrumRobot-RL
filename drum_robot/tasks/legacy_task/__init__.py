# drum_robot/tasks/legacy_task/__init__.py

from __future__ import annotations

import gymnasium as gym

from . import agents

# Point Reaching
# 양 팔의 팁 위치를 목표 위치로 수렴하는 동작 학습

gym.register(
    id="DrumRobot-point_reaching-Direct-v0",     # id : 실행할 때 사용할 이름
    entry_point=f"{__name__}.pr_env:DrumRobotEnv",  # 환경 클래스 위치
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pr_cfg:DrumRobotEnvCfg",    # 설정 클래스 위치
        "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_pr_cfg:get_default_ppo_cfg",    # RL 알고리즘(skrl) 설정 경로를 알려줍니다.
    },
)

"""
../../../isaaclab.sh \
  -p drum_robot/scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-point_reaching-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --checkpoint="/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/logs/skrl/drum_robot/ppo_pr/checkpoints/agent_800000.pt"
"""

# Drum Striking
# 양 팔이 목표 드럼을 타격하는 동작 학습

gym.register(
    id="DrumRobot-drum_reaching-Direct-v0",     # id : 실행할 때 사용할 이름
    entry_point=f"{__name__}.dr_env:DrumRobotEnv",  # 환경 클래스 위치
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dr_cfg:DrumRobotEnvCfg",    # 설정 클래스 위치
        "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_dr_cfg:get_default_ppo_cfg",    # RL 알고리즘(skrl) 설정 경로를 알려줍니다.
    },
)

"""
../../../isaaclab.sh \
  -p drum_robot/scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-drum_reaching-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --checkpoint="/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/logs/skrl/drum_robot/ppo_dr/checkpoints/agent_700000.pt"
"""

# Drum Striking
# 양 팔이 목표 드럼을 타격하는 동작 학습

gym.register(
    id="DrumRobot-drum_striking-Direct-v0",     # id : 실행할 때 사용할 이름
    entry_point=f"{__name__}.ds_env:DrumRobotEnv",  # 환경 클래스 위치
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ds_cfg:DrumRobotEnvCfg",    # 설정 클래스 위치
        "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_ds_cfg:get_default_ppo_cfg",    # RL 알고리즘(skrl) 설정 경로를 알려줍니다.
    },
)

"""
../../../isaaclab.sh \
  -p drum_robot/scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-drum_striking-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --checkpoint="/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/logs/skrl/drum_robot/ppo_ds/checkpoints/agent_100000.pt"
"""

# Rhythmic Drum Striking
# RDS 기반 악보 정보를 관측하여 정해진 타이밍에 목표 드럼을 타격하는 동작 학습

gym.register(
    id="DrumRobot-rhythmic_drum_reaching-Direct-v0",     # id : 실행할 때 사용할 이름
    entry_point=f"{__name__}.rdr_env:DrumRobotEnv",  # 환경 클래스 위치
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rdr_cfg:DrumRobotEnvCfg",    # 설정 클래스 위치
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_rdr_cfg.yaml",    # RL 알고리즘(skrl) 설정 경로를 알려줍니다.
    },
)

"""
../../../isaaclab.sh \
  -p drum_robot/scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-rhythmic_drum_reaching-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --checkpoint="/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/logs/skrl/drum_robot/ppo_rdr/checkpoints/agent_1000000.pt"
"""

# Rhythmic Drum Striking
# 랜덤으로 RDS를 생성해서 정해진 타이밍에 목표 드럼을 타격하는 동작 학습
# 기존 MIDI를 사용하여 100K 학습된 체크포인트에서 랜덤 RDS로 학습함 (처음부터 랜덤 RDS를 사용하면 수렴하지 못함)

gym.register(
    id="DrumRobot-random_rhythmic_drum_reaching-Direct-v0",     # id : 실행할 때 사용할 이름
    entry_point=f"{__name__}.rrdr_env:DrumRobotEnv",  # 환경 클래스 위치
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rrdr_cfg:DrumRobotEnvCfg",    # 설정 클래스 위치
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_rrdr_cfg.yaml",    # RL 알고리즘(skrl) 설정 경로를 알려줍니다.
    },
)

"""
../../../isaaclab.sh \
  -p drum_robot/scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-random_rhythmic_drum_reaching-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --checkpoint="/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/logs/skrl/drum_robot/05_100K/checkpoints/agent_100000.pt"
"""

# Rhythmic Drum Striking
# 랜덤으로 RDS를 생성해서 정해진 타이밍에 목표 드럼을 타격하는 동작 학습
# 처음부터 랜덤 RDS로 학습함

gym.register(
    id="DrumRobot-rhythmic_drum_striking-Direct-v0",     # id : 실행할 때 사용할 이름
    entry_point=f"{__name__}.rds_env:DrumRobotEnv",  # 환경 클래스 위치
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rds_cfg:DrumRobotEnvCfg",    # 설정 클래스 위치
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_rds_cfg.yaml",    # RL 알고리즘(skrl) 설정 경로를 알려줍니다.
    },
)

"""
../../../isaaclab.sh \
  -p drum_robot/scripts/reinforcement_learning/skrl/play.py \
  --task=DrumRobot-rhythmic_drum_striking-Direct-v0 \
  --num_envs=1 \
  --checkpoint="/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/logs/skrl/drum_robot/06_1M/checkpoints/agent_1000000.pt"

../../../isaaclab.sh \
  -p drum_robot/scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-rhythmic_drum_striking-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --checkpoint="/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/logs/skrl/drum_robot/ppo_rds/checkpoints/agent_5000000.pt"
"""