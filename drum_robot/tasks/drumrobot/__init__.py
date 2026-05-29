
"""
path: drum_robot/tasks/drumrobot/__init__.py
"""

from __future__ import annotations

import gymnasium as gym

from . import agents

gym.register(
    id="DrumRobot-drum_score-Direct-v0",     # id : 실행할 때 사용할 이름
    entry_point=f"{__name__}.drumrobot_env:DrumRobotEnv",  # 환경 클래스 위치
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drumrobot_cfg:DrumRobotEnvCfg",    # 설정 클래스 위치
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",    # RL 알고리즘(skrl) 설정 경로를 알려줍니다.
    },
)

"""
체크포인트 불러와서 학습 실행 명령어

../../../isaaclab.sh \
  -p drum_robot/scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-drum_score-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --checkpoint="/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/logs/skrl/drum_robot/2026-05-22_09-11-39_ppo_torch_drum_score/checkpoints/agent_5000000.pt"


테스트 명령어

../../../isaaclab.sh \
  -p drum_robot/scripts/reinforcement_learning/skrl/play.py \
  --task=DrumRobot-drum_score-Direct-v0 \
  --num_envs=1 \
  --checkpoint="/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/logs/skrl/drum_robot/2026-05-15_18-11-06_ppo_torch_drum_score/checkpoints/agent_1000000.pt"
"""