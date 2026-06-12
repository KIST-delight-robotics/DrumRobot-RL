from __future__ import annotations

import gymnasium as gym

from . import agents

gym.register(
    id="DrumRobot-drum_robot-Direct-v0",     # id : 실행할 때 사용할 이름
    entry_point=f"{__name__}.drumrobot_env:DrumRobotEnv",  # 환경 클래스 위치
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drumrobot_cfg:DrumRobotEnvCfg",    # 설정 클래스 위치
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",    # RL 알고리즘(skrl) 설정 경로를 알려줍니다.
    },
)