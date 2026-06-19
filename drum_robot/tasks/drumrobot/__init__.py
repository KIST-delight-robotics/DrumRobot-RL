from __future__ import annotations
                        # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import gymnasium as gym # pyright: ignore[reportMissingImports]

from . import agents

gym.register(
    id="DrumRobot-drum_robot-Direct-v0",     # id : 실행할 때 사용할 이름
    entry_point=f"{__name__}.drumrobot_env:DrumRobotEnv",  # 환경 클래스 위치
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drumrobot_cfg:DrumRobotEnvCfg",    # 설정 클래스 위치
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",    # RL 알고리즘(skrl) 설정 경로를 알려줍니다.
    },
) # 테스트