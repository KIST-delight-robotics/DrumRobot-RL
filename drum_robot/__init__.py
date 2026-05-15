# source/extensions/drum_robot/drum_robot/__init__.py

# 폴더 안의 등록 코드를 불러오도록 강제
from .tasks.legacy_task import *
from .tasks.test_sac import *
from .tasks.drumrobot import *

"""
# 가상환경 실행

conda activate env_isaaclab

conda deactivate

"""

"""
# 아이작 심 실행

./isaaclab.sh -s

"""

