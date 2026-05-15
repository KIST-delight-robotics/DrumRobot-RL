# source/extensions/drum_robot/drum_robot/tasks/__init__.py

# Import task modules to register gym tasks on package import
from . import legacy_task   # noqa: F401
from . import drumrobot
from . import test_sac
