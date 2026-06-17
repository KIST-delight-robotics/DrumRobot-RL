"""
모든 컴포넌트가 공유할 자원
"""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class EnvSpec:
    num_envs: int
    num_drums: int
    episode_length_step: int
    max_lookahead_step: int
    hit_window_step: int
    dt: float

@dataclass
class PartLength:
    upper_arm: float = 0.2303
    lower_arm: float = 0.200
    stick: float = 0.325 + 0.048
    waist: float = 0.520

@dataclass
class RobotSpec:
    part_length: PartLength = field(default_factory=PartLength)