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

    # 관절 이름과 ik 푼 결과의 인덱스 매칭
    joint_name_to_ik_result_idx: dict = field(default_factory=lambda: {
        "waist_joint":          0,
        "left_shoulder_1":      2,
        "left_shoulder_2":      5,
        "left_elbow":           6,
        "right_shoulder_1":     1,
        "right_shoulder_2":     3,
        "right_elbow":          4,
        "left_wrist":           8,
        "right_wrist":          7,
    })

@dataclass
class DrumSpec:
    # 악기의 x, y, z 좌표 (허리 조인트 기준)
    position: dict[str, tuple[float, float, float]] = field(default_factory=lambda: {
        "snare":  (-0.100,  0.361,  -0.480),
        "floor":  ( 0.232,  0.359,  -0.485),
        "mid":    ( 0.216,  0.597,  -0.378),
        "high":   (-0.069,  0.607,  -0.321),
        "hihat":  (-0.292,  0.493,  -0.224),
        "ride":   ( 0.326,  0.644,  -0.146),
        "crash_r":( 0.485,  0.424,  -0.249),
        "crash_l":(-0.184,  0.669,  -0.147),
    })