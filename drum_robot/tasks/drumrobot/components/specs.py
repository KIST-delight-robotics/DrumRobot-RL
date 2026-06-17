"""
모든 컴포넌트가 공유할 자원
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass
class EnvSpec:
    num_envs: int
    num_drums: int
    episode_length_step: int
    max_lookahead_step: int
    hit_window_step: int
    dt: float