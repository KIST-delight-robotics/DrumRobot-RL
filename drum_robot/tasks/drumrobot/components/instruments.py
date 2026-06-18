from __future__ import annotations

from dataclasses import dataclass, field
import torch

@dataclass
class Instrument:
    id: int
    position: tuple[float, float, float]    # 악기의 x, y, z 좌표 (허리 조인트 기준)

@dataclass
class Instruments:
    items: dict[str, Instrument] = field(default_factory=lambda: {
        "snare":   Instrument(1, (-0.100,  0.361,  -0.480)),
        "floor":   Instrument(2, ( 0.232,  0.359,  -0.485)),
        "mid":     Instrument(3, ( 0.216,  0.597,  -0.378)),
        "high":    Instrument(4, (-0.069,  0.607,  -0.321)),
        "hihat":   Instrument(5, (-0.292,  0.493,  -0.224)),
        "ride":    Instrument(6, ( 0.326,  0.644,  -0.146)),
        "crash_r": Instrument(7, ( 0.485,  0.424,  -0.249)),
        "crash_l": Instrument(8, (-0.184,  0.669,  -0.147)),
    })

    inst_noise_scale: float = 0.02
