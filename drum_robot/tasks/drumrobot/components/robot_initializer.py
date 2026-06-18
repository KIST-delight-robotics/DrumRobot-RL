"""
로봇 초기 위치 초기화 클래스 
"""

from __future__ import annotations

from dataclasses import dataclass, field
                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch    # pyright: ignore[reportMissingImports]
import math

from .specs import RobotSpec, Instruments
from .ik_solver import IKSolver

@dataclass
class RobotInitializerCfg:
    # 제어하는 관절 개수
    num_ctrl_joint: int = 9

    # 양 팔이 위치 가능한 드럼 조합 [L, R]
    drum_pairs: list = field(default_factory=lambda: [
        (1, 1), (5, 1),
        (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (8, 2),
        (1, 3), (3, 3), (4, 3), (5, 3), (8, 3),
        (1, 4), (4, 4), (5, 4), (8, 4),
        (1, 5), (5, 5),
        (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (8, 6),
        (1, 7), (2, 7), (3, 7), (4, 7), (6, 7), (7, 7),
        (1, 8), (4, 8), (5, 8), (8, 8),
    ])

    height_above_drum: float = 0.1

    joint_noise_scale: float = 5*math.pi/180

class RobotInitializer:
    def __init__(
            self,
            device: torch.device | str,
            cfg: RobotInitializerCfg,
            ctrl_joint_names: list,
    ):
        self.device = torch.device(device)
        self.cfg = cfg
        self.ctrl_joint_names = ctrl_joint_names
        self.instruments = Instruments()
        self.robot = RobotSpec()

        self._init_pos_angle()

    def _init_pos_angle(self):
        N = len(self.cfg.drum_pairs)
        ik_solver = IKSolver(device=self.device)

        drum_pos = torch.tensor(
            [inst.position for inst in self.instruments.items.values()],
            device=self.device,
            dtype=torch.float32
        )   # (8, 3)

        drum_pairs = torch.tensor(
            self.cfg.drum_pairs,
            device=self.device,
            dtype=torch.int32
        )   # (N, 2)
        drum_pairs_idx = drum_pairs - 1

        p = drum_pos[drum_pairs_idx, :]  # (N, 2, 3)
        pl = p[:, 0, :]     # (N, 3)
        pr = p[:, 1, :]

        pl[:, 2] = pl[:, 2] + self.cfg.height_above_drum
        pr[:, 2] = pr[:, 2] + self.cfg.height_above_drum

        pm_xy = (pl[:, 0:2] + pr[:, 0:2]) / 2
        theta0 = torch.atan2(pm_xy[:, 1], pm_xy[:, 0]) - 90*math.pi/180

        theta7 = torch.full((N,), 25*math.pi/180, device=self.device)
        theta8 = torch.full((N,), 25*math.pi/180, device=self.device)

        out, err = ik_solver.solve_geometric_ik(pr, pl, theta0, theta7, theta8)    # (N, 9)

        self.pos_angle = torch.zeros((N, self.cfg.num_ctrl_joint), device=self.device)

        """ 반드시 self.ctrl_joint_names 순서대로 텐서를 만들어여 함 """
        for i in range(self.cfg.num_ctrl_joint):
            name = self.ctrl_joint_names[i]
            idx = self.robot.joint_name_to_ik_result_idx[name]
            self.pos_angle[:, i] = out[:, idx]
    
    def reset_init_pos(self, env_ids):
        N = len(self.cfg.drum_pairs)

        rand = torch.randint(0, N, (len(env_ids),), device=self.device)

        init_pos = self.pos_angle[rand, :]
        init_pos = init_pos + self.cfg.joint_noise_scale * torch.randn_like(init_pos)

        return init_pos