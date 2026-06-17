"""
로봇 초기 위치 초기화 클래스 
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
import math

@dataclass
class PartLength:
    upper_arm: float = 0.2303
    lower_arm: float = 0.200
    stick: float = 0.325 + 0.048
    waist: float = 0.520

class IKSolver:
    def __init__(self, part: PartLength, device: torch.device | str):

        self.device = torch.device(device)

        self.upper_arm = torch.tensor(part.upper_arm, device=self.device, dtype=torch.float32)
        self.lower_arm = torch.tensor(part.lower_arm, device=self.device, dtype=torch.float32)
        self.stick = torch.tensor(part.stick, device=self.device, dtype=torch.float32)
        self.waist = torch.tensor(part.waist, device=self.device, dtype=torch.float32)

    def _get_length(self, theta: torch.Tensor) -> torch.Tensor:

        x = self.lower_arm + self.stick * torch.cos(theta)
        y = self.stick * torch.sin(theta)

        return torch.sqrt(x * x + y * y)

    def _get_theta(self, theta: torch.Tensor) -> torch.Tensor:

        x = self.lower_arm + self.stick * torch.cos(theta)
        y = self.stick * torch.sin(theta)

        return torch.atan2(y, x)

    @torch.no_grad()
    def solve_geometric_ik(
        self,
        pR: torch.Tensor,       # (N, 3)
        pL: torch.Tensor,       # (N, 3)
        theta0: torch.Tensor,   # (N, 1)
        theta7: torch.Tensor,   # (N, 1)
        theta8: torch.Tensor,   # (N, 1)
    ) -> torch.Tensor:

        err = torch.zeros_like(theta0)

        XR, YR, ZR = pR[:, 0], pR[:, 1], pR[:, 2]
        XL, YL, ZL = pL[:, 0], pL[:, 1], pL[:, 2]

        # constants
        L1 = self.upper_arm
        S  = self.waist

        L2_R = self._get_length(theta7)
        L2_L = self._get_length(theta8)

        # shoulder positions
        shoulderXR = 0.5 * S * torch.cos(theta0)
        shoulderYR = 0.5 * S * torch.sin(theta0)
        shoulderXL = -0.5 * S * torch.cos(theta0)
        shoulderYL = -0.5 * S * torch.sin(theta0)

        # ---- q1 ----
        theta01 = torch.atan2(YR - shoulderYR, XR - shoulderXR)
        theta1 = theta01 - theta0

        # theta1 range: 0 ~ 150deg
        bad1 = (theta1 < 0.0) | (theta1 > 150.0 * math.pi / 180.0)
        err = torch.where(bad1, torch.ones_like(err), err)

        # ---- q2 ----
        theta02 = torch.atan2(YL - shoulderYL, XL - shoulderXL)
        theta2 = theta02 - theta0

        # theta2 range: 30deg ~ 180deg
        bad2 = (theta2 < 30.0 * math.pi / 180.0) | (theta2 > math.pi)
        err = torch.where(bad2, torch.ones_like(err), err)

        # =========================
        # Right arm geometry
        # =========================
        zeta_r = - 1 * ZR
        r2_r = (YR - shoulderYR) ** 2 + (XR - shoulderXR) ** 2

        x_r = zeta_r * zeta_r + r2_r - L1 * L1 - L2_R * L2_R
        rad_r = 4.0 * L1 * L1 * L2_R * L2_R - x_r * x_r

        # C++: rad<0 이면 즉시 return (theta0=99, err=1)
        sqrt_bad_r = rad_r < 0.0
        err = torch.where(sqrt_bad_r, torch.ones_like(err), err)

        # 안전 sqrt
        y_r = torch.sqrt(torch.clamp(rad_r, min=0.0))

        theta4 = torch.atan2(y_r, x_r)
        theta34 = torch.atan2(torch.sqrt(torch.clamp(r2_r, min=0.0)), zeta_r)
        theta3 = theta34 - torch.atan2(L2_R * torch.sin(theta4), L1 + L2_R * torch.cos(theta4))

        # theta3 range: -45 ~ 90deg
        bad3 = (theta3 < -45.0 * math.pi / 180.0) | (theta3 > 90.0 * math.pi / 180.0)
        err = torch.where(bad3, torch.ones_like(err), err)

        # =========================
        # Left arm geometry
        # =========================
        zeta_l = -1 * ZL
        r2_l = (YL - shoulderYL) ** 2 + (XL - shoulderXL) ** 2

        x_l = zeta_l * zeta_l + r2_l - L1 * L1 - L2_L * L2_L
        rad_l = 4.0 * L1 * L1 * L2_L * L2_L - x_l * x_l

        sqrt_bad_l = rad_l < 0.0
        err = torch.where(sqrt_bad_l, torch.ones_like(err), err)

        y_l = torch.sqrt(torch.clamp(rad_l, min=0.0))

        theta6 = torch.atan2(y_l, x_l)
        theta56 = torch.atan2(torch.sqrt(torch.clamp(r2_l, min=0.0)), zeta_l)
        theta5 = theta56 - torch.atan2(L2_L * torch.sin(theta6), L1 + L2_L * torch.cos(theta6))

        # theta5 range: -45 ~ 90deg
        bad5 = (theta5 < -45.0 * math.pi / 180.0) | (theta5 > 90.0 * math.pi / 180.0)
        err = torch.where(bad5, torch.ones_like(err), err)

        # adjust theta4/theta6 by stick geometry
        theta4 = theta4 - self._get_theta(theta7)
        theta6 = theta6 - self._get_theta(theta8)

        # theta4 range: 0 ~ 140deg
        bad4 = (theta4 < 0.0) | (theta4 > 140.0 * math.pi / 180.0)
        err = torch.where(bad4, torch.ones_like(err), err)

        # theta6 range: 0 ~ 140deg
        bad6 = (theta6 < 0.0) | (theta6 > 140.0 * math.pi / 180.0)
        err = torch.where(bad6, torch.ones_like(err), err)

        out = torch.stack(
            [theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, err],
            dim=-1
        )  # (N,10)

        # nan 체크
        nan_bad = torch.isnan(out[:, :9]).any(dim=-1)
        out[nan_bad, 9] = 1.0

        # C++의 "sqrt 음수면 즉시 return theta0=99" 동작을 배치로 반영
        # (우측/좌측 중 하나라도 sqrt_bad면 해당 row를 에러상태로 강제)
        sqrt_bad = sqrt_bad_r | sqrt_bad_l
        if sqrt_bad.any():
            out[sqrt_bad, :] = 0.0
            out[sqrt_bad, 0] = 99.0
            out[sqrt_bad, 9] = 1.0

        return out

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

    inst_name_to_idx: dict = field(default_factory=lambda: {
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

    joint_noise_scale: float = 5*math.pi/180

class RobotInitializer:
    def __init__(
            self,
            device: torch.device | str,
            cfg: RobotInitializerCfg,
            ctrl_joint_names: list,
            instruments: dict,
    ):
        self.device = torch.device(device)
        self.cfg = cfg
        self.ctrl_joint_names = ctrl_joint_names
        self.instruments = instruments

        self._init_pos_angle()

    def _init_pos_angle(self):
        N = len(self.cfg.drum_pairs)
        ik_solver = IKSolver(part=PartLength(), device=self.device)

        inst_pos = torch.tensor(
            list(self.instruments.values()),
            device=self.device,
            dtype=torch.float32
        )   # (8, 3)

        drum_pairs = torch.tensor(
            self.cfg.drum_pairs,
            device=self.device,
            dtype=torch.int32
        )   # (N, 2)
        drum_pairs_idx = drum_pairs - 1

        p = inst_pos[drum_pairs_idx, :]  # (N, 2, 3)
        pl = p[:, 0, :]     # (N, 3)
        pr = p[:, 1, :]

        pl[:, 2] = pl[:, 2] + self.cfg.height_above_drum
        pr[:, 2] = pr[:, 2] + self.cfg.height_above_drum

        pm_xy = (pl[:, 0:2] + pr[:, 0:2]) / 2
        the0 = torch.atan2(pm_xy[:, 1], pm_xy[:, 0]) - 90*math.pi/180

        the7 = torch.full((N,), 25*math.pi/180, device=self.device)
        the8 = torch.full((N,), 25*math.pi/180, device=self.device)

        out = ik_solver.solve_geometric_ik(pr, pl, the0, the7, the8)    # (N, 10)

        self.pos_angle = torch.zeros((N, self.cfg.num_ctrl_joint), device=self.device)

        """ 반드시 self.ctrl_joint_names 순서대로 텐서를 만들어여 함 """
        for i in range(self.cfg.num_ctrl_joint):
            name = self.ctrl_joint_names[i]
            idx = self.cfg.inst_name_to_idx[name]
            self.pos_angle[:, i] = out[:, idx]
    
    def reset_init_pos(self, env_ids):
        N = len(self.cfg.drum_pairs)

        rand = torch.randint(0, N, (len(env_ids),), device=self.device)

        init_pos = self.pos_angle[rand, :]
        init_pos = init_pos + self.cfg.joint_noise_scale * torch.randn_like(init_pos)

        return init_pos