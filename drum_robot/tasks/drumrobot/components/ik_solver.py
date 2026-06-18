"""
역기구학
"""

from __future__ import annotations
                # 실제 패키지 누락이 아니라 Isaac Sim이 런타임에서 import path를 추가하는 구조 때문
import torch    # pyright: ignore[reportMissingImports]
import math

from .specs import PartLength

class IKSolver:
    def __init__(self, device: torch.device | str):

        self.device = torch.device(device)

        part_length = PartLength()
        self.upper_arm = torch.tensor(part_length.upper_arm, device=self.device, dtype=torch.float32)
        self.lower_arm = torch.tensor(part_length.lower_arm, device=self.device, dtype=torch.float32)
        self.stick = torch.tensor(part_length.stick, device=self.device, dtype=torch.float32)
        self.waist = torch.tensor(part_length.waist, device=self.device, dtype=torch.float32)

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
            [theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8],
            dim=-1
        )  # (N,9)

        # nan 체크
        nan_bad = torch.isnan(out[:, :9]).any(dim=-1)
        err[nan_bad] = 1.0

        # (우측/좌측 중 하나라도 sqrt_bad면 해당 row를 에러상태로 강제)
        sqrt_bad = sqrt_bad_r | sqrt_bad_l
        if sqrt_bad.any():
            out[sqrt_bad, :] = 0.0
            err[sqrt_bad] = 1.0

        return out, err