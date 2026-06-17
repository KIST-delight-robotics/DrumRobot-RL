"""
타격 감지 클래스
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .specs import EnvSpec

@dataclass
class HitDetectorCfg:
    alpha: float = 0.9  # 속도 계산 필터

    # 타격 판정
    drum_xy_radius: float = 0.13
    drum_z_range: float = 0.07
    min_impact_velocity: float = 0.2
    rearm_height: float = 0.18
    hit_window_step: int = 10

class HitDetector:
    def __init__(
            self,
            device: torch.device | str,
            cfg: HitDetectorCfg,
            env: EnvSpec,
    ):
        self.device = device
        self.cfg = cfg
        self.env = env

        self._alloc_buffers()
        self.env_arange = torch.arange(self.env.num_envs, device=self.device)

    def detect(
            self,
            tip_pos,
            inst_pos,
            hit_armed,
            steps,
            rds,
            rds_visit,
    ):    
        # 팁 속도
        alpha = self.cfg.alpha
        prev_tip_pos = self.tip_pos
        prev_tip_vel = self.tip_vel
        tip_vel = self._compute_tip_velocity(tip_pos, prev_tip_pos, prev_tip_vel, alpha)

        # 팁 드럼 거리 계산
        dist_xy_sq, diff_z = self._compute_tip_drum_dist_sq(tip_pos, inst_pos)

        # 접촉 확인
        contact_mask = self._check_contact_drum(dist_xy_sq, diff_z)

        # 타격 판정
        hit_per_arm = self._check_hitting(
            contact_mask=contact_mask,
            hit_armed=hit_armed,
            tip_vel=tip_vel,
            diff_z=diff_z,
        )   # (N, 2, M)
        
        # 양팔 중 하나라도 해당 drum을 strike하면 hit
        hit_mask = torch.any(hit_per_arm, dim=1)   # (N, M)

        # 잘못친 타격 판정
        wrong_hit = self._detect_wrong_hits(hit_mask, rds, steps)

        # 윈도우 끝났을 때 타격 성공 확인
        success, missed_target, time_error = self._finalize_target_outcomes(
            rds=rds,
            rds_visit=rds_visit,
            steps=steps,
        )

        # re-arm 확인
        next_hit_armed = self._check_rearm(hit_armed, hit_per_arm, contact_mask, diff_z)

        # 
        self.tip_pos = tip_pos
        self.prev_tip_pos = prev_tip_pos

        return (
            hit_mask,
            tip_pos, tip_vel, prev_tip_pos, next_hit_armed,
            success, wrong_hit, missed_target, time_error,
            hit_per_arm,
        )
    
    def reset(self, env_ids, tip_pos):
        # 팁 위치/속도 리셋
        self.tip_pos[env_ids] = tip_pos[env_ids]
        self.prev_tip_pos[env_ids] = tip_pos[env_ids]
        self.tip_vel[env_ids] = 0.0
    
    def _alloc_buffers(self):
        N = self.env.num_envs

        # 로봇의 tip 위치를 저장할 텐서
        self.tip_pos = torch.zeros((N, 2, 3), device=self.device)
        self.prev_tip_pos = torch.zeros((N, 2, 3), device=self.device)

        # 로봇의 tip 속도 저장할 텐서
        self.tip_vel = torch.zeros((N, 2, 3), device=self.device)
    
    def _compute_tip_velocity(self, tip_pos, prev_tip_pos, prev_tip_vel, alpha):
        tip_vel = (tip_pos - prev_tip_pos) / self.env.dt

        tip_vel_f = (1 - alpha) * tip_vel + alpha * prev_tip_vel

        return tip_vel_f
    
    def _compute_tip_drum_dist_sq(self, tip_pos, inst_pos):
        # tip_pos: (N, 2, 3), inst_pos: (N, M, 3)
        # N: num env, M: num drum

        diff_xy = tip_pos[:, :, None, 0:2] - inst_pos[:, None, :, 0:2] # (N, 2, M, 2)

        dist_xy_sq = torch.sum(diff_xy * diff_xy, dim=-1)    # (N, 2, M)

        diff_z = tip_pos[:, :, None, 2] - inst_pos[:, None, :, 2] # (N, 2, M)

        return dist_xy_sq, diff_z
    
    def _check_contact_drum(self, dist_xy_sq, diff_z):
        # N: num env, M: num drum
        # xy 범위 확인
        radius_sq = self.cfg.drum_xy_radius ** 2
        in_xy_range = dist_xy_sq <= radius_sq   # (N, 2, M)

        # z 높이 확인
        in_z_range = (diff_z <= self.cfg.drum_z_range) & (diff_z >= 0.0)    # (N, 2, M)

        contact_mask = in_xy_range & in_z_range

        return contact_mask

    def _check_hitting(self, contact_mask, hit_armed, tip_vel, diff_z):
        # contact_mask: (N, 2, M)
        # hit_armed:    (N, 2, M)
        # tip_vel:      (N, 2, 3)
        # diff_z:       (N, 2, M)

        # 아래 방향 속도 조건
        tip_vel_z = tip_vel[:, :, 2].unsqueeze(-1)   # (N, 2, 1)
        is_downward = tip_vel_z < (-self.cfg.min_impact_velocity)   # (N, 2, 1), broadcast 가능

        # strike candidate
        hit_per_arm = contact_mask & hit_armed & is_downward   

        return hit_per_arm

    def _detect_wrong_hits(self, hit_mask, rds, steps):
        # hit_mask: (N, M)
        # rds: (N, T, M)
        # steps: (N,)

        N = self.env.num_envs
        T = self.env.episode_length_step
        M = self.env.num_drums
        W = self.cfg.hit_window_step

        window_target_union = torch.zeros((N, M), device=self.device, dtype=torch.bool)

        for offset in range(-W, W + 1):
            cand_steps = steps + offset
            valid = (cand_steps >= 0) & (cand_steps < T)
            cand_steps_clamped = cand_steps.clamp(0, T - 1)

            target_mask = rds[self.env_arange, cand_steps_clamped, :] > 0.5     # (N, M)
            target_mask &= valid.unsqueeze(-1)

            window_target_union |= target_mask

        wrong_hit = hit_mask & (~window_target_union)

        return wrong_hit

    def _finalize_target_outcomes(self, rds, rds_visit, steps):
        # rds, rds_visit: (N, T, M)
        # steps: (N,)

        N = self.env.num_envs
        T = self.env.episode_length_step
        M = self.env.num_drums
        W = self.cfg.hit_window_step

        success = torch.zeros((N, M), device=self.device, dtype=torch.bool)
        time_error = torch.full((N, M), -1, device=self.device, dtype=torch.int64)

        # 윈도우 왼쪽 끝 스텝
        window_end_step = steps - W
        valid = (window_end_step >= 0) & (window_end_step < T)
        window_end_step = window_end_step.clamp(0, T - 1)

        target_mask = (rds[self.env_arange, window_end_step, :] > 0.5)
        target_mask &= valid.unsqueeze(-1)     # (N, M)

        offsets = self._get_hit_window_offsets(W)
        for offset in offsets:
            cand_steps = window_end_step + offset
            valid = (cand_steps >= 0) & (cand_steps < T)
            cand_steps_clamped = cand_steps.clamp(0, T - 1)

            hit_mask = rds_visit[self.env_arange, cand_steps_clamped, :] > 0.5  # (N, M)

            match_mask = hit_mask & valid.unsqueeze(-1) & target_mask & (~success)
            success |= match_mask

            time_error[match_mask] = abs(offset)  # step 차이
        
        missed_target = target_mask & (~success)
        
        return success, missed_target, time_error

    def _get_hit_window_offsets(self, W):
        offsets = [0]
        for i in range(1, W + 1):
            offsets.append(-i)
            offsets.append(i)

        return offsets

    def _check_rearm(self, hit_armed, hit_per_arm, contact_mask, diff_z):
        """
        준비

        Args:
            hit_armed:  (N, 2, M)   # 이전 준비 상태
            contact_mask: (N, 2, M) # 접촉 여부
            diff_z: (N, 2, M)       # z 거리

        Returns:
            next_hit_armed: (N, 2, M)
        """
        next_hit_armed = hit_armed.clone()

        # 충분히 벗어나고 올라가면 rearm
        rearm_mask = diff_z > self.cfg.rearm_height
        next_hit_armed[rearm_mask] = True

        # 접촉 중이면 disarm
        contact_expanded = contact_mask.any(dim=2, keepdim=True)  # (N,2,1)
        next_hit_armed = next_hit_armed.masked_fill(contact_expanded, False)

        return next_hit_armed