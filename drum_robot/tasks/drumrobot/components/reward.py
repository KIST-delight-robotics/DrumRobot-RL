"""
보상 클래스
"""

from __future__ import annotations

import torch
from dataclasses import dataclass

# === 모듈 레벨 jit 함수들 ===
@torch.jit.script
def assignment_reward_for_targets(
        target_mask: torch.Tensor,  # (N, M)
        inst_pos: torch.Tensor,     # (N, M, 3)
        left_tip_pos: torch.Tensor, # (N, 3)
        right_tip_pos: torch.Tensor,# (N, 3)
        tip_vel: torch.Tensor,      # (N, 2, 3)
        hit_armed: torch.Tensor,    # (N, 2, M)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = target_mask.device
    N, M = target_mask.shape

    # target index 추출 (최대 2개)
    masked_idx = torch.where(
        target_mask,
        torch.arange(M, device=device).expand(N, M),
        -1,
    )

    values, indices = torch.topk(masked_idx, k=2, dim=1, largest=True)  # (N, 2)   # 입력 받은 tensor의 상위 k개의 index와 value를 반환함.

    idx0 = values[:, 0].unsqueeze(1)    # (N, 1)
    idx0 = torch.where(idx0 >= 0, idx0, torch.zeros_like(idx0)) # 타겟이 없는 경우 인덱싱 오류 방지

    idx1 = values[:, 1].unsqueeze(1)
    idx1 = torch.where(idx1 >= 0, idx1, torch.zeros_like(idx1)) # 타겟이 2개가 아닌 경우 인덱싱 오류 방지

    # --------------------

    # XYZ 거리
    diff_l = left_tip_pos[:, None, :] - inst_pos    # (N, M, 3)
    diff_r = right_tip_pos[:, None, :] - inst_pos

    dist_l = torch.sum(diff_l * diff_l, dim=-1)     # (N, M)
    dist_r = torch.sum(diff_r * diff_r, dim=-1)

    d_l0 = dist_l.gather(1, idx0).squeeze(1)   # (N,)  # gather(input, dim, index): 지정한 dim 축에서 index 위치의 값을 가져오기
    d_r0 = dist_r.gather(1, idx0).squeeze(1)

    d_l1 = dist_l.gather(1, idx1).squeeze(1)
    d_r1 = dist_r.gather(1, idx1).squeeze(1)

    # --------------------

    down_vel_l = torch.clamp(-tip_vel[:, 0, 2], 0.0, 2.0)   # (N,)
    down_vel_r = torch.clamp(-tip_vel[:, 1, 2], 0.0, 2.0)

    up_vel_l = torch.clamp( tip_vel[:, 0, 2], 0.0, 2.0)     # (N,)
    up_vel_r = torch.clamp( tip_vel[:, 1, 2], 0.0, 2.0)

    hit_armed_l = hit_armed[:, 0, :]    # (N, M)
    hit_armed_r = hit_armed[:, 1, :]

    h_l0 = hit_armed_l.gather(1, idx0).squeeze(1)      # (N,)
    h_r0 = hit_armed_r.gather(1, idx0).squeeze(1)

    h_l1 = hit_armed_l.gather(1, idx1).squeeze(1)
    h_r1 = hit_armed_r.gather(1, idx1).squeeze(1)

    # --------------------

    target_count = target_mask.float().sum(dim=1)
    cost = torch.zeros(N, device=device)
    upward_reward = torch.zeros(N, device=device)
    downward_reward = torch.zeros(N, device=device)

    # --------------------
    # case 1: 타겟 1개
    # --------------------
    one_mask = (target_count == 1)

    one_cost = torch.minimum(d_l0, d_r0)
    cost = torch.where(one_mask, one_cost, cost)

    use_left = d_l0 <= d_r0

    one_upward_reward = torch.where(
        use_left,
        (~h_l0).float() * up_vel_l,
        (~h_r0).float() * up_vel_r,
    )

    one_downward_reward = torch.where(
        use_left,
        h_l0.float() * down_vel_l,
        h_r0.float() * down_vel_r,
    )

    upward_reward = torch.where(one_mask, one_upward_reward, upward_reward)
    downward_reward = torch.where(one_mask, one_downward_reward, downward_reward)

    # --------------------
    # case 2: 타겟 2개
    # --------------------
    two_mask = (target_count >= 2)

    # assignment 2가지
    cost_case1 = (d_l0 + d_r1) / 2
    cost_case2 = (d_l1 + d_r0) / 2

    two_cost = torch.minimum(cost_case1, cost_case2)
    cost = torch.where(two_mask, two_cost, cost)

    use_case1 = cost_case1 <= cost_case2

    two_upward_reward = torch.where(
        use_case1,
        (~h_l0).float() * up_vel_l + (~h_r1).float() * up_vel_r,
        (~h_l1).float() * up_vel_l + (~h_r0).float() * up_vel_r,
    ) / 2

    two_downward_reward = torch.where(
        use_case1,
        h_l0.float() * down_vel_l + h_r1.float() * down_vel_r,
        h_l1.float() * down_vel_l + h_r0.float() * down_vel_r,
    ) / 2
    
    upward_reward = torch.where(two_mask, two_upward_reward, upward_reward)
    downward_reward = torch.where(two_mask, two_downward_reward, downward_reward)
    
    # --------------------
    # case 3: 타겟 0개
    # --------------------
    #
    # 나중에 필요하면 작성
    #

    return cost, upward_reward, downward_reward

@torch.jit.script
def compute_reward_terms(

    success: torch.Tensor,          # (N, M)
    wrong_hit: torch.Tensor,        # (N, M)
    missed_target: torch.Tensor,    # (N, M)
    time_error: torch.Tensor,       # (N, M)

    left_tip_pos: torch.Tensor,     # (N, 3) [m]
    right_tip_pos: torch.Tensor,    # (N, 3) [m]
    prev_left_tip_pos: torch.Tensor,
    prev_right_tip_pos: torch.Tensor,
    inst_pos: torch.Tensor,         # (N, M, 3) [m]
    next_hits: torch.Tensor,        # (N, K, M+2)

    tip_vel: torch.Tensor,          # (N, 2, 3)
    hit_armed: torch.Tensor,        # (N, 2, M)

    joint_vel: torch.Tensor,        # (N, num_joints) [rad/s]
    action: torch.Tensor,           # (N, 9) [-1,1]
    robot_pos: torch.Tensor,        # (N, 9) [rad]
    joint_low: torch.Tensor,        # (1, 9) [rad]
    joint_high: torch.Tensor,       # (1, 9) [rad]

    w_inst_success: torch.Tensor,   # (1, M)

    k_accuracy: float,
    k_time_to_hit: float,
    limit_margin: float,

    x_limit: float,
    y_limit_l: float,
    y_limit_h: float,
    z_limit: float,
    drum_xy_margin: float,
    drum_z_margin: float,
):
    # -------------------------------------------------
    # goal terms
    # -------------------------------------------------
    success_reward = (
        success.float() * w_inst_success           # (N, M)
    ).sum(dim=-1)                                  # (N,)
    # success_reward = success.float().sum(dim=-1)        # (N,)
    wrong_cost = wrong_hit.float().sum(dim=-1)        # (N,)
    missed_cost = missed_target.float().sum(dim=-1)   # (N,)

    valid_mask = time_error >= 0
    time_accuracy_reward = torch.exp(-k_accuracy * time_error)
    time_accuracy_reward[~valid_mask] = 0
    time_accuracy_reward = time_accuracy_reward.sum(dim=-1)

    # -------------------------------------------------
    # proximity terms: nearest imminent target only
    # -------------------------------------------------
    _, M, _ = inst_pos.shape

    nearest_target_mask = next_hits[:, 0, :M] > 0.5
    first_time = next_hits[:, 0, M]

    curr_cost, upward_reward, downward_reward = assignment_reward_for_targets(
        target_mask=nearest_target_mask,
        inst_pos=inst_pos,
        left_tip_pos=left_tip_pos,
        right_tip_pos=right_tip_pos,
        tip_vel=tip_vel,
        hit_armed=hit_armed,
    )

    prev_cost, _, _ = assignment_reward_for_targets(
        target_mask=nearest_target_mask,
        inst_pos=inst_pos,
        left_tip_pos=prev_left_tip_pos,
        right_tip_pos=prev_right_tip_pos,
        tip_vel=tip_vel,
        hit_armed=hit_armed,
    )

    proximity_cost = torch.exp(-k_time_to_hit * first_time) * curr_cost
    progress_reward = prev_cost - curr_cost

    # -------------------------------------------------
    # tip position penalties
    # -------------------------------------------------

    tip_limit_pen = (
        (left_tip_pos[:, 0] > x_limit)
        | (left_tip_pos[:, 0] < -x_limit)
        | (left_tip_pos[:, 1] < y_limit_l)
        | (left_tip_pos[:, 1] > y_limit_h)
        | (left_tip_pos[:, 2] < z_limit)

        | (right_tip_pos[:, 0] > x_limit)
        | (right_tip_pos[:, 0] < -x_limit)
        | (right_tip_pos[:, 1] < y_limit_l)
        | (right_tip_pos[:, 1] > y_limit_h)
        | (right_tip_pos[:, 2] < z_limit)
    )

    diff_xy_l = left_tip_pos[:, None, 0:2] - inst_pos[:, :, 0:2]    # (N, M, 2)
    diff_xy_r = right_tip_pos[:, None, 0:2] - inst_pos[:, :, 0:2]

    diff_z_l = left_tip_pos[:, None, 2] - inst_pos[:, :, 2]  # (N, M)
    diff_z_r = right_tip_pos[:, None, 2] - inst_pos[:, :, 2]

    dist_xy_l = torch.sum(diff_xy_l * diff_xy_l, dim=-1)     # (N, M)
    dist_xy_r = torch.sum(diff_xy_r * diff_xy_r, dim=-1)

    in_xy_l = dist_xy_l <= drum_xy_margin ** 2
    in_xy_r = dist_xy_r <= drum_xy_margin ** 2
    
    under_drum_l = in_xy_l & (diff_z_l < -drum_z_margin)
    under_drum_r = in_xy_r & (diff_z_r < -drum_z_margin)

    under_drum_pen = under_drum_l.float().sum(dim=-1) + under_drum_r.float().sum(dim=-1)

    # -------------------------------------------------
    # global penalties
    # -------------------------------------------------
    action_l2 = torch.sum(action * action, dim=-1)
    joint_vel_l2 = torch.sum(joint_vel * joint_vel, dim=-1)

    low_v = torch.clamp((joint_low + limit_margin) - robot_pos, min=0.0)
    high_v = torch.clamp(robot_pos - (joint_high - limit_margin), min=0.0)
    limit_pen = torch.sum(low_v * low_v + high_v * high_v, dim=-1)

    return (
        success_reward, wrong_cost, missed_cost, time_accuracy_reward,
        proximity_cost, progress_reward,
        upward_reward, downward_reward,
        action_l2, joint_vel_l2, limit_pen, tip_limit_pen, under_drum_pen,
    )

@torch.jit.script
def compute_rewards(
    success_reward: torch.Tensor,
    wrong_cost: torch.Tensor,
    missed_cost: torch.Tensor,
    time_accuracy_reward: torch.Tensor,

    proximity_cost: torch.Tensor,
    progress_reward: torch.Tensor,

    upward_reward: torch.Tensor,
    downward_reward: torch.Tensor,

    action_l2: torch.Tensor,
    joint_vel_l2: torch.Tensor,
    limit_pen: torch.Tensor,
    tip_limit_pen: torch.Tensor,
    under_drum_pen: torch.Tensor,

    w_success: float,
    w_wrong: float,
    w_miss: float,
    w_time_accuracy: float,

    w_progress: float,
    w_proximity: float,

    w_upward: float,
    w_downward: float,

    w_action: float,
    w_joint_vel: float,
    w_limit: float,
    w_tip_limit: float,
    w_under_drum: float,
):
    reward = (
        w_success * success_reward
        - w_wrong * wrong_cost
        - w_miss * missed_cost
        + w_time_accuracy * time_accuracy_reward

        - w_proximity * proximity_cost
        + w_progress * progress_reward

        + w_upward * upward_reward
        + w_downward * downward_reward

        - w_action * action_l2
        - w_joint_vel * joint_vel_l2
        - w_limit * limit_pen
        - w_tip_limit * tip_limit_pen
        - w_under_drum * under_drum_pen
    )
    return reward

# === 클래스는 jit 함수 호출해서 사용 ===
@dataclass
class RewardComputerCfg:
    # 하이퍼 파라미터
    limit_margin = 0.08
    k_accuracy = 1.0
    k_time_to_hit = 3.0

    # 팁이 가면 안되는 범위
    x_limit = 0.5
    y_limit_l = 0.2
    y_limit_h = 0.8
    z_limit = -0.6
    drum_xy_margin = 0.15
    drum_z_margin = 0.05

    # 가중치
    w_success = 5.0
    w_wrong = 1.5
    w_miss = 2.0
    w_time_accuracy = 5.0

    w_progress = 4.0
    w_proximity = 1.5

    w_upward = 0.35
    w_downward = 0.30

    w_action = 0.0005
    w_joint_vel = 0.0003
    w_limit = 0.5
    w_tip_limit = 0.15
    w_under_drum = 0.08

class RewardComputer:
    def __init__(self, device: torch.device | str, cfg: RewardComputerCfg):
        self.device = torch.device(device)
        self.cfg = cfg

        # 악기별 보상 가중치
        M = 8
        self.w_inst_success = torch.ones((1, M), device=self.device)
        self.n_success = torch.zeros((1, M), device=self.device)
        self.n_hit = torch.zeros((1, M), device=self.device)

    def compute(
            self,
            success: torch.Tensor,
            wrong_hit: torch.Tensor,
            missed_target: torch.Tensor,
            time_error: torch.Tensor,
            tip_pos: torch.Tensor,
            prev_tip_pos: torch.Tensor,
            inst_pos: torch.Tensor,
            next_hits: torch.Tensor,
            tip_vel: torch.Tensor,
            hit_armed_for_reward: torch.Tensor,
            robot_vel: torch.Tensor,
            actions: torch.Tensor,
            robot_pos: torch.Tensor,
            joint_low: torch.Tensor,
            joint_high: torch.Tensor,
    ):
        (
            success_reward, wrong_cost, missed_cost, time_accuracy_reward,
            proximity_cost, progress_reward,
            upward_reward, downward_reward,
            action_l2, joint_vel_l2, limit_pen, tip_limit_pen, under_drum_pen,
        ) = compute_reward_terms(
            success=success,
            wrong_hit=wrong_hit,
            missed_target=missed_target,
            time_error=time_error,

            left_tip_pos=tip_pos[:, 0, :],
            right_tip_pos=tip_pos[:, 1, :],
            prev_left_tip_pos=prev_tip_pos[:, 0, :],
            prev_right_tip_pos=prev_tip_pos[:, 1, :],
            inst_pos=inst_pos,
            next_hits=next_hits,

            tip_vel=tip_vel,
            hit_armed=hit_armed_for_reward,

            joint_vel=robot_vel,
            action=actions,
            robot_pos=robot_pos,
            joint_low=joint_low,
            joint_high=joint_high,

            w_inst_success=self.w_inst_success,

            k_accuracy=self.cfg.k_accuracy,
            k_time_to_hit=self.cfg.k_time_to_hit,
            limit_margin=self.cfg.limit_margin,

            x_limit=self.cfg.x_limit,
            y_limit_l=self.cfg.y_limit_l,
            y_limit_h=self.cfg.y_limit_h,
            z_limit=self.cfg.z_limit,
            drum_xy_margin=self.cfg.drum_xy_margin,
            drum_z_margin=self.cfg.drum_z_margin,
        )

        reward = compute_rewards(
            success_reward=success_reward,
            wrong_cost=wrong_cost,
            missed_cost=missed_cost,
            time_accuracy_reward=time_accuracy_reward,
            
            proximity_cost=proximity_cost,
            progress_reward=progress_reward,

            upward_reward=upward_reward,
            downward_reward=downward_reward,

            action_l2=action_l2,
            joint_vel_l2=joint_vel_l2,
            limit_pen=limit_pen,
            tip_limit_pen=tip_limit_pen,
            under_drum_pen=under_drum_pen,

            w_success=self.cfg.w_success,
            w_wrong=self.cfg.w_wrong,
            w_miss=self.cfg.w_miss,
            w_time_accuracy=self.cfg.w_time_accuracy,

            w_progress=self.cfg.w_progress,
            w_proximity=self.cfg.w_proximity,

            w_upward=self.cfg.w_upward,
            w_downward=self.cfg.w_downward,
            
            w_action=self.cfg.w_action,
            w_joint_vel=self.cfg.w_joint_vel,
            w_limit=self.cfg.w_limit,
            w_tip_limit=self.cfg.w_tip_limit,
            w_under_drum=self.cfg.w_under_drum,
        )

        num_hit_inst = success.float() + missed_target.float()

        num_hit = num_hit_inst.float().sum(dim=-1)
        num_success = success.float().sum(dim=-1)
        num_wrong = wrong_hit.float().sum(dim=-1)
        num_missed = missed_target.float().sum(dim=-1)

        terms = {
            "reward": reward,
            "proximity": proximity_cost,
            "progress(x100)": progress_reward * 100,
            "upward": upward_reward,
            "downward": downward_reward,
            "action_l2": action_l2,
            "joint_vel_l2": joint_vel_l2,
            "limit_pen(x100)": limit_pen * 100,
            "tip_limit_pen": tip_limit_pen,
            "under_drum_pen": under_drum_pen,
        }

        p_terms = {
            "success_rate": torch.stack([num_success, num_hit], dim=-1),
            "wrong_rate": torch.stack([num_wrong, num_hit], dim=-1),
            "miss_rate": torch.stack([num_missed, num_hit], dim=-1),

            "snare_success_rate": torch.stack([success[:, 0], num_hit_inst[:, 0]], dim=-1),
            "snare_wrong_rate": torch.stack([wrong_hit[:, 0], num_hit_inst[:, 0]], dim=-1),
            "snare_miss_rate": torch.stack([missed_target[:, 0], num_hit_inst[:, 0]], dim=-1),

            "floor_success_rate": torch.stack([success[:, 1], num_hit_inst[:, 1]], dim=-1),
            "floor_wrong_rate": torch.stack([wrong_hit[:, 1], num_hit_inst[:, 1]], dim=-1),
            "floor_miss_rate": torch.stack([missed_target[:, 1], num_hit_inst[:, 1]], dim=-1),

            "mid_success_rate": torch.stack([success[:, 2], num_hit_inst[:, 2]], dim=-1),
            "mid_wrong_rate": torch.stack([wrong_hit[:, 2], num_hit_inst[:, 2]], dim=-1),
            "mid_miss_rate": torch.stack([missed_target[:, 2], num_hit_inst[:, 2]], dim=-1),

            "high_success_rate": torch.stack([success[:, 3], num_hit_inst[:, 3]], dim=-1),
            "high_wrong_rate": torch.stack([wrong_hit[:, 3], num_hit_inst[:, 3]], dim=-1),
            "high_miss_rate": torch.stack([missed_target[:, 3], num_hit_inst[:, 3]], dim=-1),

            "hihat_success_rate": torch.stack([success[:, 4], num_hit_inst[:, 4]], dim=-1),
            "hihat_wrong_rate": torch.stack([wrong_hit[:, 4], num_hit_inst[:, 4]], dim=-1),
            "hihat_miss_rate": torch.stack([missed_target[:, 4], num_hit_inst[:, 4]], dim=-1),

            "ride_success_rate": torch.stack([success[:, 5], num_hit_inst[:, 5]], dim=-1),
            "ride_wrong_rate": torch.stack([wrong_hit[:, 5], num_hit_inst[:, 5]], dim=-1),
            "ride_miss_rate": torch.stack([missed_target[:, 5], num_hit_inst[:, 5]], dim=-1),

            "crash1_success_rate": torch.stack([success[:, 6], num_hit_inst[:, 6]], dim=-1),
            "crash1_wrong_rate": torch.stack([wrong_hit[:, 6], num_hit_inst[:, 6]], dim=-1),
            "crash1_miss_rate": torch.stack([missed_target[:, 6], num_hit_inst[:, 6]], dim=-1),

            "crash2_success_rate": torch.stack([success[:, 7], num_hit_inst[:, 7]], dim=-1),
            "crash2_wrong_rate": torch.stack([wrong_hit[:, 7], num_hit_inst[:, 7]], dim=-1),
            "crash2_miss_rate": torch.stack([missed_target[:, 7], num_hit_inst[:, 7]], dim=-1),
        }

        return reward, terms, p_terms
    
    def update_difficulty_weights(
            self,
            is_update: bool,
            success: torch.Tensor,
            missed_target: torch.Tensor,
    ):
        # 어려운 드럼에 더 큰 가중치
        if is_update:
            eps = 1e-6
            success_ratio = self.n_success / (self.n_hit + eps)
            self.w_inst_success = 1.0 / (success_ratio + eps)

            # 평균 1로 정규화
            self.w_inst_success = (self.w_inst_success / self.w_inst_success.mean())

            self.n_success[:] = 0
            self.n_hit[:] = 0
        else:
            self.n_success = self.n_success + success.float().sum(dim=0)
            self.n_hit = self.n_hit + success.float().sum(dim=0) + missed_target.float().sum(dim=0)