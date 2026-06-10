# drum_robot/tasks/legacy_task/dr_env.py

from __future__ import annotations

import torch
from typing import Sequence
import math
from isaaclab.utils import math as math_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane

from .dr_cfg import DrumRobotEnvCfg
from drum_robot.utils.logger import EnvLogger, LoggerCfg

import numpy as np
import gymnasium as gym

import omni.usd
from pxr import UsdGeom, Gf


class DrumRobotEnv(DirectRLEnv):
    cfg: DrumRobotEnvCfg

    # ============================================================
    # [Override Functions from DirectRLEnv]
    # These functions are automatically called by the RL loop.
    # DO NOT change signature.
    # 실행 순서 _pre_physics_step -> _apply_action (*decimation) -> _get_dones -> _get_rewards -> _get_observations
    # ============================================================

    def __init__(self, cfg: DrumRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._setup_spaces()    # Space 정의
        self._alloc_buffers()   # 버퍼 초기화
        self._setup_constants_from_cfg()    # cfg 상수 텐서화

        self._bind_asset_handles()      # body/joint id resolve
        self._build_joint_tensors()     # limits, norm params, dir tensor
        self._build_drum_tensors()

        self._init_obs_normalization()  # 관측값 정규화를 위한 변수 초기화

        # 로그
        self.logger = EnvLogger(self.num_envs, self.device, LoggerCfg(interval=2000, sample_env_id=0))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg()) # add ground plane
        self.scene.articulations["robot"] = self.robot  # add articulation to scene
        self.scene.clone_environments(copy_from_source=False)   # clone and replicate

        self._init_visualization()  # 시각화
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))  # 광원 추가
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):     
        # RL 에이전트로부터 받은 actions 저장
        actions = actions.to(torch.float32)

        # action 범위를 [-1, 1]로 강제
        actions = torch.clamp(actions, -1.0, 1.0)

        # 로봇 위치 가져오기
        usd_q = self.robot.data.joint_pos[:, self.ctrl_joint_ids]
        robot_q = self._convert_usd_to_robot(usd_q)

        # 목표 위치 = 현재 위치 + actions
        robot_q_1 = robot_q + actions * self.cfg.action_scale
        robot_q_1 = torch.max(torch.min(robot_q_1, self.joint_high), self.joint_low)  # joint_limit 내로 clip
        usd_q_1 = self._convert_robot_to_usd(robot_q_1)

        self.actions = actions.clone()
        self.target_joint_pos = usd_q_1

    def _apply_action(self):
        self.robot.set_joint_position_target(self.target_joint_pos, joint_ids=self.ctrl_joint_ids)

    def _get_observations(self) -> dict:
        # 로봇의 관절 위치
        usd_pos = self.robot.data.joint_pos[:, self.ctrl_joint_ids]        # (num_envs, 9)
        joint_pos = self._convert_usd_to_robot(usd_pos)

        # 로봇의 관절 속도
        usd_vel = self.robot.data.joint_vel[:, self.ctrl_joint_ids]        # (num_envs, 9)
        joint_vel = self._convert_usd_to_robot(usd_vel)

        # 타겟 one-hot 벡터
        one_hot_target_id_L = self.one_hot_target_id[:,0,:].float()
        one_hot_target_id_R = self.one_hot_target_id[:,1,:].float()

        # 위치 오차
        err = self.targets - self.tip_pos

        # 팁 속도
        tip_vel = self.tip_vel

        # 성공 상태 플래그
        success_flag_L = self.success_L.float().unsqueeze(1)
        success_flag_R = self.success_R.float().unsqueeze(1)

        # impact armed 상태
        impact_armed_flag_L = self.impact_armed_L.float().unsqueeze(1)
        impact_armed_flag_R = self.impact_armed_R.float().unsqueeze(1)

        obs = self._normalize_and_pack_obs(
            joint_pos,
            joint_vel,
            one_hot_target_id_L,
            one_hot_target_id_R,
            err,
            tip_vel,
            success_flag_L,
            success_flag_R,
            impact_armed_flag_L,
            impact_armed_flag_R,
            )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:

        # robot joint pos (robot frame) for limit barrier
        usd_pos = self.robot.data.joint_pos[:, self.ctrl_joint_ids]
        usd_vel = self.robot.data.joint_vel[:, self.ctrl_joint_ids]
        robot_pos = self._convert_usd_to_robot(usd_pos)  # (N, 9)
        robot_vel = self._convert_usd_to_robot(usd_vel)

        (
            xy_term,
            z_term,
            exp_xy_term,
            exp_z_term,
            downward_term,
            upward_term,
            action_l2,
            joint_vel_l2,
            limit_pen,
            left_xy_dist,
            right_xy_dist,
            left_z_err,
            right_z_err,
        ) = compute_reward_terms(
            self.tip_pos[:, :3],
            self.tip_pos[:, 3:],
            self.tip_vel[:, :3],
            self.tip_vel[:, 3:],
            self.targets[:, :3],
            self.targets[:, 3:],
            self.impact_armed_L.to(torch.float32),
            self.impact_armed_R.to(torch.float32),
            self.actions,
            robot_vel,
            robot_pos,
            self.joint_low,
            self.joint_high,
            float(self.cfg.exp_k_xy),
            float(self.cfg.exp_k_z),
            float(self.cfg.limit_margin),
        )

        r = compute_rewards(
            xy_term,
            z_term,
            exp_xy_term,
            exp_z_term,
            downward_term,
            upward_term,
            action_l2,
            joint_vel_l2,
            limit_pen,
            float(self.cfg.rew_w_xy),
            float(self.cfg.rew_w_z),
            float(self.cfg.rew_w_exp_xy),
            float(self.cfg.rew_w_exp_z),
            float(self.cfg.rew_w_down),
            float(self.cfg.rew_w_up),
            float(self.cfg.rew_w_action),
            float(self.cfg.rew_w_joint_vel),
            float(self.cfg.rew_w_limit),
        )

        # 성공 보너스
        bonus = float(self.cfg.success_bonus)
        r = r + self.first_hit_L.to(torch.float32) * bonus + self.first_hit_R.to(torch.float32) * bonus

        # 로그 출력
        terms = {
            "reward": r,
            "left_xy_dist": left_xy_dist,
            "right_xy_dist": right_xy_dist,
            "left_z_err": left_z_err,
            "right_z_err": right_z_err,
            "downward_term": downward_term,
            "upward_term": upward_term,
            "action_l2": action_l2,
            "joint_vel_l2": joint_vel_l2,
            "limit_pen": limit_pen,
            "impact_armed_L": self.impact_armed_L.to(torch.float32),
            "impact_armed_R": self.impact_armed_R.to(torch.float32),
            "first_hit_L": self.first_hit_L.to(torch.float32),
            "first_hit_R": self.first_hit_R.to(torch.float32),
        }
        self.logger.add(terms)
        self.logger.maybe_flush()

        return r
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 팁 위치
        tip_pos = self._compute_tip_position()           # (num_envs, 3)
        
        # 팁 속도
        tip_prev = self.tip_pos
        vel = (tip_pos - tip_prev) / self.dt

        # 가장 가까운 드럼 찾기
        inst_pos = self.inst_pos
        nearest_idx, nearest_dist_sq = self._find_nearest_drum(tip_pos, inst_pos)    # (num_envs, 2)

        # 타격 확인
        hit_mask = self._check_hit(tip_pos, vel, inst_pos, nearest_idx, nearest_dist_sq)

        self.tip_pos = tip_pos
        self.tip_vel = vel
        self.hit = hit_mask

        # 팁 표시
        if self.cfg.enable_tip_markers:
            self._update_tip_markers()

        # 성공 확정 시 종료(terminate)  # 자가 충돌은 보상을 먼저 넣어 회피하게 하고 안정되면 죽이는 코드 만들기
        died = self.episode_success.clone() # 이전 step까지의 성공 상태, 다음 step 종료용 (성공 보상을 주기 위함)

        self.first_hit_L = (~self.success_L) & hit_mask[:, 0]
        self.first_hit_R = (~self.success_R) & hit_mask[:, 1]

        self.success_L = self.success_L | hit_mask[:, 0]
        self.success_R = self.success_R | hit_mask[:, 1]

        self.episode_success = self.success_L & self.success_R

        # episode time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # 부모 클래스의 리셋(버퍼 초기화 등) 호출
        super()._reset_idx(env_ids)

        # 목표 위치(드럼) 리셋
        target_pos, inst_pos, pairs = self._reset_targets(env_ids)
        self.inst_pos[env_ids] = inst_pos
        self.targets[env_ids] = target_pos

        # 타겟 one-hot 벡터 만들기
        self.one_hot_target_id[env_ids] = 0
        self.one_hot_target_id[env_ids, 0, pairs[:,0]] = 1
        self.one_hot_target_id[env_ids, 1, pairs[:,1]] = 1

        # 드럼 표시
        if self.cfg.enable_drums:
            self._update_drums_pos(env_ids, inst_pos)
            self._update_drums_color(env_ids, pairs)

        # 로봇 자세 리셋
        joint_pos, joint_vel = self._reset_init_pos(env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 텐서 변수들 리셋
        self._reset_tensors(env_ids)
        
    # ============================================================
    # [Custom Functions]
    # Internal utility methods (NOT called by RL engine directly)
    # ============================================================

    """ init """
    def _setup_spaces(self):
        # 반드시 single-env shape로 정의
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
        )

        # print("[DEBUG] action_space:", self.action_space)
        # print("[DEBUG] observation_space:", self.observation_space)
        # print("[DEBUG] cfg.action_dim:", getattr(self.cfg, "action_dim", None))

    def _alloc_buffers(self):
        # env_ids
        self.env_ids = torch.arange(self.num_envs, device=self.device)

        # 로봇의 tip 위치를 저장할 텐서
        self.tip_pos = torch.zeros((self.num_envs, 6), device=self.device)

        # 로봇의 tip 속도 저장할 텐서
        self.tip_vel = torch.zeros((self.num_envs, 6), device=self.device)
        
        # 목표 지점을 저장할 텐서 (num_envs, 3차원 x 2)
        self.targets = torch.zeros((self.num_envs, 6), device=self.device)

        # 타격 상태 버퍼
        self.impact_armed_L = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.impact_armed_R = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.hit = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)

        # 성공 버퍼
        self.first_hit_L = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.first_hit_R = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.success_L = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.success_R = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.episode_success = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    def _setup_constants_from_cfg(self):
        # dt
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # offset wrist link to tip
        L_off = torch.tensor(self.cfg.tip_offset_left, device=self.device, dtype=torch.float32)  # (3,)
        R_off = torch.tensor(self.cfg.tip_offset_right, device=self.device, dtype=torch.float32)

        self.tip_offset_L = L_off.unsqueeze(0).expand(self.num_envs, 3)
        self.tip_offset_R = R_off.unsqueeze(0).expand(self.num_envs, 3)

    def _bind_asset_handles(self):
        self._bind_body_ids()
        self._bind_joint_ids()
    
    def _bind_body_ids(self):
        # 양손 스틱 링크의 인덱스
        left = self.robot.find_bodies("left_wrist")     # ([10], ['left_wrist'])
        right = self.robot.find_bodies("right_wrist")   # ([11], ['right_wrist'])
        if len(left) == 0 or len(right) == 0:
            raise RuntimeError("wrist body not found. Check USD body names.")
        self.left_stick_idx = left[0]
        self.right_stick_idx = right[0]

        # print("[DEBUG] body_names: ", self.robot.data.body_names)
    
    def _bind_joint_ids(self):

        # 제어할 관절만
        joint_names = [
            "waist_joint",
            "left_shoulder_1","left_shoulder_2","left_elbow",
            "right_shoulder_1","right_shoulder_2","right_elbow",
            "left_wrist","right_wrist",
        ]

        ids, names = self.robot.find_joints(joint_names)
        # self.joint_name_to_id = { name: jid for name, jid in zip(names, ids) }
        
        self.ctrl_joint_ids = ids
        self.ctrl_joint_names = names
        if len(self.ctrl_joint_ids) != len(joint_names):
            raise RuntimeError(f"ctrl_joint_ids mismatch: {len(self.ctrl_joint_ids)} (expected 9). names={self.ctrl_joint_names}")
        
        # print("[DEBUG] find_joints ids: ", self.ctrl_joint_ids)
        # print("[DEBUG] find_joints names: ", self.ctrl_joint_names)

    def _build_joint_tensors(self):

        """ 반드시 self.ctrl_joint_names 순서대로 텐서를 만들어여 함 """

        # 관절 제한값
        self.joint_low = torch.tensor(
            [self.cfg.joint_limit[name][0] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)  # unsqueeze(n) n번째 차원에 1인 차원을 생성

        self.joint_high = torch.tensor(
            [self.cfg.joint_limit[name][1] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

        # 로봇 관절-USD 관절 방향
        self.dir_tensor = torch.tensor(
            [self.cfg.joint_usd_dir[name] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

        # print("[DEBUG] self.joint_low: ", self.joint_low)
        # print("[DEBUG] self.joint_high: ", self.joint_high)

        # 초기 위치값
        self.init_joint_min = torch.tensor(
            [self.cfg.init_joint_range[name][0] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

        self.init_joint_max = torch.tensor(
            [self.cfg.init_joint_range[name][1] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

    def _build_drum_tensors(self):
        self.drum_set_tensor = torch.tensor(
            self.cfg.drum_set,
            device=self.device,
            dtype=torch.long
        )
        self.num_drum_sets = self.drum_set_tensor.shape[0]

        self.inst_names = list(self.cfg.instruments.keys())
        self.basic_inst_pos = torch.tensor(
            list(self.cfg.instruments.values()),
            device=self.device,
            dtype=torch.float32
        )

        # 각 env/각 에피소드에서 악기 위치
        self.inst_pos = self.basic_inst_pos.unsqueeze(0).repeat(self.num_envs, 1, 1)    # (num_envs, 8, 3)

        # 목표 악기을 저장할 one-hot 텐서 (num_envs, 2, 악기 개수)
        self.one_hot_target_id = torch.zeros((self.num_envs, 2, len(self.inst_names)), device=self.device, dtype=torch.int32)

    def _init_obs_normalization(self):
        # joint
        self.joint_center = 0.5 * (self.joint_low + self.joint_high)
        self.joint_half_range = 0.5 * (self.joint_high - self.joint_low) + 1e-6

        self.joint_vel_scale = torch.tensor(self.cfg.joint_vel_scale, device=self.device, dtype=torch.float32)

        # target
        inst_min = self.basic_inst_pos.min(dim=0).values   # (3,)
        inst_max = self.basic_inst_pos.max(dim=0).values   # (3,)

        center = 0.5 * (inst_min + inst_max)
        half_range = 0.5 * (inst_max - inst_min) + 1e-6
        
        self.target_center = torch.cat([center, center], dim=0).unsqueeze(0)
        self.target_half_range = torch.cat([half_range, half_range], dim=0).unsqueeze(0)

        self.tip_vel_scale = torch.tensor(self.cfg.tip_vel_scale, device=self.device, dtype=torch.float32)

    """ util """
    def _convert_usd_to_robot(self, usd_data):
        return self.dir_tensor * usd_data

    def _convert_robot_to_usd(self, robot_data):
        return self.dir_tensor * robot_data

    """ func (_get_observations) """
    def _normalize_and_pack_obs(self,
            joint_pos,
            joint_vel,
            one_hot_target_id_L,
            one_hot_target_id_R,
            err,
            tip_vel,
            success_flag_L,
            success_flag_R,
            impact_armed_flag_L,
            impact_armed_flag_R,
            ) -> torch.Tensor:
        # normalize
        joint_pos_n = (joint_pos - self.joint_center) / self.joint_half_range
        joint_pos_n = torch.clamp(joint_pos_n, -1.5, 1.5)

        joint_vel_n = joint_vel / self.joint_vel_scale
        joint_vel_n = torch.clamp(joint_vel_n, -10.0, 10.0)

        err_n = err / self.target_half_range
        err_n = torch.clamp(err_n, -2.0, 2.0)

        tip_vel_n = tip_vel / self.tip_vel_scale
        tip_vel_n = torch.clamp(tip_vel_n, -10.0, 10.0)

        obs = torch.cat(
            [
                joint_pos_n,
                joint_vel_n,
                one_hot_target_id_L,
                one_hot_target_id_R,
                err_n,
                tip_vel_n,
                success_flag_L,
                success_flag_R,
                impact_armed_flag_L,
                impact_armed_flag_R,
            ],
            dim=-1
        )
        
        return obs

    """ func (_get_dones) """
    def _compute_tip_position(self):
        # 스틱 링크의 월드 좌표계 위치 가져오기
        all_body_pos = self.robot.data.body_pos_w      # (num_envs, num_bodies, 3)
        all_body_quat = self.robot.data.body_quat_w    # (num_envs, num_bodies, 4)  (w,x,y,z)인 경우가 많음

        L_wrist_pos = all_body_pos[:, self.left_stick_idx, :].squeeze(1)       # (num_envs, 1, 3) -> (num_envs, 3) 1인 차원을 제거
        R_wrist_pos = all_body_pos[:, self.right_stick_idx, :].squeeze(1)
        L_quat = all_body_quat[:, self.left_stick_idx, :].squeeze(1)     # (num_envs, 1, 4) -> (num_envs, 4)
        R_quat = all_body_quat[:, self.right_stick_idx, :].squeeze(1)

        # 팁 위치 구하기
        L_tip_w = L_wrist_pos + math_utils.quat_apply(L_quat, self.tip_offset_L)
        R_tip_w = R_wrist_pos + math_utils.quat_apply(R_quat, self.tip_offset_R)

        # 월드 기준 -> env 기준
        L_tip = L_tip_w - self.scene.env_origins
        R_tip = R_tip_w - self.scene.env_origins

        tip_pos = torch.cat([L_tip, R_tip], dim=-1)

        return tip_pos

    def _find_nearest_drum(self, tip_pos, inst_pos):
        
        tip_L = tip_pos[:, :3]
        tip_R = tip_pos[:, 3:]

        nearest_idx_L, nearest_dist_sq_L = self._find_nearest_drum_for_hand(tip_L, inst_pos)
        nearest_idx_R, nearest_dist_sq_R = self._find_nearest_drum_for_hand(tip_R, inst_pos)

        nearest_idx = torch.cat([nearest_idx_L, nearest_idx_R],dim=-1)  # (num_envs, 2)
        nearest_dist_sq = torch.cat([nearest_dist_sq_L, nearest_dist_sq_R],dim=-1)

        # # 테스트
        # if self.cfg.enable_drums:
        #     self._update_drums_color(self.env_ids, nearest_idx)

        return nearest_idx, nearest_dist_sq

    def _find_nearest_drum_for_hand(self, tip_pos, inst_pos):
        # one hand tip_pos [N, 3], inst_pos[N, 8, 3]

        # xy 거리 (이인우) xy 거리로 하면 드럼이 겹쳐진 경우 치는 드럼이 가장 가까운 드럼이 아닐 수도 있음
        diff = tip_pos[:, None, 0:2] - inst_pos[:, :, 0:2]  # (N, 8, 2) = (N, 1, 2) - (N, 8, 2)

        # dist^2 계산
        dist_sq = torch.sum(diff * diff, dim=-1)                  # (N, 8)

        # 가까운 드럼
        nearest_idx = torch.argmin(dist_sq, dim=1).unsqueeze(1)                # (N, 1)

        # 가장 가까운 드럼 거리
        nearest_dist_sq = torch.gather(dist_sq, 1, nearest_idx)

        return nearest_idx, nearest_dist_sq

    def _check_hit(self, tip_pos, vel, inst_pos, nearest_idx, nearest_dist_sq):

        tip_L = tip_pos[:, :3]
        tip_R = tip_pos[:, 3:]

        vel_L = vel[:, :3]
        vel_R = vel[:, 3:]
        
        nearest_idx_L = nearest_idx[:, 0]
        nearest_idx_R = nearest_idx[:, 1]

        nearest_dist_sq_L = nearest_dist_sq[:, 0]
        nearest_dist_sq_R = nearest_dist_sq[:, 1]

        impact_armed_L = self.impact_armed_L
        impact_armed_R = self.impact_armed_R

        drum_z_L = inst_pos[self.env_ids, nearest_idx_L, 2]
        drum_z_R = inst_pos[self.env_ids, nearest_idx_R, 2]

        hit_L, impact_armed_L = self._check_hit_for_hand(tip_L, vel_L, drum_z_L, nearest_dist_sq_L, impact_armed_L)
        hit_R, impact_armed_R = self._check_hit_for_hand(tip_R, vel_R, drum_z_R, nearest_dist_sq_R, impact_armed_R)

        self.impact_armed_L = impact_armed_L
        self.impact_armed_R = impact_armed_R

        hit_mask = torch.cat([hit_L, hit_R], dim=-1)

        return hit_mask

    def _check_hit_for_hand(self, pos, vel, drum_z, dist_sq, impact_armed):

        pos_z = pos[:, 2]
        vel_z = vel[:, 2]

        # 범위 체크
        radius_sq = self.cfg.drum_xy_range ** 2
        in_range = dist_sq <= radius_sq

        z_in_margin = torch.abs(pos_z - drum_z) <= self.cfg.drum_z_margin

        # z 속도 체크
        impact_mask = vel_z < -self.cfg.min_impact_velocity
        rebound_mask = vel_z > self.cfg.min_rebound_velocity

        # 충분히 내려치는 구간을 한번이라도 거치면 armed
        impact_armed = impact_armed | (impact_mask & in_range)

        # 범위 밖으로 나가면 armed 해제
        impact_armed = impact_armed & in_range

        # armed 상태에서 rebound + z margin이면 hit
        hit_mask = impact_armed & rebound_mask & z_in_margin

        # hit 후 armed 해제
        impact_armed = impact_armed & (~hit_mask)

        hit_mask = hit_mask.unsqueeze(1)

        return hit_mask, impact_armed

    """ func (_reset_idx) """
    def _reset_targets(self, env_ids):
        # (0) target perturbation
        inst_pos = self.basic_inst_pos.unsqueeze(0).repeat(len(env_ids), 1, 1)  # + 알파

        # (1) env마다 drum_set에서 하나 고르기
        set_idx = torch.randint(0, self.num_drum_sets, (len(env_ids),), device=self.device)   # (N,)
        pairs = self.drum_set_tensor[set_idx] - 1                                      # (N,2), drum_set_tensor 값은 1~8 -> 0~7 인덱스로 변환 (파이썬 인덱싱용)

        L_idx = pairs[:, 0]
        R_idx = pairs[:, 1]

        # (3) 타겟 좌표 가져오기 (waist joint 기준)
        env_arange = torch.arange(len(env_ids), device=self.device)
        L_pos = inst_pos[env_arange, L_idx]     # (N,3)
        R_pos = inst_pos[env_arange, R_idx]     # (N,3)

        # (4) (N,6)으로 합치기: [Lx,Ly,Lz,Rx,Ry,Rz]
        target_pos = torch.cat([L_pos, R_pos], dim=1)  # (N,6)

        # (5) z 축 보정 (waist joint 기준 -> env/world 기준)
        target_pos[:, 2] += self.cfg.robot_waist_joint_offset_z
        target_pos[:, 5] += self.cfg.robot_waist_joint_offset_z

        return target_pos, inst_pos, pairs

    def _reset_init_pos(self, env_ids):
        # 랜덤 각도 초기 위치
        rand = torch.rand((len(env_ids), len(self.ctrl_joint_names)), device=self.device)   # (N,9)
        init_pos = self.init_joint_min + rand * (self.init_joint_max - self.init_joint_min)
        usd_init_pos = self._convert_robot_to_usd(init_pos)

        # 기본 자세 가져오기
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        default_joint_vel = self.robot.data.default_joint_vel[env_ids]

        joint_pos = default_joint_pos.clone()
        joint_pos[:,self.ctrl_joint_ids] = usd_init_pos
        joint_vel = torch.zeros_like(default_joint_vel)

        return joint_pos, joint_vel

    def _reset_tensors(self, env_ids):
        # 팁 위치/속도 리셋
        tip_pos = self._compute_tip_position()
        self.tip_pos[env_ids] = tip_pos[env_ids]
        self.tip_vel[env_ids] = 0.0

        # 타격 상태 버퍼 리셋
        self.impact_armed_L[env_ids] = False
        self.impact_armed_R[env_ids] = False
        self.hit[env_ids] = False

        # 성공 버퍼 리셋
        self.first_hit_L[env_ids] = False
        self.first_hit_R[env_ids] = False
        self.success_L[env_ids] = False
        self.success_R[env_ids] = False
        self.episode_success[env_ids] = False

    # ============================================================
    # [Debug / Visualization]
    # Not used for training logic
    # ============================================================

    def _init_visualization(self):

        self.color_L = Gf.Vec3f(
            float(self.cfg.color_L[0]),
            float(self.cfg.color_L[1]),
            float(self.cfg.color_L[2])
        )
        self.color_R = Gf.Vec3f(
            float(self.cfg.color_R[0]),
            float(self.cfg.color_R[1]),
            float(self.cfg.color_R[2])
        )

        if self.cfg.enable_tip_markers:
            self._create_tip_markers()

        if self.cfg.enable_drums:
            self._drum_base_color = Gf.Vec3f(0.5, 0.5, 0.5)  # 드럼 기본 색(중립)
            self._drum_both_target_color = Gf.Vec3f(0.5, 0.0, 0.5)  # 목표가 겹칠 때 드럼 색

            self._create_drums()

    def _create_tip_markers(self):
        # 고속 업데이트를 위한 TranslateOp 캐시
        # self.tip_marker_xform_L = []
        # self.tip_marker_xform_R = []
        self._tip_marker_translate_ops_L = []
        self._tip_marker_translate_ops_R = []

        # sphere 설정
        sphere_cfg_L = sim_utils.SphereCfg(
            radius=self.cfg.tip_marker_radius,
        )
        sphere_cfg_R = sim_utils.SphereCfg(
            radius=self.cfg.tip_marker_radius,
        )

        # 현재 IsaacSim의 USD Stage 접근
        stage = omni.usd.get_context().get_stage()

        for i in range(self.num_envs):

            # USD Stage 위에 Prim을 생성 (이미 존재하면 타입을 유지한 채 반환)
            viz_root = f"/World/envs/env_{i}/_viz"
            stage.DefinePrim(viz_root, "Xform")     # Xform = Transform 노드
            xform_L_path = f"{viz_root}/TipMarkerL"
            xform_R_path = f"{viz_root}/TipMarkerR"
            stage.DefinePrim(xform_L_path, "Xform")
            stage.DefinePrim(xform_R_path, "Xform")
            sphere_L_path = f"{xform_L_path}/sphere"
            sphere_R_path = f"{xform_R_path}/sphere"

            # IsValid 체크 후 sphere prim 생성
            if not stage.GetPrimAtPath(sphere_L_path).IsValid():
                sphere_cfg_L.func(sphere_L_path, sphere_cfg_L)
            if not stage.GetPrimAtPath(sphere_R_path).IsValid():
                sphere_cfg_R.func(sphere_R_path, sphere_cfg_R)

            # 색상 설정
            sphere_L_prim = stage.GetPrimAtPath(sphere_L_path)
            sphere_R_prim = stage.GetPrimAtPath(sphere_R_path)
            gprim_L = UsdGeom.Gprim(sphere_L_prim)
            gprim_R = UsdGeom.Gprim(sphere_R_prim)

            gprim_L.CreateDisplayColorPrimvar().Set([self.color_L])
            gprim_R.CreateDisplayColorPrimvar().Set([self.color_R])

            # TranslateOp를 1회 생성하고 캐싱
            prim_L = stage.GetPrimAtPath(xform_L_path)
            prim_R = stage.GetPrimAtPath(xform_R_path)

            xf_L = UsdGeom.Xformable(prim_L)    # prim이 transform 연산을 가질 수 있도록 감싸는 wrapper
            xf_R = UsdGeom.Xformable(prim_R)

            ops_L = xf_L.GetOrderedXformOps()   # 현재 들어있는 transform 연산 목록 가져오기
            if len(ops_L) > 0 and ops_L[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:    # 첫 번째 연산이 TranslateOp
                t_op_L = ops_L[0]
            else:
                # Clear 후 TranslateOp 추가
                xf_L.ClearXformOpOrder()
                t_op_L = xf_L.AddTranslateOp()

            ops_R = xf_R.GetOrderedXformOps()
            if len(ops_R) > 0 and ops_R[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t_op_R = ops_R[0]
            else:
                xf_R.ClearXformOpOrder()
                t_op_R = xf_R.AddTranslateOp()

            # self.tip_marker_xform_L.append(xform_L_path)
            # self.tip_marker_xform_R.append(xform_R_path)
            self._tip_marker_translate_ops_L.append(t_op_L)
            self._tip_marker_translate_ops_R.append(t_op_R)

    def _update_tip_markers(self):
        left_pos = self.tip_pos[:, :3]
        right_pos = self.tip_pos[:, 3:]

        for i in range(self.num_envs):
            
            lp = left_pos[i]
            rp = right_pos[i]

            # torch -> float 바로 (cpu numpy 변환 없이)
            self._tip_marker_translate_ops_L[i].Set(
                Gf.Vec3f(float(lp[0]), float(lp[1]), float(lp[2]))
            )
            self._tip_marker_translate_ops_R[i].Set(
                Gf.Vec3f(float(rp[0]), float(rp[1]), float(rp[2]))
            )

    def _create_drums(self):
        # 고속 업데이트를 위한 TranslateOp 캐시
        self._drum_translate_ops = []
        
        # 고속 업데이트를 위한 색 Primvar 캐시
        self._drum_color_ops = []

        # 현재 IsaacSim의 USD Stage 접근
        stage = omni.usd.get_context().get_stage()

        for i in range(self.num_envs):

            # USD Stage 위에 Prim을 생성 (이미 존재하면 타입을 유지한 채 반환)
            viz_root = f"/World/envs/env_{i}/_viz"
            stage.DefinePrim(viz_root, "Xform")     # Xform = Transform 노드
            drums_root = f"{viz_root}/Drums"
            stage.DefinePrim(drums_root, "Xform")

            translate_ops_env = []
            color_ops_env = []

            inst_names = list(self.cfg.instruments.keys())
            inst_pos = list(self.cfg.instruments.values())

            for j, name in enumerate(inst_names):
                xform_path = f"{drums_root}/{name}"
                cyl_path = f"{xform_path}/cyl"

                stage.DefinePrim(xform_path, "Xform")

                # IsValid 체크 후 prim 생성
                if not stage.GetPrimAtPath(cyl_path).IsValid():
                    stage.DefinePrim(cyl_path, "Cylinder")
                    cyl = UsdGeom.Cylinder(stage.GetPrimAtPath(cyl_path))
                    cyl.CreateRadiusAttr().Set(float(self.cfg.drum_radius))
                    cyl.CreateHeightAttr().Set(float(self.cfg.drum_height))


                # TranslateOp를 1회 생성하고 캐싱
                prim = stage.GetPrimAtPath(xform_path)
                xf = UsdGeom.Xformable(prim)    # prim이 transform 연산을 가질 수 있도록 감싸는 wrapper
                xf.ClearXformOpOrder()  # Clear 후 TranslateOp 추가
                t_op = xf.AddTranslateOp()
                t_op.Set(Gf.Vec3f(float(inst_pos[j][0]), float(inst_pos[j][1]), float(inst_pos[j][2]) + self.cfg.robot_waist_joint_offset_z))
                translate_ops_env.append(t_op)

                # 색 Primvar 캐싱
                cyl_prim = stage.GetPrimAtPath(cyl_path)
                gprim = UsdGeom.Gprim(cyl_prim)
                pv = gprim.GetDisplayColorPrimvar()
                if not pv:
                    pv = gprim.CreateDisplayColorPrimvar()
                pv.Set([self._drum_base_color])
                color_ops_env.append(pv)

            self._drum_translate_ops.append(translate_ops_env)
            self._drum_color_ops.append(color_ops_env)

    def _update_drums_pos(self, env_ids, inst_pos):

        i = 0

        for eid in env_ids:
            eid = int(eid)
            for j in range(len(self.inst_names)):
                p = inst_pos[i, j]
                self._drum_translate_ops[eid][j].Set(
                    Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]) + self.cfg.robot_waist_joint_offset_z)
                )
            
            i = i + 1

    def _update_drums_color(self, env_ids, inst_color):
        left_inst_color = inst_color[:,0]
        right_inst_color = inst_color[:,1]
        
        i = 0

        for eid in env_ids:
            eid = int(eid)
            for j in range(len(self.inst_names)):
                self._drum_color_ops[eid][j].Set([self._drum_base_color])

            if left_inst_color[i] == right_inst_color[i]:
                self._drum_color_ops[eid][left_inst_color[i]].Set([self._drum_both_target_color])
            else:
                self._drum_color_ops[eid][left_inst_color[i]].Set([self.color_L])
                self._drum_color_ops[eid][right_inst_color[i]].Set([self.color_R])

            i = i + 1

@torch.jit.script
def compute_reward_terms(
    left_tip_pos: torch.Tensor,       # (N,3) [m]
    right_tip_pos: torch.Tensor,      # (N,3) [m]
    left_tip_vel: torch.Tensor,       # (N,3) [m/s]
    right_tip_vel: torch.Tensor,      # (N,3) [m/s]
    left_target: torch.Tensor,        # (N,3) [m]
    right_target: torch.Tensor,       # (N,3) [m]
    impact_armed_L: torch.Tensor,     # (N,1)
    impact_armed_R: torch.Tensor,     # (N,1)
    action: torch.Tensor,             # (N, 9) [-1,1]
    joint_vel: torch.Tensor,          # (N, num_joints) [rad/s]
    robot_pos: torch.Tensor,          # (N, 9) [rad]
    joint_low: torch.Tensor,          # (1, 9) [rad]
    joint_high: torch.Tensor,         # (1, 9) [rad]
    exp_k_xy: float,
    exp_k_z: float,
    limit_margin: float,
):
    # -------------------------
    # XY distance
    # -------------------------
    left_xy = left_target[:, :2] - left_tip_pos[:, :2]     # (N,2)
    right_xy = right_target[:, :2] - right_tip_pos[:, :2]  # (N,2)

    left_xy_dist = torch.norm(left_xy, dim=-1)              # (N,)
    right_xy_dist = torch.norm(right_xy, dim=-1)            # (N,)
    xy_dist_sum = left_xy_dist + right_xy_dist              # (N,)

    # -------------------------
    # Z error
    # -------------------------
    left_z_err = torch.abs(left_target[:, 2] - left_tip_pos[:, 2])     # (N,)
    right_z_err = torch.abs(right_target[:, 2] - right_tip_pos[:, 2])  # (N,)
    z_err_sum = left_z_err + right_z_err                                 # (N,)

    # -------------------------
    # Shaping
    # -------------------------
    xy_term = -xy_dist_sum
    z_term = -z_err_sum

    exp_xy_term = torch.exp(-exp_k_xy * xy_dist_sum)
    exp_z_term = torch.exp(-exp_k_z * z_err_sum)

    # -------------------------
    # velocity shaping
    # -------------------------
    near_xy = (left_xy_dist < 0.15).to(torch.float32)
    near_z = (left_z_err < 0.15).to(torch.float32)
    left_downward = torch.clamp(-left_tip_vel[:,2], min=0.0) * near_xy * near_z * (1.0-impact_armed_L)
    left_upward = torch.clamp(left_tip_vel[:,2], min=0.0) * near_xy * near_z * impact_armed_L

    near_xy = (right_xy_dist < 0.15).to(torch.float32)
    near_z = (right_z_err < 0.15).to(torch.float32)
    right_downward = torch.clamp(-right_tip_vel[:,2], min=0.0) * near_xy * near_z * (1.0-impact_armed_R)
    right_upward = torch.clamp(right_tip_vel[:,2], min=0.0) * near_xy * near_z * impact_armed_R

    downward_term = left_downward + right_downward
    upward_term = left_upward + right_upward

    # -------------------------
    # Smoothness / energy
    # -------------------------
    action_l2 = torch.sum(action * action, dim=-1)          # (N,)
    joint_vel_l2 = torch.sum(joint_vel * joint_vel, dim=-1) # (N,)

    # -------------------------
    # Joint limit barrier
    # -------------------------
    low_v = torch.clamp((joint_low + limit_margin) - robot_pos, min=0.0)
    high_v = torch.clamp(robot_pos - (joint_high - limit_margin), min=0.0)
    limit_pen = torch.sum(low_v * low_v + high_v * high_v, dim=-1)

    return (
        xy_term,
        z_term,
        exp_xy_term,
        exp_z_term,
        downward_term,
        upward_term,
        action_l2,
        joint_vel_l2,
        limit_pen,
        left_xy_dist,
        right_xy_dist,
        left_z_err,
        right_z_err,
    )

@torch.jit.script
def compute_rewards(
    xy_term: torch.Tensor,
    z_term: torch.Tensor,
    exp_xy_term: torch.Tensor,
    exp_z_term: torch.Tensor,
    downward_term: torch.Tensor,
    upward_term: torch.Tensor,
    action_l2: torch.Tensor,
    joint_vel_l2: torch.Tensor,
    limit_pen: torch.Tensor,
    w_xy: float,
    w_z: float,
    w_exp_xy: float,
    w_exp_z: float,
    w_down: float,
    w_up: float,
    w_action: float,
    w_joint_vel: float,
    w_limit: float,
):
    reward = (
        w_xy * xy_term
        + w_z * z_term
        + w_exp_xy * exp_xy_term
        + w_exp_z * exp_z_term
        + w_down * downward_term
        + w_up * upward_term
        - w_action * action_l2
        - w_joint_vel * joint_vel_l2
        - w_limit * limit_pen
    )
    return reward