
"""
환경 클래스

path: drum_robot/tasks/drumrobot/drumrobot_env.py
"""

from __future__ import annotations

import torch
from typing import Sequence
import math
from isaaclab.utils import math as math_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane

from .rrdr_cfg import DrumRobotEnvCfg
from drum_robot.utils.logger import EnvLogger, LoggerCfg
from .components.resetters import RdsGeneratorCfg, RdsGenerator, PosGeneratorCfg, PosGenerator

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
    # 실행 순서 
    # 초기화: _setup_scene -> __init__
    # 루프: _get_observations -> _pre_physics_step -> _apply_action (*decimation) -> _get_dones -> _get_rewards
    # ============================================================

    def __init__(self, cfg: DrumRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._setup_rl_spaces()    # Space 정의
        self._init_default_values()
        self._load_config()  
        self._alloc_buffers()   # 버퍼 할당
        self._init_obs_norm_stats()  # 관측값 정규화를 위한 변수 초기화

        # 로그
        self.logger = EnvLogger(self.num_envs, self.device, LoggerCfg(interval=2000, sample_env_id=0))
        
        # RDS Generator
        self.rds_generator = RdsGenerator(
            self.num_envs,
            self.device,
            RdsGeneratorCfg(
                episode_length_s=self.cfg.episode_length_s,
                dt=self.dt,
                num_drum=self.num_drum,
                slow_factor=1.5,
                start_offset_steps=20,
                hit_window_step=self.cfg.hit_window_step,
            ),
        )

        self.pos_generator = PosGenerator(
            self.device,
            PosGeneratorCfg(
                num_ctrl_joint=len(self.ctrl_joint_names),
                height_above_drum=0.1,
                joint_noise_scale= 5*math.pi/180
            ),
            ctrl_joint_names=self.ctrl_joint_names,
            instruments=self.cfg.instruments,
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg()) # add ground plane
        self.scene.articulations["robot"] = self.robot  # add articulation to scene
        self.scene.clone_environments(copy_from_source=False)   # clone and replicate

        if self.cfg.enable_visualization:
            self._init_visualization()  # 시각화
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))  # 광원 추가
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        # 로봇의 관절 위치
        usd_pos = self.robot.data.joint_pos[:, self.ctrl_joint_ids]        # (num_envs, 9)
        joint_pos = self._convert_usd_to_robot(usd_pos)

        # 로봇의 관절 속도
        usd_vel = self.robot.data.joint_vel[:, self.ctrl_joint_ids]        # (num_envs, 9)
        joint_vel = self._convert_usd_to_robot(usd_vel)

        # 위치
        tip_pos = self.tip_pos
        inst_pos = self.inst_pos

        # 악보
        rds_window = self.get_rds_window(rds=self.rds, step=self.steps)    # 현재 위치 다음 스텝부터 미래 L개 스텝
        self.rds_window = rds_window

        # 로봇 상태
        hit_armed = self.hit_armed

        obs = self._normalize_and_pack_obs(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            tip_pos=tip_pos,
            inst_pos=inst_pos,
            rds_onehot=rds_window,
            hit_armed=hit_armed.float(),
            )
        
        if torch.isnan(obs).any():
            raise RuntimeError("NaN obs")
        
        return {"policy": obs}

    def _pre_physics_step(self, actions: torch.Tensor):
        # RL 에이전트로부터 받은 actions 저장
        actions = actions.to(torch.float32)

        # action 범위를 [-1, 1]로 강제
        actions = torch.clamp(actions, -1.0, 1.0)

        # 로봇 위치 가져오기
        usd_q = self.robot.data.joint_pos[:, self.ctrl_joint_ids]
        robot_q = self._convert_usd_to_robot(usd_q)

        # 목표 위치 = 현재 위치 + actions
        target_q_d = actions * self.cfg.action_scale
        robot_q_next = robot_q + target_q_d * self.dt
        robot_q_next = torch.max(torch.min(robot_q_next, self.joint_high), self.joint_low)  # joint_limit 내로 clip
        usd_q_next = self._convert_robot_to_usd(robot_q_next)

        self.actions = actions.clone()
        self.target_joint_pos = usd_q_next

        if torch.isnan(actions).any():
            raise RuntimeError("NaN actions")

    def _apply_action(self):
        self.robot.set_joint_position_target(self.target_joint_pos, joint_ids=self.ctrl_joint_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 팁 위치
        tip_pos = self._compute_tip_position()          # (num_envs, 2, 3)
        
        # 팁 속도
        prev_tip_pos = self.tip_pos
        prev_tip_vel = self.tip_vel
        alpha = 0.9
        tip_vel = self._compute_tip_velocity(tip_pos, prev_tip_pos, prev_tip_vel, alpha)

        # 팁 드럼 거리 계산
        inst_pos = self.inst_pos
        dist_xy_sq, diff_z = self._compute_tip_drum_dist_sq(tip_pos, inst_pos)

        # 접촉 확인
        contact_mask = self._check_contact_drum(dist_xy_sq, diff_z)

        # 타격 판정
        hit_armed = self.hit_armed
        hit_per_arm = self._check_hitting(
            contact_mask=contact_mask,
            hit_armed=hit_armed,
            tip_vel=tip_vel,
            diff_z=diff_z,
        )   # (N, 2, M)
        
        # 양팔 중 하나라도 해당 drum을 strike하면 hit
        hit_mask = torch.any(hit_per_arm, dim=1)   # (N, M)

        # 타격 시간 기록
        steps = self.steps
        self.rds_visit[self.env_arange, steps, :] = hit_mask    # 타격한 시간에 방문 처리

        # 잘못친 타격 판정
        rds = self.rds
        wrong_hit = self._detect_wrong_hits(hit_mask, rds, steps)

        # 윈도우 끝났을 때 타격 성공 확인
        success, missed_target, time_error = self._finalize_target_outcomes(
            rds=rds,
            rds_visit=self.rds_visit,
            steps=steps,
        )

        # re-arm 확인
        next_hit_armed = self._check_rearm(hit_armed, hit_per_arm, contact_mask, diff_z)
        
        self.tip_pos = tip_pos
        self.tip_vel = tip_vel
        self.prev_tip_pos = prev_tip_pos
        self.hit_armed = next_hit_armed

        # reward 로 넘기는 결과값
        self.success = success
        self.wrong_hit = wrong_hit
        self.missed_target = missed_target
        self.time_error = time_error
        self.hit_armed_for_reward = hit_armed

        # 팁 표시, 드럼 색상 변경
        if self.cfg.enable_visualization:
            self._translate_tip(self.tip_pos)
            self._update_drum_color(self.rds_window, hit_mask)

        # episode time out
        self.steps = self.steps + 1
        time_out = self.steps >= self.episode_length - 1
        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        usd_pos = self.robot.data.joint_pos[:, self.ctrl_joint_ids]
        usd_vel = self.robot.data.joint_vel[:, self.ctrl_joint_ids]
        robot_pos = self._convert_usd_to_robot(usd_pos)  # (N, 9)
        robot_vel = self._convert_usd_to_robot(usd_vel)

        (
            goal_success, goal_wrong, goal_missed, time_error_term,
            proximity_term, progress_term,
            strike_phase_term, rearm_phase_term,
            action_l2, joint_vel_l2, limit_pen, tip_limit_pen, under_drum_pen,
        ) = compute_reward_terms(
            success=self.success,
            wrong_hit=self.wrong_hit,
            missed_target=self.missed_target,
            time_error=self.time_error,

            left_tip_pos=self.tip_pos[:, 0, :],
            right_tip_pos=self.tip_pos[:, 1, :],
            prev_left_tip_pos=self.prev_tip_pos[:, 0, :],
            prev_right_tip_pos=self.prev_tip_pos[:, 1, :],
            inst_pos=self.inst_pos,
            rds_window=self.rds_window,

            tip_vel=self.tip_vel,
            hit_armed=self.hit_armed_for_reward,

            joint_vel=robot_vel,
            action=self.actions,
            robot_pos=robot_pos,
            joint_low=self.joint_low,
            joint_high=self.joint_high,

            idle_pos=self.idle_pos,

            k_time_error=self.cfg.k_time_error,
            k_idx=self.cfg.k_idx,
            limit_margin=self.cfg.limit_margin,

            x_limit=self.cfg.x_limit,
            y_limit_l=self.cfg.y_limit_l,
            y_limit_h=self.cfg.y_limit_h,
            z_limit=self.cfg.z_limit,
            drum_xy_margin=self.cfg.drum_xy_margin,
            drum_z_margin=self.cfg.drum_z_margin,
        )

        reward = compute_rewards(
            success_term=goal_success,
            wrong_term=goal_wrong,
            miss_term=goal_missed,
            time_error_term=time_error_term,
            
            proximity_term=proximity_term,
            progress_term=progress_term,

            strike_phase_term=strike_phase_term,
            rearm_phase_term=rearm_phase_term,

            action_l2=action_l2,
            joint_vel_l2=joint_vel_l2,
            limit_pen=limit_pen,
            tip_limit_pen=tip_limit_pen,
            under_drum_pen=under_drum_pen,

            w_success=self.cfg.w_success,
            w_wrong=self.cfg.w_wrong,
            w_miss=self.cfg.w_miss,
            w_time_error=self.cfg.w_time_error,

            w_progress=self.cfg.w_progress,
            w_proximity=self.cfg.w_proximity,

            w_strike_phase=self.cfg.w_strike_phase,
            w_rearm_phase=self.cfg.w_rearm_phase,
            
            w_action=self.cfg.w_action,
            w_joint_vel=self.cfg.w_joint_vel,
            w_limit=self.cfg.w_limit,
            w_tip_limit=self.cfg.w_tip_limit,
            w_under_drum=self.cfg.w_under_drum,
        )

        # 로그 출력
        terms = {
            "reward": reward,
            "proximity_term": proximity_term,
            "progress(x100)": progress_term * 100,
            "strike_phase_term": strike_phase_term,
            "rearm_phase_term": rearm_phase_term,
            "action_l2": action_l2,
            "joint_vel_l2": joint_vel_l2,
            "limit_pen": limit_pen,
            "tip_limit_pen": tip_limit_pen,
            "under_drum_pen": under_drum_pen,
        }
        self.logger.add(terms)

        num_hit = goal_success + goal_wrong + goal_missed
        num_hit_inst = self.success.float() + self.wrong_hit.float() + self.missed_target.float()
        
        p_terms = {
            "success_rate": torch.stack([goal_success, num_hit], dim=-1),
            "wrong_rate": torch.stack([goal_wrong, num_hit], dim=-1),
            "miss_rate": torch.stack([goal_missed, num_hit], dim=-1),

            "snare_success_rate": torch.stack([self.success[:, 0], num_hit_inst[:, 0]], dim=-1),
            # "snare_wrong_rate": torch.stack([self.wrong_hit[:, 0], num_hit_inst[:, 0]], dim=-1),
            # "snare_miss_rate": torch.stack([self.missed_target[:, 0], num_hit_inst[:, 0]], dim=-1),

            "floor_success_rate": torch.stack([self.success[:, 1], num_hit_inst[:, 1]], dim=-1),
            # "floor_wrong_rate": torch.stack([self.wrong_hit[:, 1], num_hit_inst[:, 1]], dim=-1),
            # "floor_miss_rate": torch.stack([self.missed_target[:, 1], num_hit_inst[:, 1]], dim=-1),

            "mid_success_rate": torch.stack([self.success[:, 2], num_hit_inst[:, 2]], dim=-1),
            # "mid_wrong_rate": torch.stack([self.wrong_hit[:, 2], num_hit_inst[:, 2]], dim=-1),
            # "mid_miss_rate": torch.stack([self.missed_target[:, 2], num_hit_inst[:, 2]], dim=-1),

            "high_success_rate": torch.stack([self.success[:, 3], num_hit_inst[:, 3]], dim=-1),
            # "high_wrong_rate": torch.stack([self.wrong_hit[:, 3], num_hit_inst[:, 3]], dim=-1),
            # "high_miss_rate": torch.stack([self.missed_target[:, 3], num_hit_inst[:, 3]], dim=-1),

            "hihat_success_rate": torch.stack([self.success[:, 4], num_hit_inst[:, 4]], dim=-1),
            # "hihat_wrong_rate": torch.stack([self.wrong_hit[:, 4], num_hit_inst[:, 4]], dim=-1),
            # "hihat_miss_rate": torch.stack([self.missed_target[:, 4], num_hit_inst[:, 4]], dim=-1),

            "ride_success_rate": torch.stack([self.success[:, 5], num_hit_inst[:, 5]], dim=-1),
            # "ride_wrong_rate": torch.stack([self.wrong_hit[:, 5], num_hit_inst[:, 5]], dim=-1),
            # "ride_miss_rate": torch.stack([self.missed_target[:, 5], num_hit_inst[:, 5]], dim=-1),

            "crash1_success_rate": torch.stack([self.success[:, 6], num_hit_inst[:, 6]], dim=-1),
            "crash2_success_rate": torch.stack([self.success[:, 7], num_hit_inst[:, 7]], dim=-1),
            # "crash_wrong_rate": torch.stack([self.wrong_hit[:, 7], num_hit_inst[:, 7]], dim=-1),
            # "crash_miss_rate": torch.stack([self.missed_target[:, 7], num_hit_inst[:, 7]], dim=-1),
        }
        self.logger.add_probability(p_terms)
        self.logger.maybe_flush()

        if torch.isnan(reward).any():
            raise RuntimeError("NaN reward")

        return reward
    
    def _reset_idx(self, env_ids: Sequence[int]| None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES   # type: ignore[arg-type]
        # 부모 클래스의 리셋(버퍼 초기화 등) 호출
        super()._reset_idx(env_ids)             # type: ignore[arg-type]

        # 드럼 위치 리셋 (perturbation)
        inst_pos = self.basic_inst_pos.unsqueeze(0).repeat(len(env_ids), 1, 1)  # type: ignore[arg-type]
        inst_pos = inst_pos + self.cfg.inst_noise_scale * torch.randn_like(inst_pos)
        inst_pos[:, :, 2] = inst_pos[:, :, 2] + self.cfg.robot_waist_joint_offset_z
        self.inst_pos[env_ids] = inst_pos

        if self.cfg.enable_visualization:
            self._translate_drum(self.inst_pos)

        # RDS 리셋
        robotic_drum_score = self.rds_generator.reset_target(env_ids)
        self.rds[env_ids] = robotic_drum_score
        self.rds_visit[env_ids] = False

        # 로봇 자세 리셋
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]  # 기본 자세 가져오기
        default_joint_vel = self.robot.data.default_joint_vel[env_ids]

        joint_pos = self._get_init_joint_pos(env_ids, default_joint_pos)
        joint_vel = torch.zeros_like(default_joint_vel)
        
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 텐서 변수들 리셋
        self._reset_tensors(env_ids)
        
    # ============================================================
    # [Custom Functions]
    # Internal utility methods (NOT called by RL engine directly)
    # ============================================================

    """ init """
    def _setup_rl_spaces(self):
        # 반드시 single-env shape로 정의
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(304,), dtype=np.float32
        )

        # print("[DEBUG] action_space:", self.action_space)
        # print("[DEBUG] observation_space:", self.observation_space)
        # print("[DEBUG] cfg.action_dim:", getattr(self.cfg, "action_dim", None))

    def _init_default_values(self):
        # env_ids
        self.env_arange = torch.arange(self.num_envs, device=self.device)

        # body/joint id resolve
        self._bind_joint_ids()
        self._bind_body_ids()

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

    def _bind_body_ids(self):
        # print("[DEBUG] body_names: ", self.robot.data.body_names)

        # 양손 스틱 링크의 인덱스
        self.left_stick_idx = self._get_body_idx("left_wrist")  # 10
        self.right_stick_idx = self._get_body_idx("right_wrist")  # 11

        # 양손 하완 링크의 인덱스
        self.left_arm_idx = self._get_body_idx("left_elbow")  # 07
        self.right_arm_idx = self._get_body_idx("right_elbow")  # 08

    def _get_body_idx(self, body_name) -> int:
        ids, _ = self.robot.find_bodies(body_name) # ([id], ['body name'])
        
        if len(ids) == 0:
            raise RuntimeError("wrist body not found. Check USD body names.")
        
        idx = ids[0] # int 값만 저장

        return idx

    def _load_config(self):
        # dt
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # episode length (step)
        self.episode_length = int(self.cfg.episode_length_s / self.dt)

        # offset wrist link to tip
        L_off = torch.tensor(self.cfg.tip_offset_left, device=self.device, dtype=torch.float32)  # (3,)
        R_off = torch.tensor(self.cfg.tip_offset_right, device=self.device, dtype=torch.float32)

        self.tip_offset_L = L_off.unsqueeze(0).expand(self.num_envs, 3)
        self.tip_offset_R = R_off.unsqueeze(0).expand(self.num_envs, 3)

        self._build_joint_tensors()
        self._build_drum_tensors()

        self.idle_pos = torch.tensor(self.cfg.idle_pos, device=self.device).unsqueeze(0)    # (1, 4)
        self.idle_pos[:, 2] = self.idle_pos[:, 2] + self.cfg.robot_waist_joint_offset_z

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

    def _build_drum_tensors(self):
        self.inst_names = list(self.cfg.instruments.keys())
        self.basic_inst_pos = torch.tensor(
            list(self.cfg.instruments.values()),
            device=self.device,
            dtype=torch.float32
        )

        self.num_drum = len(self.basic_inst_pos[:,0])

    def _alloc_buffers(self):
        N = self.num_envs
        T = self.episode_length
        M = self.num_drum
        L = self.cfg.rds_observation_length

        # 로봇의 tip 위치를 저장할 텐서
        self.tip_pos = torch.zeros((N, 2, 3), device=self.device)
        self.prev_tip_pos = torch.zeros((N, 2, 3), device=self.device)

        # 로봇의 tip 속도 저장할 텐서
        self.tip_vel = torch.zeros((N, 2, 3), device=self.device)

        # 각 env/각 에피소드에서 악기 위치
        inst_pos = self.basic_inst_pos.unsqueeze(0).repeat(N, 1, 1)    # (M, 3) -차원 추가-> (1, M, 3) -복제-> (N, M, 3)
        inst_pos[:, :, 2] = inst_pos[:, :, 2] + self.cfg.robot_waist_joint_offset_z
        self.inst_pos = inst_pos

        # 타격 상태 버퍼
        self.hit_armed = torch.ones((N, 2, M), device=self.device, dtype=torch.bool)
        self.hit_armed_for_reward = torch.ones((N, 2, M), device=self.device, dtype=torch.bool) # 보상 계산용

        # 목표 악보을 저장할 텐서
        self.rds = torch.zeros((N, T, M), device=self.device, dtype=torch.int64)
        self.rds_window = torch.zeros((N, L, M), device=self.device, dtype=torch.int64)
        self.rds_visit = torch.zeros((N, T, M), device=self.device, dtype=torch.bool)

        # step
        self.steps = torch.zeros((N,), device=self.device, dtype=torch.int64)

    def _init_obs_norm_stats(self):
        # joint
        self.joint_center = 0.5 * (self.joint_low + self.joint_high)
        self.joint_half_range = 0.5 * (self.joint_high - self.joint_low) + 1e-6

        self.joint_vel_scale = torch.tensor(self.cfg.joint_vel_scale, device=self.device, dtype=torch.float32)

        # target
        inst_min = self.basic_inst_pos.min(dim=0).values
        inst_max = self.basic_inst_pos.max(dim=0).values

        center = 0.5 * (inst_min + inst_max)
        half_range = 0.5 * (inst_max - inst_min) + 1e-6
        
        self.task_center = torch.cat([center], dim=0).unsqueeze(0).unsqueeze(1)          # (1, 1, 3)
        self.task_half_range = torch.cat([half_range], dim=0).unsqueeze(0).unsqueeze(1)  # (1, 1, 3)       

    """ util """
    def _convert_usd_to_robot(self, usd_data):
        return self.dir_tensor * usd_data

    def _convert_robot_to_usd(self, robot_data):
        return self.dir_tensor * robot_data

    """ func (_get_observations) """
    def get_rds_window(self, rds, step):
        # rds: (N, T, M)
        # step: (N,)
        L = self.cfg.rds_observation_length
        M = self.num_drum
        T = self.episode_length

        # (N, L) step index 만들기
        offset = torch.arange(L, device=self.device).unsqueeze(0)    # (1, L), [0 L-1]
        idx = step.unsqueeze(1) + offset                                # (N, L)

        # 범위 체크
        valid = self._is_valid_index(idx, T)

        # 안전하게 index clamp
        idx_clamped = idx.clamp(0, T - 1)

        # gather로 (N, L, M) 만들기
        out = rds.gather(1, idx_clamped.unsqueeze(-1).expand(-1, -1, M))

        # invalid 위치는 0으로
        out[~valid] = 0

        return out
    
    def _is_valid_index(self, idx, max_idx):
        return (idx >= 0) & (idx < max_idx)

    def _normalize_and_pack_obs(self,
            joint_pos,
            joint_vel,
            tip_pos,
            inst_pos,
            rds_onehot,
            hit_armed,
            ) -> torch.Tensor:
        # joint_pos, joint_vel: (N, 9)
        # tip_pos: (N, 2, 3)
        # inst_pos: (N, M, 3)
        # rds_onehot: (N, L, M)
        # hit_armed: (N, 2, M)

        # normalize
        joint_pos_n = (joint_pos - self.joint_center) / self.joint_half_range
        joint_pos_n = torch.clamp(joint_pos_n, -1.5, 1.5)

        joint_vel_n = joint_vel / self.joint_vel_scale
        joint_vel_n = torch.clamp(joint_vel_n, -10.0, 10.0)

        tip_pos_n = (tip_pos - self.task_center) / self.task_half_range
        tip_pos_n = torch.clamp(tip_pos_n, -1.5, 1.5)

        inst_pos_n = (inst_pos - self.task_center) / self.task_half_range
        inst_pos_n = torch.clamp(inst_pos_n, -1.5, 1.5)
        
        obs = torch.cat(
            [
                joint_pos_n,
                joint_vel_n,
                tip_pos_n.reshape(self.num_envs, 6),
                inst_pos_n.reshape(self.num_envs, 24),
                rds_onehot.reshape(self.num_envs, 240),
                hit_armed.reshape(self.num_envs, 16),
            ],
            dim=-1
        )
        
        return obs

    """ func (_get_dones) """
    def _compute_tip_position(self):
        # 스틱 링크의 월드 좌표계 위치 가져오기
        all_body_pos = self.robot.data.body_pos_w      # (num_envs, num_bodies, 3)
        all_body_quat = self.robot.data.body_quat_w    # (num_envs, num_bodies, 4)  (w,x,y,z)인 경우가 많음

        L_wrist_pos = all_body_pos[:, self.left_stick_idx]      # (num_envs, 3)
        R_wrist_pos = all_body_pos[:, self.right_stick_idx]
        L_quat = all_body_quat[:, self.left_stick_idx]
        R_quat = all_body_quat[:, self.right_stick_idx]

        # 팁 위치 구하기
        L_tip_w = L_wrist_pos + math_utils.quat_apply(L_quat, self.tip_offset_L)
        R_tip_w = R_wrist_pos + math_utils.quat_apply(R_quat, self.tip_offset_R)

        # 월드 기준 -> env 기준
        L_tip = L_tip_w - self.scene.env_origins
        R_tip = R_tip_w - self.scene.env_origins

        tip_pos = torch.stack([L_tip, R_tip], dim=1)   # (num_envs, 2, 3)

        return tip_pos
    
    def _compute_tip_velocity(self, tip_pos, prev_tip_pos, prev_tip_vel, alpha):
        tip_vel = (tip_pos - prev_tip_pos) / self.dt

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

        N = self.num_envs
        T = self.episode_length
        M = self.num_drum
        W = self.cfg.hit_window_step

        window_target_union = torch.zeros((N, M), device=self.device, dtype=torch.bool)

        for offset in range(-W, W + 1):
            cand_steps = steps + offset
            valid = self._is_valid_index(cand_steps, T)
            cand_steps_clamped = cand_steps.clamp(0, T - 1)

            target_mask = rds[self.env_arange, cand_steps_clamped, :] > 0.5     # (N, M)
            target_mask &= valid.unsqueeze(-1)

            window_target_union |= target_mask

        wrong_hit = hit_mask & (~window_target_union)

        return wrong_hit

    def _finalize_target_outcomes(self, rds, rds_visit, steps):
        # rds, rds_visit: (N, T, M)
        # steps: (N,)

        N = self.num_envs
        T = self.episode_length
        M = self.num_drum
        W = self.cfg.hit_window_step

        success = torch.zeros((N, M), device=self.device, dtype=torch.bool)
        time_error = torch.full((N, M), -1, device=self.device, dtype=torch.int64)

        # 윈도우 왼쪽 끝 스텝
        window_end_step = steps - W
        vaild = self._is_valid_index(window_end_step, T)
        window_end_step = window_end_step.clamp(0, T - 1)

        target_mask = (rds[self.env_arange, window_end_step, :] > 0.5)
        target_mask &= vaild.unsqueeze(-1)     # (N, M)

        offsets = self._get_hit_window_offsets(W)
        for offset in offsets:
            cand_steps = window_end_step + offset
            vaild = self._is_valid_index(cand_steps, T)
            cand_steps_clamped = cand_steps.clamp(0, T - 1)

            hit_mask = rds_visit[self.env_arange, cand_steps_clamped, :] > 0.5  # (N, M)

            match_mask = hit_mask & vaild.unsqueeze(-1) & target_mask & (~success)
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
        rearm_mask = (~contact_mask) & (diff_z > self.cfg.rearm_height)

        next_hit_armed = hit_armed.clone()

        # hit 나면 disarm
        next_hit_armed[hit_per_arm] = False

        # 충분히 벗어나고 올라가면 rearm
        next_hit_armed[rearm_mask] = True

        return next_hit_armed

    """ func (_reset_idx) """
    def _get_init_joint_pos(self, env_ids, default_joint_pos):
        joint_pos = default_joint_pos.clone()

        init_pos = self.pos_generator.reset_init_pos(env_ids)
        usd_init_pos = self._convert_robot_to_usd(init_pos)
        joint_pos[:,self.ctrl_joint_ids] = usd_init_pos

        return joint_pos

    def _reset_tensors(self, env_ids):
        # 팁 위치/속도 리셋
        tip_pos = self._compute_tip_position()
        self.tip_pos[env_ids] = tip_pos[env_ids]
        self.prev_tip_pos[env_ids] = tip_pos[env_ids]
        self.tip_vel[env_ids] = 0.0

        # 타격 상태 버퍼 리셋
        self.hit_armed[env_ids] = True  # 초기 위치는 악기 높이 차이 상관 없이 타격 가능

        # 스텝 리셋
        self.steps[env_ids] = 0 # torch.randint(0, 9, (len(env_ids),), device=self.device)

    # ============================================================
    # [Debug / Visualization]
    # Not used for training logic
    # ============================================================

    def _init_visualization(self):
        self._init_tip()
        self._init_drum()

    def _create_sphere(self, node_name, radius, color):
        # 고속 업데이트를 위한 오퍼레이터 캐시
        translate_ops = []
        color_ops = []

        # sphere 설정
        sphere_cfg = sim_utils.SphereCfg(
            radius=radius,
        )

        # 현재 IsaacSim의 USD Stage 접근
        stage = omni.usd.get_context().get_stage()

        for i in range(self.num_envs):

            # USD Stage 위에 Prim을 생성 (이미 존재하면 타입을 유지한 채 반환)
            viz_root = f"/World/envs/env_{i}/_viz"
            stage.DefinePrim(viz_root, "Xform")     # Xform = Transform 노드

            xform_path = f"{viz_root}/{node_name}"
            stage.DefinePrim(xform_path, "Xform")

            sphere_path = f"{xform_path}/sphere"
            # IsValid 체크 후 sphere prim 생성
            if not stage.GetPrimAtPath(sphere_path).IsValid():
                sphere_cfg.func(sphere_path, sphere_cfg)

            # TranslateOp를 1회 생성하고 캐싱
            prim = stage.GetPrimAtPath(xform_path)
            xf = UsdGeom.Xformable(prim)    # prim이 transform 연산을 가질 수 있도록 감싸는 wrapper
            ops = xf.GetOrderedXformOps()   # 현재 들어있는 transform 연산 목록 가져오기
            if len(ops) > 0 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:    # 첫 번째 연산이 TranslateOp
                t_op = ops[0]
            else:
                # Clear 후 TranslateOp 추가
                xf.ClearXformOpOrder()
                t_op = xf.AddTranslateOp()

            translate_ops.append(t_op)

            # 색상 설정
            sphere_prim = stage.GetPrimAtPath(sphere_path)
            gprim = UsdGeom.Gprim(sphere_prim)
            pv = gprim.GetDisplayColorPrimvar()
            if not pv:
                pv = gprim.CreateDisplayColorPrimvar()
            pv.Set([color])

            # 색 Primvar 캐싱
            color_ops.append(pv)

        return translate_ops, color_ops

    def _create_cylinder(self, node_name, radius, height, color):
        # 고속 업데이트를 위한 오퍼레이터 캐시
        translate_ops = []
        color_ops = []

        # 현재 IsaacSim의 USD Stage 접근
        stage = omni.usd.get_context().get_stage()

        for i in range(self.num_envs):

            # USD Stage 위에 Prim을 생성 (이미 존재하면 타입을 유지한 채 반환)
            viz_root = f"/World/envs/env_{i}/_viz"
            stage.DefinePrim(viz_root, "Xform")     # Xform = Transform 노드

            xform_path = f"{viz_root}/{node_name}"
            stage.DefinePrim(xform_path, "Xform")

            cylinder_path = f"{xform_path}/cylinder"
            # IsValid 체크 후 cylinder prim 생성
            if not stage.GetPrimAtPath(cylinder_path).IsValid():
                stage.DefinePrim(cylinder_path, "Cylinder")
                cyl = UsdGeom.Cylinder(stage.GetPrimAtPath(cylinder_path))
                cyl.CreateRadiusAttr().Set(radius)
                cyl.CreateHeightAttr().Set(height)

            # TranslateOp를 1회 생성하고 캐싱
            prim = stage.GetPrimAtPath(xform_path)
            xf = UsdGeom.Xformable(prim)    # prim이 transform 연산을 가질 수 있도록 감싸는 wrapper
            ops = xf.GetOrderedXformOps()   # 현재 들어있는 transform 연산 목록 가져오기
            if len(ops) > 0 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate:    # 첫 번째 연산이 TranslateOp
                t_op = ops[0]
            else:
                # Clear 후 TranslateOp 추가
                xf.ClearXformOpOrder()
                t_op = xf.AddTranslateOp()

            translate_ops.append(t_op)

            # 색상 설정
            cylinder_prim = stage.GetPrimAtPath(cylinder_path)
            gprim = UsdGeom.Gprim(cylinder_prim)
            pv = gprim.GetDisplayColorPrimvar()
            if not pv:
                pv = gprim.CreateDisplayColorPrimvar()
            pv.Set([color])

            # 색 Primvar 캐싱
            color_ops.append(pv)

        return translate_ops, color_ops

    def _translate(self, t_op, pos):
        n = len(pos[:,0])

        for i in range(n):
            p = pos[i]
            t_op[i].Set(
                Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
            )

    def _color(self, c_op, env_ids, color):
        env_ids = env_ids.tolist()

        for id in env_ids:
            c_op[id].Set([color])

    def _init_tip(self):
        color_L = Gf.Vec3f(
            float(self.cfg.color_L[0]),
            float(self.cfg.color_L[1]),
            float(self.cfg.color_L[2])
        )
        t_op, c_op = self._create_sphere(
            node_name="TipMarkerL",
            radius=self.cfg.tip_marker_radius,
            color=color_L
        )
        self.tip_translate_ops_L = t_op
        self.tip_color_ops_L = c_op

        color_R = Gf.Vec3f(
            float(self.cfg.color_R[0]),
            float(self.cfg.color_R[1]),
            float(self.cfg.color_R[2])
        )
        t_op, c_op  = self._create_sphere(
            node_name="TipMarkerR",
            radius=self.cfg.tip_marker_radius,
            color=color_R
        )
        self.tip_translate_ops_R = t_op
        self.tip_color_ops_R = c_op

    def _translate_tip(self, tip_pos):
        self._translate(self.tip_translate_ops_L, tip_pos[:, 0, :])
        self._translate(self.tip_translate_ops_R, tip_pos[:, 1, :])

    def _init_drum(self):
        # 고속 업데이트를 위한 TranslateOp 캐시
        self._drum_translate_ops = []

        # 고속 업데이트를 위한 색 Primvar 캐시
        self._drum_color_ops = []

        inst_names = list(self.cfg.instruments.keys())
        self.base_color = Gf.Vec3f(0.5, 0.5, 0.5)

        for _, name in enumerate(inst_names):
            t_op, c_op = self._create_cylinder(
                node_name=name,
                radius=self.cfg.drum_radius,
                height=self.cfg.drum_height,
                color=self.base_color,
            )

            self._drum_translate_ops.append(t_op)
            self._drum_color_ops.append(c_op)

    def _translate_drum(self, inst_pos):
        for i in range(self.num_drum):
            p = inst_pos[:, i, :]
            self._translate(self._drum_translate_ops[i], p)

    def _update_drum_color(self, rds_window, hit_mask):
        _, L, M = rds_window.shape
        
        far_color  = Gf.Vec3f(1.0, 1.0, 0.0)   # 노랑 (Yellow)
        near_color = Gf.Vec3f(1.0, 0.0, 0.0)   # 빨강 (Red)

        for i in range(M):

            far_mask  = (rds_window[:, 3:L, i] > 0.5).any(dim=1)
            near_mask = (rds_window[:, 0:3, i] > 0.5).any(dim=1)

            # 우선순위: near > mid > far
            near_ids = torch.where(near_mask)[0]
            far_ids  = torch.where(~near_mask & far_mask)[0]
            none_ids = torch.where(~far_mask & ~near_mask)[0]

            self._color(self._drum_color_ops[i], none_ids, self.base_color)
            self._color(self._drum_color_ops[i], far_ids, far_color)
            self._color(self._drum_color_ops[i], near_ids, near_color)

            # hit_inst_mask = hit_mask[:, i]
            # hit_ids = torch.where(hit_inst_mask)[0]
            # self._color(self._drum_color_ops[i], hit_ids, near_color)

""" helper 함수 """
@torch.jit.script
def assignment_cost_for_targets(
        target_mask: torch.Tensor,  # (N, M)
        inst_pos: torch.Tensor,     # (N, M, 3)
        left_tip_pos: torch.Tensor, # (N, 3)
        right_tip_pos: torch.Tensor,# (N, 3)
        idle_pos: torch.Tensor,     # (1, 4)
        waist: torch.Tensor,        # (N,)
) -> torch.Tensor:
    device = target_mask.device
    N, M = target_mask.shape

    # XYZ 거리
    diff_l = left_tip_pos[:, None, :] - inst_pos    # (N, M, 3)
    diff_r = right_tip_pos[:, None, :] - inst_pos

    dist_l = torch.sum(diff_l * diff_l, dim=-1)     # (N, M)
    dist_r = torch.sum(diff_r * diff_r, dim=-1)
    
    # target index 추출 (최대 2개)
    masked_idx = torch.where(
        target_mask,
        torch.arange(M, device=device).expand(N, M),
        M - 1  # 타겟이 없는 경우 잘못된 타겟이 들어가는 걸 방지하고자 인덱스 중 최대값 사용, 타겟이 하나인 경우는 상관 없음
    )

    top2_idx = torch.topk(masked_idx, k=2, dim=1, largest=False).values  # (N, 2)

    target_count = target_mask.float().sum(dim=1)
    cost = torch.zeros(N, device=device)
    
    # # --------------------
    # # case 1: 타겟 0개
    # # --------------------
    # idle_mask = (target_count == 0)

    # ilde_diff_l = left_tip_pos - idle_pos[:, 0:3]   # (N, 3)
    # ilde_diff_r = right_tip_pos - idle_pos[:, 0:3]
    
    # ilde_dist_l = torch.sum(ilde_diff_l * ilde_diff_l, dim=-1)     # (N,)
    # ilde_dist_r = torch.sum(ilde_diff_r * ilde_diff_r, dim=-1)

    # ilde_diff_w = waist - idle_pos[0, 3]  # (N,)

    # idle_cost = 0.02 * ilde_dist_l + 0.02 * ilde_dist_r + 0.05 * torch.abs(ilde_diff_w)
    # cost = torch.where(idle_mask, idle_cost, cost)

    # --------------------
    # case 2: 타겟 1개
    # --------------------
    one_mask = (target_count == 1)

    idx0 = top2_idx[:, 0]

    d_l0 = dist_l.gather(1, idx0.unsqueeze(1)).squeeze(1)
    d_r0 = dist_r.gather(1, idx0.unsqueeze(1)).squeeze(1)

    one_cost = torch.minimum(d_l0, d_r0)
    cost = torch.where(one_mask, one_cost, cost)

    # --------------------
    # case 3: 타겟 2개
    # --------------------
    two_mask = (target_count >= 2)

    idx1 = top2_idx[:, 0]
    idx2 = top2_idx[:, 1]

    d_l1 = dist_l.gather(1, idx1.unsqueeze(1)).squeeze(1)
    d_l2 = dist_l.gather(1, idx2.unsqueeze(1)).squeeze(1)

    d_r1 = dist_r.gather(1, idx1.unsqueeze(1)).squeeze(1)
    d_r2 = dist_r.gather(1, idx2.unsqueeze(1)).squeeze(1)

    # assignment 2가지
    cost_case1 = d_l1 + d_r2
    cost_case2 = d_l2 + d_r1

    two_cost = torch.minimum(cost_case1, cost_case2)
    cost = torch.where(two_mask, two_cost, cost)

    return cost

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
    rds_window:torch.Tensor,        # (N, L, M)

    tip_vel:torch.Tensor,           # (N, 2, 3)
    hit_armed:torch.Tensor,         # (N, 2, M)

    joint_vel: torch.Tensor,        # (N, num_joints) [rad/s]
    action: torch.Tensor,           # (N, 9) [-1,1]
    robot_pos: torch.Tensor,        # (N, 9) [rad]
    joint_low: torch.Tensor,        # (1, 9) [rad]
    joint_high: torch.Tensor,       # (1, 9) [rad]

    idle_pos: torch.Tensor,         # (1, 4) [m, rad]

    k_time_error: float,
    k_idx: float,
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
    goal_success = success.float().sum(dim=-1)        # (N,)
    goal_wrong = wrong_hit.float().sum(dim=-1)        # (N,)
    goal_missed = missed_target.float().sum(dim=-1)   # (N,)

    valid_mask = time_error >= 0
    time_error_term = torch.exp(-k_time_error * time_error)
    time_error_term[~valid_mask] = 0
    time_error_term = time_error_term.sum(dim=-1)

    # -------------------------------------------------
    # proximity terms: nearest imminent target only
    # -------------------------------------------------
    N, L, M = rds_window.shape
    device = rds_window.device

    has_target_each_step = (rds_window > 0.5).any(dim=2)  # (N, L)

    # target이 있는 첫 step 찾기
    first_idx = torch.argmax(has_target_each_step.to(torch.int64), dim=1)  # (N,)

    idx = first_idx.clamp(0, L - 1)
    nearest_mask = rds_window[torch.arange(N, device=device), idx, :] > 0.5  # (N, M)

    # argmax는 전부 False여도 0을 반환하므로 (최대값이 0) window 안에 target이 없으면 zero mask
    any_target = has_target_each_step.any(dim=1)  # (N,)
    nearest_target_mask = nearest_mask & any_target.unsqueeze(1)

    curr_cost = assignment_cost_for_targets(
        target_mask=nearest_target_mask,
        inst_pos=inst_pos,
        left_tip_pos=left_tip_pos,
        right_tip_pos=right_tip_pos,
        idle_pos=idle_pos,
        waist=robot_pos[:,0],
    )

    prev_cost = assignment_cost_for_targets(
        target_mask=nearest_target_mask,
        inst_pos=inst_pos,
        left_tip_pos=prev_left_tip_pos,
        right_tip_pos=prev_right_tip_pos,
        idle_pos=idle_pos,
        waist=robot_pos[:,0],
    )

    proximity_term = torch.exp(-k_idx * first_idx) * curr_cost
    progress_term = prev_cost - curr_cost

    # # -------------------------------------------------
    # # phase terms
    # # -------------------------------------------------

    target = nearest_target_mask[:, None, :].float()      # (N, 1, M)

    down_vel = torch.clamp(-tip_vel[:, :, 2], 0.0, 2.0)[:, :, None] # (N, 2, 1)
    up_vel = torch.clamp( tip_vel[:, :, 2], 0.0, 2.0)[:, :, None]

    hit_armed_f = hit_armed.float()
    hit_disarmed_f = (~hit_armed).float()

    strike_phase = target * hit_armed_f * down_vel    # (N, 2, M)
    rearm_phase = target * hit_disarmed_f * up_vel

    strike_phase_term = strike_phase.sum(dim=(1, 2))
    rearm_phase_term = rearm_phase.sum(dim=(1, 2))

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
        goal_success, goal_wrong, goal_missed, time_error_term,
        proximity_term, progress_term,
        strike_phase_term, rearm_phase_term,
        action_l2, joint_vel_l2, limit_pen, tip_limit_pen, under_drum_pen,
    )

@torch.jit.script
def compute_rewards(
    success_term: torch.Tensor,
    wrong_term: torch.Tensor,
    miss_term: torch.Tensor,
    time_error_term: torch.Tensor,

    proximity_term: torch.Tensor,
    progress_term: torch.Tensor,

    strike_phase_term: torch.Tensor,
    rearm_phase_term: torch.Tensor,

    action_l2: torch.Tensor,
    joint_vel_l2: torch.Tensor,
    limit_pen: torch.Tensor,
    tip_limit_pen: torch.Tensor,
    under_drum_pen: torch.Tensor,

    w_success: float,
    w_wrong: float,
    w_miss: float,
    w_time_error: float,

    w_progress: float,
    w_proximity: float,

    w_strike_phase: float,
    w_rearm_phase: float,

    w_action: float,
    w_joint_vel: float,
    w_limit: float,
    w_tip_limit: float,
    w_under_drum: float,
):
    reward = (
        w_success * success_term
        - w_wrong * wrong_term
        - w_miss * miss_term
        + w_time_error * time_error_term

        + w_progress * progress_term
        - w_proximity * proximity_term

        + w_strike_phase * strike_phase_term
        + w_rearm_phase * rearm_phase_term

        - w_action * action_l2
        - w_joint_vel * joint_vel_l2
        - w_limit * limit_pen
        - w_tip_limit * tip_limit_pen
        - w_under_drum * under_drum_pen
    )
    return reward