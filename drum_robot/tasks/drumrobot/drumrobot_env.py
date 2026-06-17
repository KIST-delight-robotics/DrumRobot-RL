from __future__ import annotations

import torch
from typing import Sequence
import math
from isaaclab.utils import math as math_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane

from .drumrobot_cfg import DrumRobotEnvCfg
from .components.specs import EnvSpec, RobotSpec
from .components.robotic_drum_score import RDSCfg, RDS
from .components.robot_initializer import RobotInitializerCfg, RobotInitializer
from .components.reward import RewardComputerCfg, RewardComputer
from .components.visualizer import VisualizerCfg, Visualizer

from drum_robot.utils.logger import EnvLogger, LoggerCfg

import numpy as np
import gymnasium as gym


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

        env_specs = EnvSpec(
            num_envs=self.num_envs,
            num_drums=self.num_drums,
            episode_length_step=self.episode_length_step,
            max_lookahead_step=self.max_lookahead_step,
            hit_window_step=self.cfg.hit_window_step,
            dt=self.dt,
        )

        # 로그
        self.logger = EnvLogger(self.num_envs, self.device, LoggerCfg(interval=2000, sample_env_id=0))
        
        # RDS
        self.rds = RDS(
            device=self.device,
            cfg=RDSCfg(),
            env=env_specs,
        )

        # 로봇 초기 위치 initializer
        self.robot_initializer = RobotInitializer(
            self.device,
            RobotInitializerCfg(
                num_ctrl_joint=len(self.ctrl_joint_names),
                height_above_drum=0.1,
                joint_noise_scale= 5*math.pi/180
            ),
            ctrl_joint_names=self.ctrl_joint_names,
            instruments=self.cfg.instruments,
            robot=RobotSpec(),
        )

        # 보상 계산
        self.reward_computer = RewardComputer(self.device, RewardComputerCfg())

        # 시각화
        self.visualizer = Visualizer(
            device=self.device,
            cfg=VisualizerCfg(),
            env=env_specs,
            enable_visualization=self.cfg.enable_visualization,
        )
        self.visualizer.init_visualization(self.cfg.instruments)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg()) # add ground plane
        self.scene.articulations["robot"] = self.robot  # add articulation to scene
        self.scene.clone_environments(copy_from_source=False)   # clone and replicate
        
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

        # 다음 타격
        self.next_hits = self.rds.get_next_hits(step=self.steps, num_hits=self.cfg.num_hits)

        # 로봇 상태
        hit_armed = self.hit_armed

        obs = self._normalize_and_pack_obs(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            tip_pos=tip_pos,
            inst_pos=inst_pos,
            next_hits=self.next_hits,
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
        tip_pos = self._compute_tip_position()          # (N, 2, 3)
        
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
        self.rds.set_rds_visit(steps, hit_mask)

        # 잘못친 타격 판정
        rds = self.rds.get_rds()
        rds_visit = self.rds.get_rds_visit()
        wrong_hit = self._detect_wrong_hits(hit_mask, rds, steps)

        # 윈도우 끝났을 때 타격 성공 확인
        success, missed_target, time_error = self._finalize_target_outcomes(
            rds=rds,
            rds_visit=rds_visit,
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

        # episode time out
        self.steps = self.steps + 1
        time_out = self.steps >= self.episode_length_step - 1
        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        # 가중치 업데이트
        self.reward_computer.update_difficulty_weights(time_out[0], success, missed_target)

        # 시각화 (팁 표시, 드럼 색상 변경)
        self.visualizer.step(self.tip_pos, self.next_hits, hit_per_arm)

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        usd_pos = self.robot.data.joint_pos[:, self.ctrl_joint_ids]
        usd_vel = self.robot.data.joint_vel[:, self.ctrl_joint_ids]
        robot_pos = self._convert_usd_to_robot(usd_pos)  # (N, 9)
        robot_vel = self._convert_usd_to_robot(usd_vel)

        reward, terms, p_terms = self.reward_computer.compute(
            success=self.success,
            wrong_hit=self.wrong_hit,
            missed_target=self.missed_target,
            time_error=self.time_error,
            tip_pos=self.tip_pos,
            prev_tip_pos=self.prev_tip_pos,
            inst_pos=self.inst_pos,
            next_hits=self.next_hits,
            tip_vel=self.tip_vel,
            hit_armed_for_reward=self.hit_armed_for_reward,
            robot_vel=robot_vel,
            actions=self.actions,
            robot_pos=robot_pos,
            joint_low=self.joint_low,
            joint_high=self.joint_high,
        )

        # 로그 출력
        self.logger.add(terms)
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

        # RDS 리셋
        self.rds.reset(env_ids=env_ids, score_ratio=0.0, selection_strength=0.0)  # 랜덤으로 RDS 생성해서 사용

        # 로봇 자세 리셋
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]  # 기본 자세 가져오기
        default_joint_vel = self.robot.data.default_joint_vel[env_ids]

        joint_pos = self._get_init_joint_pos(env_ids, default_joint_pos)
        joint_vel = torch.zeros_like(default_joint_vel)
        
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 텐서 변수들 리셋
        self._reset_tensors(env_ids)

        # 시각화
        self.visualizer.reset(self.inst_pos)
        
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
            low=-np.inf, high=np.inf, shape=(94,), dtype=np.float32
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
        self.episode_length_step = int(self.cfg.episode_length_s / self.dt)

        # lookahead step
        self.max_lookahead_step = int(self.cfg.max_lookahead_time / self.dt)

        # offset wrist link to tip
        L_off = torch.tensor(self.cfg.tip_offset_left, device=self.device, dtype=torch.float32)  # (3,)
        R_off = torch.tensor(self.cfg.tip_offset_right, device=self.device, dtype=torch.float32)

        self.tip_offset_L = L_off.unsqueeze(0).expand(self.num_envs, 3)
        self.tip_offset_R = R_off.unsqueeze(0).expand(self.num_envs, 3)

        self._build_joint_tensors()
        self._build_drum_tensors()

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

        self.num_drums = len(self.basic_inst_pos[:,0])

    def _alloc_buffers(self):
        N = self.num_envs
        T = self.episode_length_step
        M = self.num_drums

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
    def _normalize_and_pack_obs(self,
            joint_pos,
            joint_vel,
            tip_pos,
            inst_pos,
            next_hits,
            hit_armed,
            ) -> torch.Tensor:
        # joint_pos, joint_vel: (N, 9)
        # tip_pos: (N, 2, 3)
        # inst_pos: (N, M, 3)
        # next_hits: (N, K, M+2)
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
                next_hits.reshape(self.num_envs, 30),
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
        T = self.episode_length_step
        M = self.num_drums
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

        N = self.num_envs
        T = self.episode_length_step
        M = self.num_drums
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

    """ func (_reset_idx) """
    def _get_init_joint_pos(self, env_ids, default_joint_pos):
        joint_pos = default_joint_pos.clone()

        init_pos = self.robot_initializer.reset_init_pos(env_ids)
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
