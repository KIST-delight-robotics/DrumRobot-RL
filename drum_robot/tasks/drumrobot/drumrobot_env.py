from __future__ import annotations

import torch
from typing import Sequence
import math

import numpy as np
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils

from .drumrobot_cfg import DrumRobotEnvCfg
from .components.specs import EnvSpec, RobotSpec, DrumSpec
from .components.robotic_drum_score import RDSCfg, RDS
from .components.robot_initializer import RobotInitializerCfg, RobotInitializer
from .components.hit_detector import HitDetector, HitDetectorCfg
from .components.reward import RewardComputerCfg, RewardComputer
from .components.visualizer import VisualizerCfg, Visualizer

from drum_robot.utils.logger import EnvLogger, LoggerCfg


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

        hit_detector_cfg = HitDetectorCfg()
        rds_cfg = RDSCfg()
        max_lookahead_step = int(rds_cfg.max_lookahead_s / self.dt)
        env_specs = EnvSpec(
            num_envs=self.num_envs,
            num_drums=self.num_drums,
            episode_length_step=self.episode_length_step,
            max_lookahead_step=max_lookahead_step,
            hit_window_step=hit_detector_cfg.hit_window_step,
            dt=self.dt,
        )

        # 로그
        self.logger = EnvLogger(
            num_envs=self.num_envs,
            device=self.device,
            cfg=LoggerCfg(interval=2000, sample_env_id=0),
        )
        
        # RDS
        self.rds = RDS(
            device=self.device,
            cfg=rds_cfg,
            env=env_specs,
        )

        # 로봇 초기 위치 initializer
        self.robot_initializer = RobotInitializer(
            device=self.device,
            cfg=RobotInitializerCfg(
                num_ctrl_joint=len(self.ctrl_joint_names),
                height_above_drum=0.1,
                joint_noise_scale= 5*math.pi/180
            ),
            ctrl_joint_names=self.ctrl_joint_names,
            instruments=self.cfg.instruments,
            robot=RobotSpec(),
        )

        # 타격 감지
        self.hit_detector = HitDetector(
            device=self.device,
            cfg=hit_detector_cfg,
            env=env_specs,
        )

        # 보상 계산
        self.reward_computer = RewardComputer(
            device=self.device,
            cfg=RewardComputerCfg(),
            env=env_specs,
            drum=DrumSpec(),
        )

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
        self.next_hits = self.rds.get_next_hits(step=self.steps)

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
        tip_pos = self._compute_tip_position()   # (N, 2, 3)

        (
            hit_mask,
            tip_pos, tip_vel, prev_tip_pos, next_hit_armed,
            hit_per_arm,
        ) = self.hit_detector.detect_hit(
            tip_pos,
            self.inst_pos,
            self.hit_armed,
        )

        # 타격 시간 기록
        self.rds.set_rds_visit(self.steps, hit_mask)

        # reward 로 넘기는 값
        self.tip_pos = tip_pos
        self.tip_vel = tip_vel
        self.prev_tip_pos = prev_tip_pos

        self.hit_armed_for_reward = self.hit_armed
        self.hit_armed = next_hit_armed

        # 결과
        rds = self.rds.get_rds()
        rds_visit = self.rds.get_rds_visit()
        success, wrong_hit, missed_target, time_error = self.hit_detector.get_result(
            hit_mask,
            self.steps,
            rds,
            rds_visit,
        )
        self.success = success
        self.wrong_hit = wrong_hit
        self.missed_target = missed_target
        self.time_error = time_error

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

        # tip 리셋
        self.tip_pos = self._compute_tip_position()
        self.hit_detector.reset(env_ids, self.tip_pos)

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
            low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.observation_space,), dtype=np.float32
        )

        # print("[DEBUG] action_space:", self.action_space)
        # print("[DEBUG] observation_space:", self.observation_space)

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

        self._build_joint_tensors()
        self._build_drum_tensors()

        # obs 차원 계산
        M = self.num_drums
        K = self.cfg.num_hits
        
        self.obs_dim_joint_pos = 9                     # ctrl 관절 수
        self.obs_dim_joint_vel = 9
        self.obs_dim_tip       = 2 * 3                 # 양손 * xyz
        self.obs_dim_inst      = M * 3                 # 드럼 * xyz
        self.obs_dim_next      = K * (M + 2)           # lookahead K개 * (one-hot M + time + valid)
        self.obs_dim_armed     = 2 * M                 # 양손 * 드럼
        
        obs_dim_total = (
            self.obs_dim_joint_pos + self.obs_dim_joint_vel
            + self.obs_dim_tip + self.obs_dim_inst
            + self.obs_dim_next + self.obs_dim_armed
        )

        if obs_dim_total != self.cfg.observation_space:
             pass   # TODO
        
        # offset wrist link to tip
        L_off = torch.tensor(self.cfg.tip_offset_left, device=self.device, dtype=torch.float32)  # (3,)
        R_off = torch.tensor(self.cfg.tip_offset_right, device=self.device, dtype=torch.float32)

        self.tip_offset_L = L_off.unsqueeze(0).expand(self.num_envs, 3)
        self.tip_offset_R = R_off.unsqueeze(0).expand(self.num_envs, 3)

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
        M = self.num_drums

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
                tip_pos_n.reshape(self.num_envs, self.obs_dim_tip),
                inst_pos_n.reshape(self.num_envs, self.obs_dim_inst),
                next_hits.reshape(self.num_envs, self.obs_dim_next),
                hit_armed.reshape(self.num_envs, self.obs_dim_armed),
            ],
            dim=-1
        )
        
        return obs
    
    """ func (_get_dones) """
    def _compute_tip_position(self):
        # 스틱 링크의 월드 좌표계 위치 가져오기
        all_body_pos = self.robot.data.body_pos_w       # (num_envs, num_bodies, 3)
        all_body_quat = self.robot.data.body_quat_w     # (num_envs, num_bodies, 4)  (w,x,y,z)인 경우가 많음

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

    """ func (_reset_idx) """
    def _get_init_joint_pos(self, env_ids, default_joint_pos):
        joint_pos = default_joint_pos.clone()

        init_pos = self.robot_initializer.reset_init_pos(env_ids)
        usd_init_pos = self._convert_robot_to_usd(init_pos)
        joint_pos[:,self.ctrl_joint_ids] = usd_init_pos

        return joint_pos

    def _reset_tensors(self, env_ids):
        # 타격 상태 버퍼 리셋
        self.hit_armed[env_ids] = True  # 초기 상태는 악기 높이 차이 상관 없이 타격 준비 상태로 초기화

        # 스텝 리셋
        self.steps[env_ids] = 0 # torch.randint(0, 9, (len(env_ids),), device=self.device)
