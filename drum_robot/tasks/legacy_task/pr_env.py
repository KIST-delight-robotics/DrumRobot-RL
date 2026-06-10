# drum_robot/tasks/legacy_task/pr_env.py

# 실행 순서
# _pre_physics_step -> _apply_action (decimation) -> _get_dones -> _get_rewards -> _get_observations

from __future__ import annotations

import torch
from typing import Sequence
import math
from isaaclab.utils import math as math_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import GroundPlaneCfg, spawn_ground_plane

from .pr_cfg import DrumRobotEnvCfg

import numpy as np
import gymnasium as gym

from tqdm import tqdm   # 로그

import omni.usd
from pxr import UsdGeom, Gf

class DrumRobotEnv(DirectRLEnv):
    cfg: DrumRobotEnvCfg

    def __init__(self, cfg: DrumRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 반드시 single-env shape로 정의
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )

        # 로봇의 tip 위치를 저장할 텐서
        self.tip_R = torch.zeros((self.num_envs, 3), device=self.device)
        self.tip_L = torch.zeros((self.num_envs, 3), device=self.device)

        # offset wrist link to tip
        L_off = torch.tensor(self.cfg.tip_offset_left, device=self.device, dtype=torch.float32)  # (3,)
        R_off = torch.tensor(self.cfg.tip_offset_right, device=self.device, dtype=torch.float32)

        # quat으로 로컬 오프셋을 월드로 회전
        self.tip_offset_L = L_off.unsqueeze(0).expand(self.num_envs, 3)
        self.tip_offset_R = R_off.unsqueeze(0).expand(self.num_envs, 3)

        # 목표 지점을 저장할 텐서 (num_envs, 3차원 x 2)
        self.targets = torch.zeros((self.num_envs, 6), device=self.device)

        self.target_min = torch.tensor(self.cfg.target_min, device=self.device, dtype=torch.float32)
        self.target_max = torch.tensor(self.cfg.target_max, device=self.device, dtype=torch.float32)

        # 성공 버퍼
        self.success_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        # print("[DEBUG] action_space:", self.action_space)
        # print("[DEBUG] observation_space:", self.observation_space)
        # print("[DEBUG] cfg.action_dim:", getattr(self.cfg, "action_dim", None))

        # view가 준비된 뒤에 ids/bodies resolve
        self._resolve_ids()

        # 로그 출력 변수들 선언
        self._init_log()

    def _resolve_ids(self):

        # 양손 스틱 링크의 인덱스 찾기
        left = self.robot.find_bodies("left_wrist")     # ([10], ['left_wrist'])
        right = self.robot.find_bodies("right_wrist")   # ([11], ['right_wrist'])
        if len(left) == 0 or len(right) == 0:
            raise RuntimeError("wrist body not found. Check USD body names.")
        self.left_hand_idx = left[0]
        self.right_hand_idx = right[0]

        # print("[DEBUG] body_names: ", self.robot.data.body_names)

        # 학습할 관절 설정
        joint_names = [
            "waist_joint",
            "left_shoulder_1","left_shoulder_2","left_elbow",
            "right_shoulder_1","right_shoulder_2","right_elbow",
            "left_wrist","right_wrist",
        ]

        ids, names = self.robot.find_joints(joint_names)
        self.joint_name_to_id = { name: jid for name, jid in zip(names, ids) }
        
        self.ctrl_joint_ids = ids
        self.ctrl_joint_names = names
        if len(self.ctrl_joint_ids) != len(joint_names):
            raise RuntimeError(f"ctrl_joint_ids mismatch: {len(self.ctrl_joint_ids)} (expected 9). names={self.ctrl_joint_names}")
        
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

        # 관측값 정규화
        self.joint_center = 0.5 * (self.joint_low + self.joint_high)
        self.joint_half_range = 0.5 * (self.joint_high - self.joint_low) + 1e-6
        self.joint_vel_scale = torch.tensor(5.0, device=self.device, dtype=torch.float32)   # GPT가 5~10 rad/s 권장
        self.target_center = torch.tensor(self.cfg.target_center, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.target_half_range = torch.tensor(self.cfg.target_half_range, device=self.device, dtype=torch.float32).unsqueeze(0) + 1e-6 

        # 로봇 관절-USD 관절 방향
        self.dir_tensor = torch.tensor(
            [self.cfg.joint_usd_dir[name] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

        # print("[DEBUG] find_joints ids: ", self.ctrl_joint_ids)
        # print("[DEBUG] find_joints names: ", self.ctrl_joint_names)
        # print("[DEBUG] self.joint_low: ", self.joint_low)
        # print("[DEBUG] self.joint_high: ", self.joint_high)
    
    def _init_log(self):
        self._global_step = 0
        self._log_interval = 2000   # n step마다 출력

        self._left_dist_sum = torch.zeros(self.num_envs, device=self.device)
        self._right_dist_sum = torch.zeros(self.num_envs, device=self.device)
        self._action_l2_sum = torch.zeros(self.num_envs, device=self.device)
        self._reward_sum = torch.zeros(self.num_envs, device=self.device)
        self._limit_pen_sum = torch.zeros(self.num_envs, device=self.device)
        self._died_sum = torch.zeros(self.num_envs, device=self.device)

        # 성공 통계
        self._success_count = 0
        self._episode_count = 0

        self._step_count = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg()) # add ground plane
        self.scene.articulations["robot"] = self.robot  # add articulation to scene
        self.scene.clone_environments(copy_from_source=False)   # clone and replicate

        if self.cfg.enable_target_markers:  # 타겟 마커 추가
            self._create_target_markers()

        if self.cfg.enable_tip_markers:     # 팁 마커 추가
            self._create_tip_markers()
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))  # 광원 추가
        light_cfg.func("/World/Light", light_cfg)
    
    def _create_target_markers(self):

        # 고속 업데이트를 위한 TranslateOp 캐시
        # self.target_marker_xform_L = []
        # self.target_marker_xform_R = []
        self._target_marker_translate_ops_L = []
        self._target_marker_translate_ops_R = []

        # sphere 설정
        sphere_cfg_L = sim_utils.SphereCfg(
            radius=self.cfg.target_marker_radius,
        )
        sphere_cfg_R = sim_utils.SphereCfg(
            radius=self.cfg.target_marker_radius,
        )

        # 현재 IsaacSim의 USD Stage 접근
        stage = omni.usd.get_context().get_stage()

        for i in range(self.num_envs):

            # USD Stage 위에 Prim을 생성 (이미 존재하면 타입을 유지한 채 반환)
            viz_root = f"/World/envs/env_{i}/_viz"
            stage.DefinePrim(viz_root, "Xform")     # Xform = Transform 노드
            xform_L_path = f"{viz_root}/TargetMarkerL"
            xform_R_path = f"{viz_root}/TargetMarkerR"
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

            color_vec = Gf.Vec3f(
                float(self.cfg.target_marker_color[0]),
                float(self.cfg.target_marker_color[1]),
                float(self.cfg.target_marker_color[2])
            )

            gprim_L.CreateDisplayColorPrimvar().Set([color_vec])
            gprim_R.CreateDisplayColorPrimvar().Set([color_vec])

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

            # self.target_marker_xform_L.append(xform_L_path)
            # self.target_marker_xform_R.append(xform_R_path)
            self._target_marker_translate_ops_L.append(t_op_L)
            self._target_marker_translate_ops_R.append(t_op_R)

    def _update_target_markers(self, env_ids):
        left_pos = self.targets[env_ids, :3]
        right_pos = self.targets[env_ids, 3:]

        for k, eid in enumerate(env_ids):
            lp = left_pos[k]
            rp = right_pos[k]

            # torch -> float 바로 (cpu numpy 변환 없이)
            self._target_marker_translate_ops_L[eid].Set(
                Gf.Vec3f(float(lp[0]), float(lp[1]), float(lp[2]))
            )
            self._target_marker_translate_ops_R[eid].Set(
                Gf.Vec3f(float(rp[0]), float(rp[1]), float(rp[2]))
            )

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

            color_vec = Gf.Vec3f(
                float(self.cfg.tip_marker_color[0]),
                float(self.cfg.tip_marker_color[1]),
                float(self.cfg.tip_marker_color[2])
            )

            gprim_L.CreateDisplayColorPrimvar().Set([color_vec])
            gprim_R.CreateDisplayColorPrimvar().Set([color_vec])

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
        left_pos = self.tip_L
        right_pos = self.tip_R

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

    def _convert_usd_to_robot(self, usd_data):
        return self.dir_tensor * usd_data

    def _convert_robot_to_usd(self, robot_data):
        return self.dir_tensor * robot_data

    def _pre_physics_step(self, actions: torch.Tensor):     
        # RL 에이전트로부터 받은 actions 저장
        actions = actions.to(torch.float32)

        # action 범위를 [-1, 1]로 강제
        actions = torch.clamp(actions, -1.0, 1.0)

        self.actions = actions.clone()

        usd_q = self.robot.data.joint_pos[:, self.ctrl_joint_ids]
        q = self._convert_usd_to_robot(usd_q)

        target = q + self.actions * self.cfg.action_scale
        target = torch.max(torch.min(target, self.joint_high), self.joint_low)  # joint_limit 내로 clip
        usd_target = self._convert_robot_to_usd(target)

        self.target_joint_pos = usd_target

    def _apply_action(self):
        self.robot.set_joint_position_target(self.target_joint_pos, joint_ids=self.ctrl_joint_ids)

    def _get_observations(self) -> dict:
        usd_pos = self.robot.data.joint_pos[:, self.ctrl_joint_ids]        # (num_envs, 9)
        usd_vel = self.robot.data.joint_vel[:, self.ctrl_joint_ids]        # (num_envs, 9)
    
        joint_pos = self._convert_usd_to_robot(usd_pos)
        joint_vel = self._convert_usd_to_robot(usd_vel)

        # 정규화
        joint_pos_n = (joint_pos - self.joint_center) / self.joint_half_range
        joint_pos_n = torch.clamp(joint_pos_n, -1.5, 1.5)  # 약간 여유

        joint_vel_n = joint_vel / self.joint_vel_scale
        joint_vel_n = torch.clamp(joint_vel_n, -5.0, 5.0)

        targets_n = (self.targets - self.target_center) / self.target_half_range
        targets_n = torch.clamp(targets_n, -2.0, 2.0)

        target_L = self.targets[:, :3]
        target_R = self.targets[:, 3:]

        err_L_n = (self.tip_L - target_L) / self.target_half_range[:, :3]
        err_R_n = (self.tip_R - target_R) / self.target_half_range[:, 3:]
        err_L_n = torch.clamp(err_L_n, -2.0, 2.0)
        err_R_n = torch.clamp(err_R_n, -2.0, 2.0)

        obs = torch.cat([joint_pos_n, joint_vel_n, targets_n, err_L_n, err_R_n], dim=-1).to(torch.float32)  # (num_envs, 30)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 스틱 링크의 월드 좌표계 위치 가져오기
        all_body_pos = self.robot.data.body_pos_w      # (num_envs, num_bodies, 3)
        all_body_quat = self.robot.data.body_quat_w    # (num_envs, num_bodies, 4)  (w,x,y,z)인 경우가 많음

        L_wrist_pos = all_body_pos[:, self.left_hand_idx, :].squeeze(1)       # (num_envs, 1, 3) -> (num_envs, 3) 1인 차원을 제거
        R_wrist_pos = all_body_pos[:, self.right_hand_idx, :].squeeze(1)
        L_quat = all_body_quat[:, self.left_hand_idx, :].squeeze(1)     # (num_envs, 1, 4) -> (num_envs, 4)
        R_quat = all_body_quat[:, self.right_hand_idx, :].squeeze(1)

        # 팁 위치 구하기
        L_tip_w = L_wrist_pos + math_utils.quat_apply(L_quat, self.tip_offset_L)
        R_tip_w = R_wrist_pos + math_utils.quat_apply(R_quat, self.tip_offset_R)

        # 월드 기준 -> env 기준
        self.tip_L = L_tip_w - self.scene.env_origins
        self.tip_R = R_tip_w - self.scene.env_origins

        # 위치 표시
        if self.cfg.enable_tip_markers:
            self._update_tip_markers()

        # robot joint pos (robot frame) for limit barrier
        usd_pos = self.robot.data.joint_pos[:, self.ctrl_joint_ids]
        usd_vel = self.robot.data.joint_vel[:, self.ctrl_joint_ids]
        robot_pos = self._convert_usd_to_robot(usd_pos)  # (N,9)
        robot_vel = self._convert_usd_to_robot(usd_vel)

        dist_term, exp_term, action_l2, joint_vel_l2, limit_pen, died, left_dist, right_dist = compute_reward_terms(
            self.tip_L,
            self.tip_R,
            self.targets[:, :3],
            self.targets[:, 3:],
            robot_vel,
            self.actions,
            robot_pos,
            self.joint_low,
            self.joint_high,
            float(self.cfg.exp_k),
            float(self.cfg.limit_margin),
        )

        r = compute_rewards(
            dist_term, exp_term, action_l2, joint_vel_l2, limit_pen, died,
            float(self.cfg.rew_w_dist),
            float(self.cfg.rew_w_exp),
            float(self.cfg.rew_w_action),
            float(self.cfg.rew_w_joint_vel),
            float(self.cfg.rew_w_limit),
            float(self.cfg.rew_w_died),
        )

        # 성공 누적 + 보너스 추가
        thr = float(self.cfg.success_dist_thr)
        k = int(self.cfg.success_hold_steps)
        success_now = (left_dist < thr) & (right_dist < thr)

        self.success_buf = torch.where(     # 성공이면 +1, 아니면 0으로 리셋
            success_now,
            self.success_buf + 1,
            torch.zeros_like(self.success_buf),
        )   

        success_confirm = self.success_buf >= k  # (num_envs,) bool

        # 성공 보너스(선택)
        bonus = float(self.cfg.success_bonus)
        if bonus > 0.0:
            r = r + success_confirm.to(torch.float32) * bonus

        # 로그는 term 기반 출력
        self._print_log_terms(left_dist, right_dist, action_l2, r, limit_pen, died)

        return r
    
    def _print_log_terms(self, left_dist, right_dist, action_l2, r, limit_pen, died):

        self._left_dist_sum += left_dist
        self._right_dist_sum += right_dist
        self._action_l2_sum += action_l2
        self._reward_sum += r
        self._limit_pen_sum += limit_pen
        self._died_sum += died

        self._step_count += 1

        self._global_step += 1
        if self._global_step % self._log_interval == 0:

            denom = (self._step_count + 1e-6)
            avg_left = (self._left_dist_sum / denom).mean()
            avg_right = (self._right_dist_sum / denom).mean()
            avg_action = (self._action_l2_sum / denom).mean()
            avg_reward = (self._reward_sum / denom).mean()
            avg_limit_pen = (self._limit_pen_sum / denom).mean()
            avg_died = (self._died_sum / denom).mean()

            if self._episode_count > 0:
                success_rate = self._success_count / self._episode_count
                self._success_count = 0
                self._episode_count = 0
            else:
                success_rate = 0.0

            tqdm.write(
                f"[STEP {self._global_step}] "
                f"reward={avg_reward:.3f} | "
                f"L_dist={avg_left:.3f} | R_dist={avg_right:.3f} | "
                f"dist_sum={avg_left+avg_right:.3f} | "
                f"action_l2={avg_action:.3f} | "
                f"limit_pen={avg_limit_pen:.3f} | "
                f"died={avg_died:.3f} | "
                f"success_rate={success_rate:.3f}"
            )

            self._left_dist_sum.zero_()
            self._right_dist_sum.zero_()
            self._action_l2_sum.zero_()
            self._reward_sum.zero_()
            self._limit_pen_sum.zero_()
            self._died_sum.zero_()

            self._step_count.zero_()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        usd_pos = self.robot.data.joint_pos[:, self.ctrl_joint_ids]     # (num_envs, 9)
        robot_pos = self._convert_usd_to_robot(usd_pos)

        eps = 5.0 * math.pi / 180.0   # 범위를 5도 이상 넘은 경우에만 죽임
        too_low = robot_pos < (self.joint_low - eps)
        too_high = robot_pos > (self.joint_high + eps)

        out_of_limit = too_low | too_high                               # (num_envs, 9)
        died = torch.any(out_of_limit, dim=1)                           # (num_envs)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 성공 확정 시 종료(terminate)
        success_done = self.success_buf >= int(self.cfg.success_hold_steps)
        died = died | success_done

        # 성공율 로그 출력
        num_success = success_done.sum().item()     # 성공한 env 수
        self._success_count += num_success
        num_done = (died | time_out).sum().item()   # 이번 step에서 종료되는 env 수
        self._episode_count += num_done

        # 자가 충돌은 보상을 먼저 넣어 회피하게 하고 안정되면 죽이는 코드 만들기

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # 부모 클래스의 리셋(버퍼 초기화 등) 호출
        super()._reset_idx(env_ids)

        # 타겟 업데이트
        rand = torch.rand((len(env_ids), len(self.target_max)), device=self.device) # (num_envs, 6)
        target_init = self.target_min.unsqueeze(0) + (self.target_max - self.target_min).unsqueeze(0) * rand

        # z 축 보정 (waist joint 기준에서 env 기준으로 변환)
        target_init[:,2] += self.cfg.robot_waist_joint_offset_z
        target_init[:,5] += self.cfg.robot_waist_joint_offset_z

        self.targets[env_ids] = target_init

        # 위치 표시
        if self.cfg.enable_target_markers:
            self._update_target_markers(env_ids)

        # 로봇 자세 리셋
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]  # 기본 자세 가져오기
        default_joint_vel = self.robot.data.default_joint_vel[env_ids]

        joint_pos = default_joint_pos.clone()

        for joint_name in self.ctrl_joint_names: 

            joint_id = self.joint_name_to_id[joint_name]
            low, high = self.cfg.joint_range[joint_name]
            dir = self.cfg.joint_usd_dir[joint_name]

            rand = torch.rand(len(env_ids), device=self.device)
            init_pos = low + (high - low) * rand

            usd_init_pos = dir * init_pos
            joint_pos[:, joint_id] = usd_init_pos

        joint_vel = torch.zeros_like(default_joint_vel)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 성공 버퍼 리셋
        self.success_buf[env_ids] = 0

@torch.jit.script
def compute_reward_terms(
    left_hand_pos: torch.Tensor,      # (N,3) world [m]
    right_hand_pos: torch.Tensor,     # (N,3) world [m]
    left_target: torch.Tensor,        # (N,3) world [m]
    right_target: torch.Tensor,       # (N,3) world [m]
    joint_vel: torch.Tensor,          # (N, num_joints) [rad/s]
    action: torch.Tensor,             # (N, 9) [-1,1]
    robot_pos: torch.Tensor,          # (N, 9) robot frame [rad]
    joint_low: torch.Tensor,          # (1, 9) [rad]
    joint_high: torch.Tensor,         # (1, 9) [rad]
    exp_k: float,
    limit_margin: float,
):
    # --- distances ---
    left_dist = torch.norm(left_target - left_hand_pos, dim=-1)    # (N,)
    right_dist = torch.norm(right_target - right_hand_pos, dim=-1) # (N,)
    dist_sum = left_dist + right_dist                               # (N,)

    # --- shaping ---
    # 가까워지면 크게, 멀어지면 0 근처
    exp_term = torch.exp(-exp_k * dist_sum)                       # (N,)
    dist_term = -dist_sum                                         # (N,)

    # --- smoothness / energy ---
    action_l2 = torch.sum(action * action, dim=-1)                 # (N,)
    joint_vel_l2 = torch.sum(joint_vel * joint_vel, dim=-1)        # (N,)

    # barrier (limit 근처 미리 벌점)
    low_v = torch.clamp((joint_low + limit_margin) - robot_pos, min=0.0)
    high_v = torch.clamp(robot_pos - (joint_high - limit_margin), min=0.0)
    limit_pen = torch.sum(low_v * low_v + high_v * high_v, dim=-1)

    # died (진짜 초과했는지)
    too_low = robot_pos < joint_low
    too_high = robot_pos > joint_high
    died = torch.any(too_low | too_high, dim=-1).to(torch.float32)

    return dist_term, exp_term, action_l2, joint_vel_l2, limit_pen, died, left_dist, right_dist

@torch.jit.script
def compute_rewards(
    dist_term: torch.Tensor,
    exp_term: torch.Tensor,
    action_l2: torch.Tensor,
    joint_vel_l2: torch.Tensor,
    limit_pen: torch.Tensor,
    died: torch.Tensor,
    w_dist: float,
    w_exp: float,
    w_action: float,
    w_joint_vel: float,
    w_limit: float,
    w_died: float,
):
    reward = (
        w_dist * dist_term
        + w_exp * exp_term
        - w_action * action_l2
        - w_joint_vel * joint_vel_l2
        - w_limit * limit_pen
        - w_died * died
    )
    return reward