from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .drumrobot_cfg import DrumRobotEnvCfg


class DrumRobotEnv(DirectRLEnv):
    cfg: DrumRobotEnvCfg

    def __init__(self, cfg: DrumRobotEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # handle
        self.robot: Articulation = self.scene["robot"]

        # joint index cache
        names = self.robot.data.joint_names
        if cfg.waist_joint_name not in names:
            raise RuntimeError(
                f"waist_joint_name='{cfg.waist_joint_name}' not found. joint_names={names}"
            )
        self.waist_idx = names.index(cfg.waist_joint_name)

        # buffers
        self._actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._waist_target = torch.zeros((self.num_envs,), device=self.device)

        # for printing once
        if self.num_envs == 1:
            print("\n==== Joint Names ====")
            for i, n in enumerate(names):
                print(f"[{i:02d}] {n}")
            print("=====================\n")
            print(f"[INFO] waist_idx={self.waist_idx} joint={cfg.waist_joint_name}\n")

    def _setup_scene(self):
        """Scene spawn is driven by cfg.scene.
        DirectRLEnv base will create self.scene and spawn assets from cfg.scene.
        """
        super()._setup_scene()

    def _pre_physics_step(self, actions: torch.Tensor):
        # clamp
        self._actions[:] = torch.clamp(actions, -1.0, 1.0)

        # scale action -> target rad
        max_rad = float(self.cfg.max_waist_rad)
        self._waist_target[:] = self._actions[:, 0] * max_rad

    def _apply_action(self):
        # current targets: full vector target
        q_tgt = self.robot.data.joint_pos_target.clone()

        # set waist target only
        q_tgt[:, self.waist_idx] = self._waist_target

        self.robot.set_joint_position_target(q_tgt)

    def _get_observations(self):
        q = self.robot.data.joint_pos[:, self.waist_idx]
        qd = self.robot.data.joint_vel[:, self.waist_idx]
        obs = torch.stack([q, qd], dim=-1)  # (num_envs, 2)
        return {"policy": obs}

    def _get_rewards(self):
        # reward: keep waist near target_waist_rad and penalize velocity
        q = self.robot.data.joint_pos[:, self.waist_idx]
        qd = self.robot.data.joint_vel[:, self.waist_idx]

        target = float(self.cfg.target_waist_rad)
        err = q - target

        r = -torch.abs(err) - 0.01 * torch.abs(qd)
        return r

    def _get_dones(self):
        # time out only (step limit)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out, dtype=torch.bool)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # default reset
        super()._reset_idx(env_ids)

        # after reset, set target buffer too
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._actions[env_ids] = 0.0
        self._waist_target[env_ids] = 0.0
