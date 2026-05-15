# ./isaaclab.sh -p source/extensions/drum_robot/drum_robot/scripts/test_gohome.py

from __future__ import annotations

import argparse
import math
from typing import Dict, List

from isaaclab.app import AppLauncher


USD_PATH = "/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/drum_robot/assets/drum_robot/usd/drum_robot.usd"
PRIM_PATH = "/World/DrumRobot"

# sim
SIM_DT = 1.0 / 120.0
DEVICE = "cuda"

# implicit actuator PD (너무 크면 딱딱, 너무 작으면 추종 느슨)
KP = 250.0
KD = 10.0


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def quintic_s(t: float, T: float) -> float:
    """0~T 동안 0->1로 부드럽게: endpoints에서 vel/acc 0."""
    if T <= 0.0:
        return 1.0
    tau = max(0.0, min(1.0, t / T))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--t_move", type=float, default=4.0)  # 천천히 이동 시간
    parser.add_argument("--t_hold", type=float, default=1.0)  # 도착 후 유지
    parser.add_argument("--log_dt", type=float, default=0.1)  # 로그 간격
    args = parser.parse_args()

    # Launch
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=False)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    sim = SimulationContext(SimulationCfg(dt=SIM_DT, device=DEVICE))
    if not args.headless:
        sim.set_camera_view([3.0, 2.0, 1.8], [0.0, 0.0, 1.0])

    # (best-effort) ground & light
    try:
        gp = sim_utils.GroundPlaneCfg()
        gp.func("/World/GroundPlane", gp)
    except Exception:
        pass

    try:
        light = sim_utils.DistantLightCfg(intensity=3000.0)
        light.func("/World/Light", light)
    except Exception:
        pass

    # Robot config
    robot_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "pos": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=KP,
                damping=KD,
            )
        },
    )
    robot = Articulation(robot_cfg.replace(prim_path=PRIM_PATH))

    # Reset
    sim.reset()
    robot.reset()

    # Warm-up
    for _ in range(10):
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)

    joint_names: List[str] = robot.data.joint_names
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    print("\n==== Joint Names ====")
    for i, n in enumerate(joint_names):
        print(f"[{i:02d}] {n}")
    print("=====================\n")

    # ---- 확정된 joint names 기반 매핑 ----
    NAME_MAP: Dict[str, str] = {
        "waist": "waist_joint",
        "head_1": "head",
        "head_2": "head_2",
        "l_shoulder_1": "left_shoulder_1",
        "l_shoulder_2": "left_shoulder_2",
        "l_elbow": "left_elbow",
        "l_wrist": "left_wrist",
        "r_shoulder_1": "right_shoulder_1",
        "r_shoulder_2": "right_shoulder_2",
        "r_elbow": "right_elbow",
        "r_wrist": "right_wrist",
    }

    # ---- 목표 각도 (deg) ----
    TARGET_DEG: Dict[str, float] = {
        "waist": 0.0,
        "head_1": 10.0,
        "head_2": 0.0,
        "l_shoulder_1": 90.0,
        "l_shoulder_2": 0.0,
        "l_elbow": -90.0,
        "l_wrist": 90.0,
        "r_shoulder_1": 90.0,
        "r_shoulder_2": 0.0,
        "r_elbow": 90.0,
        "r_wrist": -90.0,
    }

    # resolve indices
    sel = []
    logical = []
    for key, real_name in NAME_MAP.items():
        if real_name not in name_to_idx:
            raise RuntimeError(f"Joint '{real_name}' not found in robot joint_names.")
        sel.append(name_to_idx[real_name])
        logical.append(key)

    # read initial pose
    q_now_all = robot.data.joint_pos[0].clone()  # (num_joints,)
    q_start = q_now_all.clone()
    q_goal = q_now_all.clone()

    for key in logical:
        idx = name_to_idx[NAME_MAP[key]]
        q_goal[idx] = deg2rad(TARGET_DEG[key])

    print("===== INITIAL (q0) =====")
    for key in logical:
        idx = name_to_idx[NAME_MAP[key]]
        print(f"{key:>14s} | idx={idx:02d} | name='{NAME_MAP[key]}' | q0={float(q_start[idx]):+.4f} rad")
    print("========================\n")

    print("===== TARGET (q_goal) =====")
    for key in logical:
        idx = name_to_idx[NAME_MAP[key]]
        print(f"{key:>14s} | idx={idx:02d} | target={float(q_goal[idx]):+.4f} rad  ({TARGET_DEG[key]:+.1f} deg)")
    print("===========================\n")

    # run trajectory
    t = 0.0
    next_log = 0.0
    total = float(args.t_move) + float(args.t_hold)
    steps = int(math.ceil(total / SIM_DT))

    print("===== RUN =====")
    print(f"device={DEVICE} dt={SIM_DT} t_move={args.t_move} t_hold={args.t_hold}\n")

    for k in range(steps):
        if t <= args.t_move:
            s = quintic_s(t, args.t_move)
            q_t = q_start + (q_goal - q_start) * s
            phase = "move"
        else:
            q_t = q_goal
            phase = "hold"

        robot.set_joint_position_target(q_t.unsqueeze(0))
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)

        if t >= next_log or k == steps - 1:
            q_now = robot.data.joint_pos[0]
            parts = []
            for key in logical:
                idx = name_to_idx[NAME_MAP[key]]
                err = float(q_t[idx] - q_now[idx])
                parts.append(f"{key}:{float(q_now[idx]):+.3f} err:{err:+.3f}")
            print(f"t={t:5.2f} {phase:4s} | " + " | ".join(parts))
            next_log += float(args.log_dt)

        t += SIM_DT

    print("\n===== DONE =====")
    q_final = robot.data.joint_pos[0]
    for key in logical:
        idx = name_to_idx[NAME_MAP[key]]
        print(
            f"{key:>14s} | q={float(q_final[idx]):+.4f} rad | target={float(q_goal[idx]):+.4f} rad"
        )
    print("================\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
