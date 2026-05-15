from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

USD_PATH = "/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/drum_robot/assets/drum_robot/usd/drum_robot.usd"
PRIM_PATH = "/World/DrumRobot"


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1/120, device="cuda:0"))
    sim.reset()

    cfg = ArticulationCfg(
        prim_path=PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),

        # ✅ 초기화 통과용 최소 init_state (라디안)
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "left_shoulder_1": 0.60,   # lower(0.524) + margin
                "right_shoulder_1": 0.60,
            }
        ),

        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            )
        },
    )

    robot = Articulation(cfg)

    # 이 reset에서 articulation initialize/validate가 돈다
    sim.reset()
    robot.reset()

    print("\n==== Joint Names ====")
    for i, name in enumerate(robot.data.joint_names):
        print(f"[{i:02d}] {name}")

    simulation_app.close()


if __name__ == "__main__":
    main()
