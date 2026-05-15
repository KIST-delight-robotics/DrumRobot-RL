# ./isaaclab.sh -p source/extensions/drum_robot/drum_robot/scripts/test_joint.py 

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    # 반드시 AppLauncher 먼저
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=False)
    simulation_app = app_launcher.app

    # AppLauncher 이후 import
    import torch

    from drum_robot.tasks.test_joint.drumrobot_cfg import DrumRobotEnvCfg
    from drum_robot.tasks.test_joint.drumrobot_env import DrumRobotEnv

    cfg = DrumRobotEnvCfg()

    # env 생성
    env = DrumRobotEnv(cfg=cfg)

    # reset
    obs, _ = env.reset()

    # simple test: action을 좌우로 왔다갔다
    t0 = time.time()
    for i in range(args.steps):
        # [-1,1] 사인파
        a = torch.sin(torch.tensor([i * 0.02], device=env.device)).view(1, 1)
        obs, rew, terminated, time_out, _ = env.step(a)

        if i % 60 == 0:
            q = obs["policy"][0, 0].item()
            qd = obs["policy"][0, 1].item()
            r = rew[0].item()
            print(f"step={i:04d} | action={a.item():+.3f} | q={q:+.3f} | qd={qd:+.3f} | r={r:+.3f}")

        if terminated.any() or time_out.any():
            obs, _ = env.reset()

    print(f"\n[DONE] elapsed={time.time() - t0:.2f}s")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
