# OPERATIONS

학습을 돌리고, 결과를 확인하고, 사고가 났을 때 대처하는 방법을 모은 문서.

---

## 로그 / 체크포인트

- 위치: `logs/skrl/<directory>/<날짜_experiment_name>/checkpoints/agent_XX.pt`
- `directory`, `experiment_name`, 저장 주기는 `drum_robot/tasks/drumrobot/agents/skrl_ppo_cfg.yaml`에서 설정
- 기본은 `auto` — 학습 전체의 10%마다 저장

## 텐서보드

```bash
tensorboard --logdir logs/
```

## 학습 검증 방법

- `total reward` 곡선이 초반에 수렴 추세인지 확인. 발산하거나 수렴 못 하면 폐기하고 다시.
- **가장 확실한 방법은 짧게 학습한 뒤 `play.py`로 시각화 확인하는 것.** 메트릭만 보고 판단하지 말 것.

## 학습된 정책 시각화

```bash
../../../isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task=DrumRobot-drum_score-Direct-v0 \
  --num_envs=1 \
  --checkpoint="logs/skrl/.../checkpoints/agent_XX.pt"
```

`--headless` 옵션을 빼고 `--num_envs=1`로 실행하면 GUI에서 정책 동작을 직접 볼 수 있다.

## 학습 시간 기준

- 4096 envs로 1M step 약 3시간 (rollout 길이와 하이퍼파라미터에 따라 변동)

---

## 알려진 사고 사례

### NaN 발생

- **증상**: `_get_observations` 또는 `_pre_physics_step`에서 `RuntimeError("NaN ...")`로 raise.
- **목격된 원인**: 관절이 한계에 도달했는데 같은 방향의 action이 계속 들어와 명령-실제 위치 괴리가 누적되면서 발생.
- **점검 포인트**: 관절 한계 페널티 가중치, `action_scale`.

### Isaac Sim이 갑자기 실행되지 않음

- PC 리부트로 해결되는 경우가 가장 많음.
- 그래도 안 되면 NVIDIA 그래픽 드라이버 재설치.