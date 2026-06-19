# DrumRobot — 강화학습 기반 드럼 연주 로봇

**Isaac Lab** 를 사용해 양팔 9-DOF 로봇이 MIDI 악보를 따라 드럼을 연주하도록 훈련하는 강화학습 프로젝트.

## 개발 환경

| 항목 | 내용 |
|------|------|
| OS | Ubuntu 22.04.5 |
| GPU | RTX 4070 SUPER |
| CUDA | 12.8 |
| Driver | 580.x |
| Isaac Sim | 4.5 |
| Isaac Lab | 2.3.0 |
| Python | 3.10 |
| RL 프레임워크 | skrl 1.4.3+ |

---

## 문서

- [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html)
- [Isaac Lab Github Code](https://github.com/isaac-sim/IsaacLab)
- [SKRL 공식 문서](https://skrl.readthedocs.io/en/latest/#)
- [한국어 튜토리얼](https://wikidocs.net/book/18009)

---

## 환경 구축

본 프로젝트는 Isaac Lab 공식 레포에 **외부 extension** 으로 얹어 사용한다.

### 1. Isaac Lab 클론

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### 2. drum_robot extension 배치

IsaacLab 루트에서:

```bash
git clone https://github.com/KIST-delight-robotics/DrumRobot-RL.git source/extensions/drum_robot
```

결과 구조:

```
IsaacLab/
├── isaaclab.sh                  # 공식 실행 런처
└── source/
    └── extensions/
        └── drum_robot/          # ← 본 프로젝트
            ├── extension.toml
            ├── pyproject.toml
            └── drum_robot/
```

- `extension.toml` → Isaac Sim (Kit) 이 extension으로 인식.
- `pyproject.toml` → Isaac Lab 설치 과정에서 `pip install -e .` 로 파이썬 패키지로 등록.

### 3. 실행 위치

이후의 모든 명령은 `source/extensions/drum_robot/` 디렉토리 기준이다. `../../../isaaclab.sh` 가 IsaacLab 루트의 공식 런처를 가리킨다.

```bash
cd source/extensions/drum_robot
```

---

## 실행

### 가상 환경
```bash
conda activate env_isaaclab
```
```bash
conda deactivate
```

### 아이작 심 (빈 프로젝트)
```bash
../../../isaaclab.sh -s
```

### 학습

```bash
../../../isaaclab.sh -p drum_robot/scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-drum_robot-Direct-v0 \
  --num_envs=4096 \
  --headless
```

### 체크포인트 이어서 학습

```bash
../../../isaaclab.sh -p drum_robot/scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-drum_robot-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --checkpoint="logs/skrl/drum_robot/.../checkpoints/agent_XX.pt"
```

### 텐서보드
```bash
tensorboard --logdir logs/
```

### 시각화

```bash
../../../isaaclab.sh -p drum_robot/scripts/reinforcement_learning/skrl/play.py \
  --task=DrumRobot-drum_robot-Direct-v0 \
  --num_envs=1 \
  --checkpoint="logs/skrl/drum_robot/.../checkpoints/agent_XX.pt"
```

---

## 디렉토리 구조 트리

```
source/extensions/drum_robot/drum_robot/
├── assets/drum_robot/                       # 로봇·드럼 자산
│   ├── config/                              # robot articulation cfg
│   ├── urdf/                                # URDF 원본
│   └── usd/                                 # IsaacSim 용 USD
│
├── scripts/reinforcement_learning/skrl/     # 학습·평가 진입 스크립트
│   ├── train.py
│   └── play.py
│
├── tasks/drumrobot/                         # RL 환경 정의
│   ├── drumrobot_env.py                     # DirectRLEnv 본체 (오케스트레이터)
│   ├── drumrobot_cfg.py                     # 환경 cfg
│   ├── agents/
│   │   └── skrl_ppo_cfg.yaml                # PPO 하이퍼파라미터
│   └── components/                          # 환경 내부 컴포넌트
│       ├── hit_detector.py                  # 타격 판정 및 결과 매칭
│       ├── reward.py                        # 보상 계산
│       ├── robotic_drum_score.py            # MIDI 악보(RDS) 관리
│       ├── robot_interface.py               # USD↔로봇 좌표 변환, 관절·팁 접근
│       ├── robot_initializer.py             # 초기 자세 생성
│       ├── ik_solver.py                     # 기하학적 IK
│       ├── specs.py                         # 공유 dataclass (로봇·드럼 상수)
│       └── visualizer.py                    # 시각화 (팁 마커, 드럼 색)
│
└── utils/
    └── logger.py                            # 학습 로그 누적·출력
```