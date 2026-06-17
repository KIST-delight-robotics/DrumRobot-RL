# DrumRobot — 강화학습 기반 드럼 연주 로봇

**Isaac Lab + skrl(PPO)** 를 사용해 양팔 9-DOF 로봇이 MIDI 악보를 따라 드럼을 연주하도록 훈련하는 강화학습 프로젝트.

| 항목 | 내용 |
|------|------|
| 시뮬레이터 | NVIDIA Isaac Sim 4.5 / Isaac Lab 2.3.0 |
| RL 프레임워크 | skrl 1.4.3+ (PPO) |
| 로봇 자유도 | 9-DOF (허리 1 + 양팔 8) |
| 악기 수 | 8 (스네어, 플로어/미드/하이 톰, 하이햇, 라이드, 크래시 ×2) |
| 에피소드 | 5초 (300 step, 60Hz 정책) |

---

## 문서

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 코드 구조, 데이터 흐름, 설계 결정.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — 현재 baseline에 이르기까지의 시행착오 압축본.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — 학습 실행, 로그 위치, 알려진 사고 사례.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — 미해결 문제와 가설 백로그.

---

## 개발 환경

```
OS         : Ubuntu 22.04.5
GPU        : RTX 4070 SUPER
CUDA       : 12.8
Driver     : 580.x
Isaac Sim  : 4.5
Isaac Lab  : 2.3.0
Python     : 3.10
```

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