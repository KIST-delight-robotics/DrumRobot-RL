# 🥁 DrumRobot — 강화학습 기반 드럼 연주 로봇

> **Isaac Lab + skrl(PPO)** 를 사용하여 양팔 로봇이 악보를 보고 드럼을 연주하도록 훈련하는 강화학습 프로젝트

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [개발 환경](#2-개발-환경)
3. [환경 구축 방법](#3-환경-구축-방법)
4. [프로젝트 구조](#4-프로젝트-구조)
5. [핵심 개념](#5-핵심-개념)
   - [RDS (Robotic Drum Score)](#rds-robotic-drum-score)
   - [관측 공간 (Observation Space)](#관측-공간-observation-space)
   - [행동 공간 (Action Space)](#행동-공간-action-space)
   - [보상 함수 (Reward Function)](#보상-함수-reward-function)
6. [환경 실행 흐름](#6-환경-실행-흐름)
7. [모듈 상세 설명](#7-모듈-상세-설명)
8. [실행 방법](#8-실행-방법)
9. [주요 설정값](#9-주요-설정값)

---

## 1. 프로젝트 개요

9-DOF 양팔 로봇이 **MIDI 악보**를 입력으로 받아 드럼을 정확한 타이밍에 타격하도록 훈련합니다.

| 항목 | 내용 |
|------|------|
| 시뮬레이터 | NVIDIA Isaac Sim 4.5 (Isaac Lab 2.3.0) |
| RL 프레임워크 | skrl 1.4.3+ (PPO 알고리즘) |
| 병렬 환경 수 | 최대 4096개 (기본 128개) |
| 악기 수 | 8개 (스네어, 플로어탐, 미드탐, 하이탐, 하이햇, 라이드, 크래시 ×2) |
| 로봇 자유도 | 9-DOF (허리 1 + 왼팔 4 + 오른팔 4) |
| 에피소드 길이 | 5초 (dt=1/60s 기준 300 스텝) |

---

## 2. 개발 환경

```
OS      : Ubuntu 22.04.5
GPU     : RTX 4070 SUPER
CUDA    : 12.8
Driver  : 580.x
Isaac Sim : 4.5
Isaac Lab : 2.3.0
Python  : 3.10
PhysX   : GPU (cuda:0)
```

---

## 3. 환경 구축 방법

본 프로젝트는 Isaac Lab 공식 레포에 **외부 extension** 형태로 얹어 사용합니다.

### 3.1 Isaac Lab 공식 레포 클론

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### 3.2 drum_robot extension 배치

본 레포의 코드를 Isaac Lab의 `source/extensions/` 아래 `drum_robot` 폴더로 넣습니다.

```bash
# IsaacLab 루트에서
git clone https://github.com/KIST-delight-robotics/DrumRobot-RL.git source/extensions/drum_robot
```

결과적으로 아래와 같은 구조가 됩니다.

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

- `extension.toml` → Isaac Sim(Kit)이 `drum_robot`을 extension으로 인식하게 해줍니다.
- `pyproject.toml` → Isaac Lab 설치 과정에서 `pip install -e .`로 파이썬 패키지로도 잡히게 해줍니다.

### 3.3 실행 위치

학습/추론은 모두 Isaac Lab 공식 런처인 `isaaclab.sh`로 실행합니다. 이 스크립트가 Isaac Sim의 Python 인터프리터와 환경변수를 잡아줍니다. 이후 본 README의 모든 실행 명령은 **`drum_robot/` 디렉터리 기준**입니다.

```bash
cd source/extensions/drum_robot
```

---

## 4. 프로젝트 구조

```
drum_robot/
├── tasks/drumrobot/
│   ├── drumrobot_env.py       # 메인 환경 클래스 (DirectRLEnv 상속)
│   ├── drumrobot_cfg.py       # 환경 설정 (로봇, 물리, 보상 파라미터)
│   ├── __init__.py            # Gymnasium 환경 등록
│   ├── components/
│   │   ├── rds_initializer.py # MIDI → RDS 변환 및 에피소드 악보 생성
│   │   ├── robot_initializer.py  # 로봇 초기 자세 생성 (IK 기반)
│   │   └── visualizer.py     # Isaac Sim USD 시각화 (팁 마커, 드럼 색상)
│   └── agents/
│       └── skrl_ppo_cfg.yaml  # PPO 하이퍼파라미터 설정
├── utils/
│   └── logger.py              # 학습 중 지표 로깅
└── scripts/
    └── reinforcement_learning/skrl/
        ├── train.py           # 학습 실행 스크립트
        └── play.py            # 체크포인트 추론 스크립트
```

---

## 5. 핵심 개념

### RDS (Robotic Drum Score)

**RDS**는 MIDI 악보를 강화학습에서 사용할 수 있는 텐서 형식으로 변환한 것입니다.

```
shape: (T, M)
  T = 에피소드 스텝 수 (300)
  M = 악기 수 (8)
  값 = 0 또는 1 (해당 스텝에 해당 악기를 쳐야 하는지 여부)
```

- MIDI 파일을 1마디 단위로 잘라 데이터셋을 구성합니다.
- 에피소드마다 **MIDI 기반** 또는 **랜덤 생성** 방식으로 악보를 선택합니다.
- `slow_factor=1.5` 를 적용하여 악보를 느리게 재생합니다 (학습 난이도 조절).

### 관측 공간 (Observation Space)

**총 94차원**, 모두 정규화하여 입력합니다.

| 항목 | 차원 | 설명 |
|------|------|------|
| 관절 위치 | 9 | 제어 관절의 현재 각도 (정규화) |
| 관절 속도 | 9 | 제어 관절의 현재 각속도 (스케일링) |
| 팁 위치 | 6 | 왼손/오른손 스틱 끝 위치 (3D × 2) |
| 악기 위치 | 24 | 8개 악기의 3D 좌표 |
| 다음 타격 이벤트 | 30 | 가장 가까운 3개 이벤트 (드럼 8 + 정규화 시간 1 + 유효 플래그 1) × 3 |
| 타격 준비 상태 | 16 | 양팔 × 8악기, 현재 타격 가능 여부 (hit_armed) |

### 행동 공간 (Action Space)

**9차원 연속 행동** `[-1, 1]`

각 관절에 대한 **속도 명령**으로, 현재 위치에 `action × action_scale × dt` 를 더한 위치를 목표로 설정합니다.

```python
target_pos = current_pos + action × π × dt
```

관절 제한 범위 내로 클리핑됩니다.

### 보상 함수 (Reward Function)

보상은 크게 **목표 달성**, **근접 유도**, **패널티** 세 그룹으로 구성됩니다.

#### 목표 달성 보상
| 항목 | 가중치 | 설명 |
|------|--------|------|
| `success_reward` | +1.5 | 정확한 타이밍 내 타격 성공 |
| `time_accuracy_reward` | +1.0 | 타이밍 정확도 `exp(-k × time_error)` |
| `wrong_cost` | -1.0 | 목표 없는 악기 타격 |
| `missed_cost` | -0.8 | 타격 윈도우 내 미타격 |

#### 근접 유도 보상 (Shaping)
| 항목 | 가중치 | 설명 |
|------|--------|------|
| `progress_reward` | +4.0 | 목표 악기와의 거리 감소량 |
| `proximity_cost` | -1.5 | 목표까지 현재 거리 (타이밍 가중) |
| `upward_reward` | +0.30 | 타격 후 팔 올리기 |
| `downward_reward` | +0.25 | 타격 직전 팔 내리기 |

#### 패널티
| 항목 | 가중치 | 설명 |
|------|--------|------|
| `action_l2` | -0.0007 | 과도한 행동 억제 |
| `joint_vel_l2` | -0.0005 | 관절 속도 억제 |
| `limit_pen` | -0.5 | 관절 제한 근접 패널티 |
| `tip_limit_pen` | -0.15 | 스틱 끝이 허용 범위 이탈 |
| `under_drum_pen` | -0.12 | 드럼 아래로 팁 진입 |

---

## 6. 환경 실행 흐름

```
초기화
  └─ _setup_scene()      로봇, 지면, 조명 생성
  └─ __init__()          공간 정의, 버퍼 할당, 컴포넌트 초기화

에피소드 리셋
  └─ _reset_idx()
       ├─ RDS 생성         (rds_initializer)
       ├─ 로봇 자세 설정   (robot_initializer, IK 풀이)
       └─ 텐서 버퍼 초기화

매 스텝 (decimation=2, 즉 120Hz 물리 / 60Hz 정책)
  ├─ _get_observations()  관측값 수집 및 정규화
  ├─ _pre_physics_step()  행동 → 목표 관절 위치 변환
  ├─ _apply_action()      시뮬레이터에 명령 전달  ×2
  ├─ _get_dones()
  │    ├─ 팁 위치/속도 계산
  │    ├─ 접촉 감지
  │    ├─ 타격 판정 (hit_armed + 하방 속도 + 접촉)
  │    ├─ 잘못된 타격 / 성공 / 미스 판정
  │    └─ 시각화 업데이트
  └─ _get_rewards()       보상 계산 및 로그 출력
```

---

## 7. 모듈 상세 설명

### `drumrobot_env.py` — 메인 환경

- `DirectRLEnv`를 상속하며 Isaac Lab의 RL 루프에 통합됩니다.
- 타격 판정은 `hit_window_step=10` (±10 스텝) 윈도우 내에서 이루어집니다.
- 타격 후 스틱이 `rearm_height=0.18m` 이상 올라와야 다음 타격이 가능합니다 (`hit_armed` 메커니즘).
- `@torch.jit.script` 로 보상 계산 함수를 컴파일하여 성능을 최적화합니다.

### `rds_initializer.py` — MIDI 파서 및 악보 생성기

- `mido` 라이브러리로 MIDI를 파싱합니다.
- 1마디 단위로 잘라 RDS 텐서 데이터셋을 구성합니다.
- 각 RDS에 **난이도 점수**를 계산 (사용 악기 수, 전환 횟수, 스네어 비율)하여 커리큘럼 샘플링에 활용합니다.
- 리셋 시 `score_ratio` 파라미터로 MIDI 기반 vs 랜덤 악보 비율을 조정할 수 있습니다.

### `robot_initializer.py` — 초기 자세 생성

- **기하학적 IK(Inverse Kinematics)** 를 직접 구현하여 드럼 위 특정 위치에 팔 끝이 오도록 초기 자세를 결정합니다.
- 미리 정의된 양팔 드럼 조합 36가지 중 랜덤하게 선택합니다.
- 에피소드마다 `joint_noise_scale=5°` 범위의 노이즈를 추가합니다.

### `visualizer.py` — Isaac Sim USD 시각화

- 스틱 끝(팁) 위치를 빨간 구체 마커로 표시합니다.
- 드럼을 원기둥으로 표시하며, 타격 임박 시 색상이 변합니다:
  - 🟡 **노란색** — 곧 쳐야 하는 드럼 (hit window 이내)
  - 🟫 **어두운 노란색** — 가까운 미래에 쳐야 하는 드럼
  - ⬜ **회색** — 해당 없음
- USD TranslateOp와 DisplayColor Primvar를 캐싱하여 매 스텝 빠르게 업데이트합니다.

### `logger.py` — 학습 지표 로거

- `interval=10000` 스텝마다 각 보상 항목과 악기별 성공/오타/미스율을 출력합니다.
- 확률 기반 지표(`add_probability`)는 분자/분모를 누적하여 정확한 비율을 계산합니다.

### `skrl_ppo_cfg.yaml` — PPO 설정

```yaml
네트워크: [256, 160, 128] ELU, Gaussian Policy
롤아웃: 64 스텝
학습률: 1e-4 (KL Adaptive LR)
할인율: 0.99, λ=0.95
보상 스케일: 0.1
총 타임스텝: 100,000
```

---

## 8. 실행 방법

> 모든 명령은 `source/extensions/drum_robot/` 디렉터리에서 실행합니다.
> `../../../isaaclab.sh` 는 IsaacLab 루트의 공식 런처를 가리킵니다.

### 학습

```bash
../../../isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-drum_score-Direct-v0 \
  --num_envs=4096 \
  --headless
```

### 체크포인트 이어서 학습

```bash
../../../isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task=DrumRobot-drum_score-Direct-v0 \
  --num_envs=4096 \
  --headless \
  --checkpoint="logs/skrl/drum_robot/.../checkpoints/agent_100000.pt"
```

### 추론 (시각화 포함)

```bash
../../../isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task=DrumRobot-drum_score-Direct-v0 \
  --num_envs=1 \
  --checkpoint="logs/skrl/drum_robot/.../checkpoints/agent_100000.pt"
```

---

## 9. 주요 설정값

> `drumrobot_cfg.py`에서 수정 가능한 핵심 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `episode_length_s` | 5.0s | 에피소드 길이 |
| `decimation` | 2 | 정책 업데이트 주기 (2 물리 스텝당 1 정책 스텝) |
| `action_scale` | π rad/s | 관절 속도 스케일 |
| `hit_window_step` | 10 | 타격 허용 윈도우 (±10 스텝 = ±167ms) |
| `drum_xy_radius` | 0.13m | 드럼 타격 유효 반경 |
| `drum_z_range` | 0.07m | 드럼 타격 유효 높이 범위 |
| `min_impact_velocity` | 0.2 m/s | 최소 타격 인정 속도 |
| `rearm_height` | 0.18m | 재타격 허용 높이 |
| `inst_noise_scale` | 0.02m | 드럼 위치 노이즈 |
| `max_lookahead_time` | 1.0s | 미래 타격 관측 범위 |
| `num_hits` | 3 | 관측하는 미래 타격 이벤트 수 |