# ARCHITECTURE

코드가 어떻게 구성돼 있고, 데이터가 어떻게 흐르며, 각 모듈이 왜 그 자리에 있는지 설명하는 문서.

**현재 코드 자체에 대한 일반적인 설명**은 코드의 docstring과 주석을 참조. 본 문서는 **신입이 코드만 읽어서는 알기 어려운 정보 (설계 의도, 모듈 간 관계, 자료구조 의미)** 를 다룬다.

> ⚠️ Observation 차원 구성, 보상 항목 같은 자료구조의 구체적인 형태는 연구가 진행되며 얼마든지 바뀔 수 있다. 본 문서는 현재 시점의 구조를 설명하며, 큰 그림과 모듈 책임 분할은 유지되지만 세부 수치는 코드를 우선 참고할 것.

---

## 디렉토리 구조

```
drum_robot/
├── assets/drum_robot/         로봇 URDF, USD, 형상 정보
├── tasks/drumrobot/           메인 RL 태스크
│   ├── drumrobot_env.py       환경 클래스 (DirectRLEnv)
│   ├── drumrobot_cfg.py       설정 (로봇, 물리, 보상 가중치)
│   ├── __init__.py            Gymnasium 환경 등록
│   ├── components/            도메인 로직 컴포넌트
│   │   ├── rds_initializer.py     MIDI → RDS 변환, 에피소드 악보 생성
│   │   ├── robot_initializer.py   초기 자세 생성 (IK 기반)
│   │   ├── visualizer.py          USD 마커, 드럼 색상 업데이트
│   │   ├── hit_detector.py        타격 판정 (접촉/속도/armed)
│   │   ├── observation.py         관측 벡터 구성 및 정규화
│   │   └── reward.py              보상 계산 (jit 컴파일된 함수 포함)
│   └── agents/
│       └── skrl_ppo_cfg.yaml      PPO 하이퍼파라미터
├── tasks/test_sac/            SAC 실험용 (남겨둠, 본 baseline과 무관)
├── utils/logger.py            학습 지표 콘솔 로깅
└── scripts/reinforcement_learning/skrl/
    ├── train.py
    └── play.py
```

### 모듈 책임 분할 원칙

`env`는 **얇은 오케스트레이터** 다. RL 인터페이스 (`_get_observations`, `_pre_physics_step`, `_get_dones`, `_get_rewards`, `_reset_idx`) 와 텐서 버퍼 관리만 담당하고, 도메인 로직은 모두 `components/` 의 모듈에 위임한다.

각 컴포넌트의 책임:

- **`rds_initializer`** — MIDI 파일 파싱, segment 분할, score 기반 sampling, 에피소드 시작 시 RDS 텐서 제공.
- **`robot_initializer`** — 양팔 드럼 조합 중 하나를 선택하고, 기하학적 IK로 초기 관절 각도 계산. 노이즈 추가.
- **`visualizer`** — USD prim 캐싱과 매 step 업데이트 (스틱 팁 마커, 드럼 색상).
- **`hit_detector`** — 매 step의 접촉/속도/armed 상태 판정, 윈도우 종료 시 성공/놓침/잘못 침 확정.
- **`observation`** — 94차원 관측 벡터 구성, 정규화 통계 관리, 다음 K개 hit 이벤트 추출.
- **`reward`** — phase별 motion shaping, goal 보상, 자동 가중치 등 보상 항 계산. 성능을 위해 일부 함수는 `@torch.jit.script`로 컴파일.

---

## 데이터 흐름

### 에피소드 리셋

```
_reset_idx(env_ids)
  │
  ├─ rds_initializer  → RDS 텐서 (T=300, M=8) 생성, env별 score 기반 sampling
  ├─ robot_initializer → 양팔 드럼 조합 선택 + IK로 초기 관절 각도 계산
  └─ 텐서 버퍼 초기화 (hit_armed, hit_per_arm, prev_tip_pos, prev_tip_vel ...)
```

### 매 step (60Hz 정책, 120Hz 물리, decimation=2)

```
_get_observations
  │
  ├─ 관절 위치/속도 (Isaac Lab의 robot.data 에서 추출)
  ├─ tip 위치 (FK; observation 모듈에서 계산)
  ├─ observation.get_next_hits(rds, steps) → 다음 K=3 이벤트
  └─ observation.normalize_and_pack → 94차원 벡터

_pre_physics_step(actions)
  │
  ├─ action_scale × π × dt 곱해 관절 각속도 명령으로 해석
  ├─ 다음 step의 목표 관절 위치 = 현재 + 명령
  └─ joint limit 안으로 clip

_apply_action  →  robot.set_joint_position_target(...)

_get_dones
  │
  ├─ hit_detector.detect → 접촉 / armed / hit_mask 갱신
  ├─ hit_detector.detect_wrong → 윈도우 밖 hit 감지
  ├─ hit_detector.finalize_targets → 윈도우 닫힌 노트의 성공/놓침 확정
  ├─ hit_detector.rearm → 충분히 들어올린 손은 armed 복귀
  └─ visualizer.update → 마커, 색상 갱신

_get_rewards
  │
  └─ reward.compute_rewards(...) → 단일 스칼라 보상 (정책 학습 신호)
       내부 항: success / time_accuracy / wrong / missed / progress
              / proximity / strike_phase / rearm_phase / penalties
```

---

## 핵심 자료구조

### RDS (Robotic Drum Score)

- **Shape**: `(num_envs, T, M)` — `T = 300` step, `M = 8` drum.
- **값**: 0 또는 1. 해당 시점에 해당 드럼을 쳐야 하는지의 binary mask.
- **현재**: binary. 향후 강도 (세게/약하게), 종류 (roll, ghost note) 등 정보가 추가될 가능성 있음.

### Observation (94차원)

| 항목 | 차원 | 비고 |
|------|------|------|
| 관절 위치 (정규화) | 9 | 제어 관절만 |
| 관절 속도 (스케일링) | 9 | |
| 양손 팁 위치 | 6 | FK로 계산 |
| 8개 드럼 위치 | 24 | 에피소드마다 노이즈 추가 |
| 다음 K=3 hit 이벤트 | 30 | (drum multi-hot 8 + 정규화 시간 1 + valid 1) × 3 |
| `hit_armed` (양팔 × 8드럼) | 16 | 현재 타격 가능 여부 |

> Observation은 연구 진행에 따라 차원과 항목이 바뀔 수 있다. 변경 시 `cfg`의 observation 차원과 `observation` 컴포넌트의 packing 로직을 함께 수정해야 한다.

### `hit_armed` 비트

- Shape: `(num_envs, 2 arms, 8 drums)`.
- 한 손이 여러 드럼에 동시에 armed될 수 있다. 양손이 같은 드럼에 동시 armed도 가능.
- 켜짐 / 꺼짐 조건:
  - 드럼 위 `rearm_height=0.18m` 이상 올라오면 → armed
  - 해당 드럼을 친 직후 → disarmed
- 정책 입장에서는 "지금 이 손-드럼 조합으로 인정되는 타격이 가능한가" 의 비트.

### `rds_visit`

- 윈도우 채점이 두 번 일어나지 않도록, 이미 처리된 RDS 셀을 표시하는 마스크.
- `hit_detector.finalize_targets` 가 윈도우 종료 시점에 한 번씩만 성공/놓침을 확정하기 위해 사용.

---

## 주요 설계 결정

### Phase × motion shaping을 환경 안에 둔다

타격 사이클의 의미를 보상이 담당한다 (`strike_phase`, `rearm_phase` 항). `hit_armed` 비트 + 팁 수직 속도 부호의 조합으로 "내려치는 단계인지 / 올라가는 단계인지" 가 정의된다.

명시적인 FSM 상태 변수는 두지 않는다 — 과거에 4단계 FSM (IDLE/LIFT/DESCEND/RETURN) 을 환경에 박아넣었던 적이 있는데, RDS의 노트 시점 자체가 이미 시간 구조를 제공하므로 별도 FSM이 불필요하다고 판단해 1비트 표현으로 회귀.

### 양손 할당은 환경의 보상이 결정

정책 출력은 9-DOF 관절 명령 하나뿐이고, "왼손이 어느 드럼, 오른손이 어느 드럼" 의 배정 자체는 학습되지 않는다. 보상이 매 step 거리 기반으로 가까운 손을 자동 매칭하고, 그 손에만 phase 보상을 적용한다.

이 구조는 "정책이 양손 배정도 학습해야 한다" 는 가설을 미루기 위한 선택이다. 가능한 가설 확장은 ROADMAP H3 참조.

### 액션은 위치 증분이 아니라 각속도 명령

정책 출력 × π × dt 만큼 현재 관절 위치에 더해 다음 목표 위치를 만든다. 한 step당 변화량은 위치 증분 방식과 유사하지만, 정책이 학습하는 표현이 "다음 자세" 가 아니라 "지금 어느 방향으로 얼마나 빠르게" 가 된다.

실제 드럼 로봇 하드웨어가 보통 속도 명령으로 제어되므로, sim-to-real 시점의 표현 일관성을 위한 선택.

### Observation은 압축이 답일 수 있다

이전 구조에서는 미래 30 step × 8 drum = 240차원 sparse one-hot을 그대로 정책에 넣었다. 다음 K=3 이벤트의 (drum + 시간 + valid) = 30차원 표현으로 압축하자 학습 성능이 4배 좋아짐.

MLP가 추출하기 좋은 구조로 정리해 넣는 것이, 정보를 다 넣는 것보다 중요하다는 핵심 통찰. EXPERIMENTS 문서 참조.

### `@torch.jit.script` 보상 함수

`reward.py`의 일부 함수는 jit 컴파일되어 있다. 대량 환경에서 매 step 호출되는 함수의 성능을 위함. 수정할 때는 일반 Python 함수로 먼저 검증한 뒤 jit 데코레이터를 다시 붙이는 워크플로가 안전하다. jit 함수는 디버깅이 어렵다.

---

## 외부 의존성

### Isaac Lab / Isaac Sim

- 본 코드는 Isaac Lab의 **외부 extension** 으로 동작한다. Isaac Lab 공식 레포에 `source/extensions/drum_robot/` 으로 배치된다.
- 검증된 버전: Isaac Sim 4.5, Isaac Lab 2.3.0.
- Isaac Lab 버전 업그레이드 시 동작 보장 없음. 신중하게.

### skrl

- PPO 사용. 설정은 `agents/skrl_ppo_cfg.yaml`.
- skrl 자체는 1.4.3 이상. 큰 변경 없이 동작.
- 본 프로젝트가 skrl을 선택한 이유는 강한 설계상 이유가 있어서가 아니라 **예제를 따라간 결과** 다. 필요하면 다른 라이브러리로 옮길 수 있음.

### DirectRLEnv

- Isaac Lab의 `ManagerBasedRLEnv` 가 아니라 `DirectRLEnv` 를 상속. 예제 따라간 선택이지만, 도메인 로직이 phase·timing 등 환경 내부 상태에 강하게 묶여 있어서 manager 기반보다는 direct 쪽이 자연스러운 구조이기도 함.

### URDF / USD

- 로봇 형상은 SolidWorks에서 설계 → URDF로 export → Isaac Sim에서 USD로 변환.
- 현재 9-DOF 형상에 맞춰져 있음. 머리/페달 4-DOF를 추가하려면 위 파이프라인을 다시 거쳐야 함.
- 결과 파일: `assets/drum_robot/usd/drum_robot.usd` (학습용).

### MIDI 데이터셋

- 실제 드럼 MIDI 파일을 segment로 분할해 사용한다.
- 데이터셋은 별도 보관 중이며 git 저장소에는 포함되지 않음 (용량 문제).

---

## 모듈 간 의존성 그림

```
                       drumrobot_cfg
                            │
                            ▼
                      drumrobot_env  ───┐
                       (얇은 셸)        │
                            │           │
        ┌───────────┬───────┼───────────┼──────────┬──────────┐
        ▼           ▼       ▼           ▼          ▼          ▼
       rds      robot   visualizer  observation  reward  hit_detector
   initializer initializer                                  │
                                                            │
                          ┌─────────────────────────────────┘
                          ▼
                       utils/logger
```

- 각 컴포넌트는 env로부터 필요한 텐서 (robot.data, rds, hit_armed 등) 와 cfg를 받아 동작.
- 컴포넌트 사이 직접 참조는 없음. 모든 의존성은 env를 통과한다.
- `utils/logger` 는 콘솔에 학습 지표를 띄우는 용도. 텐서보드는 skrl이 자동 처리.