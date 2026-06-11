# Drum Striking

> **한 줄 요약**: 보상을 IDLE → LIFT → DESCEND → RETURN의 phase FSM에 묶어, 같은 변수(xy 거리, z 오차, vz)가 현재 phase에 따라 다른 의미로 작동하도록 재구조화한 실험. 짧은 학습으로 실제 "타격 동작"을 학습시킴.
>
> **Status**: 부분 성공 (체크포인트 100K steps)
>
> **시기**: 2026-03

---

## 1. 동기 (Why)

이전 실험(`dr`, 02)에서 학습된 정책은 **드럼에 도달해 머무는 동작**이었지 **타격 동작**이 아니었다. `first_hit ≈ 0`이 그 증거였고, 원인은 "dense한 위치 보상 + sparse한 hit 보너스"라는 비대칭 구조에 있다고 진단했다.

진단을 한 줄로 정리하면:

> 정적인 보상 함수로 동적인 동작을 학습시킬 수 없다.

따라서 이 실험의 가설은 다음과 같다:

> **타격 동작 자체를 phase로 쪼개고, 각 phase에서 적절한 motion에 dense 보상을 주면, 정책은 정지가 아닌 사이클을 학습할 것이다.**

이를 위해 4-state FSM(IDLE/LIFT/DESCEND/RETURN)을 환경 측에 두고, 각 phase에서 어떤 행동이 좋은지를 보상이 직접 명시한다.

---

## 2. 문제 정의 (What)

- **목표**: 양손이 각각 할당된 드럼에 대해 IDLE → LIFT → DESCEND → RETURN 사이클을 완주
- **사이클 정의**:
  - **IDLE**: 시작 상태. 드럼 xy 범위 안에 들어가면 LIFT로 전이
  - **LIFT**: 드럼 위 일정 높이까지 들어올림 (xy 가까이 유지 + z 들어올림)
  - **DESCEND**: 임팩트 속도로 z=0 (드럼 표면) 통과
  - **RETURN**: 다시 들어올려서 드럼 위로 복귀
- **성공**: RETURN 단계까지 완주 (= 양손 각각이 들어올림 → 임팩트 → 복귀까지 완료)
- **실패**: 각 phase에서 fail 조건 충족 (xy 범위 이탈, z를 너무 빠르게 통과하는 miss 등)

`dr`이 "한 번 hit"을 성공으로 봤다면, `ds`는 **"타격 모션 전체 사이클의 완주"**를 성공으로 본다.

---

## 3. 설계 (How)

### `dr` 대비 변경점 한눈에

| 항목 | `dr_cfg` / `dr_env` | `ds_cfg` / `ds_env` |
|---|---|---|
| 관측 차원 | 50 | **58** |
| hit 판정 | `impact_armed` 1비트 잠금 + rebound | **4-state FSM**(IDLE/LIFT/DESCEND/RETURN) |
| 보상 구조 | 9항 (xy, z, exp_xy, exp_z, down, up, action, joint_vel, limit) | **10항 + 성공/실패 항** (phase별 마스킹) |
| dense 보상 | 위치(xy/z) 항상 켜짐 | **phase mask에 의해 항상 정확히 한 묶음만 켜짐** |
| 성공 보상 | `first_hit_bonus = 50` | `first_success = 15`, `episode_success = 15` |
| 실패 패널티 | 없음 | `first_fail = -5` |
| exp shaping 강도 | `k_xy=8`, `k_z=10` | **`k_xy=80`, `k_z=120`** (DESCEND에서만 사용) |
| 학습 step | 700K | **100K로 수렴** |

### 관측 (58차원)

`dr`의 50차원에서 다음이 바뀜:

- 빠진 것: `impact_armed_L/R` (1차원 × 2 = 2차원)
- 추가된 것: `state_one_hot_L`, `state_one_hot_R` (5차원 × 2 = 10차원, IDLE/LIFT/DESCEND/RETURN + 여유 1)

순 증가 +8차원. 핵심은 **정책이 현재 자기가 어느 phase에 있는지를 명시적으로 본다**는 점이다. 같은 위치/속도에서도 phase에 따라 다른 행동을 해야 하므로, phase 관측 없이는 학습 자체가 불가능하다 (동일 입력에 다른 출력을 요구하게 됨).

### 핵심 아이디어: phase × motion shaping

`compute_reward_terms`에서 각 항이 다음 구조를 가진다.

```
phase_term = phase_mask × motion_shape
```

여기서 `phase_mask`는 "현재 그 phase인가"를 0/1로, `motion_shape`은 해당 phase에서 좋은 움직임을 정의한다. 같은 변수(예: vz)가 phase에 따라 다른 기여를 한다.

| Phase | xy 항 | z 항 | velocity 항 |
|---|---|---|---|
| IDLE | `-xy_dist` (드럼 가까이) | — | — |
| LIFT | `-xy_dist` (xy는 유지) | `+clamp(z_err / 0.2)` (위로 들어올림) | `+clamp(vz / 0.05)` (상향 속도) |
| DESCEND | `+exp(-80·xy_dist)` (매우 정밀) | `+exp(-120·z_err)` (매우 정밀) | `+clamp(-vz / 0.5)` (강한 하향 속도) |
| RETURN | `-xy_dist` | `+clamp(z_err / 0.1)` (다시 위로) | `+clamp(vz / 0.3)` (상향) |

**여기서 가장 중요한 설계 선택은 DESCEND의 exp_k**이다. `k_xy=80`, `k_z=120`이라는 매우 가파른 exp는 **드럼 표면 바로 위 좁은 영역에서만 큰 값**을 가진다. 즉 정책이 임팩트 지점에 정확히 들어왔을 때만 큰 보상이 발생한다. `dr`의 `k_xy=8`, `k_z=10`은 멀리서도 점진적 유도를 했지만, `ds`에서는 IDLE/LIFT에서 거리 기반 유도를 끝내고 DESCEND는 정밀 보상만 가진다.

### 성공/실패 보상

- `w_first_success = 15.0` × `(left_first_success + right_first_success)` — 한 손이 처음으로 사이클을 완주한 step
- `w_episode_success = 15.0` × `episode_success` — 양손 모두 완주 중인 동안 매 step
- `w_first_fail = 5.0` × `(left_first_fail + right_first_fail)` — 처음 실패한 step에 음의 보상
- `w_success_motion = 0.02` × `success_motion_pen` — 이미 성공한 손은 가만히 있도록 (vel L2 패널티)

`dr`은 `first_hit_bonus = 50` 하나에 보상을 집중시켰지만, `ds`는 **성공·실패 양쪽에 작은 신호를 동시에 둔다**. 실패에 음의 신호를 두는 것은 `pr`의 `died`와 비슷한 역할이지만, 에피소드를 종료시키지는 않고 그 손만 잠금(`fail`)된 상태로 둔다.

### FSM 전이 로직 (`_check_hitting`)

상태 전이는 환경이 결정한다. 각 phase의 `_should_exit_*` 함수가 done/fail을 반환:

- IDLE → LIFT: xy 거리가 `idle_xy_radius=0.29` 이내
- LIFT → DESCEND: xy 거리 ≤ 0.24 ∧ z 높이 ≥ 0.2 m
- DESCEND → RETURN: xy 거리 ≤ 0.1 ∧ z 부호 전환(드럼 표면 통과) ∧ 하향 속도 ≥ 0.5 m/s
- RETURN → 성공: xy 거리 ≤ 0.15 ∧ z 높이 ≥ 0.1 m

각 단계에는 fail 반경도 함께 정의되어 있어, xy가 너무 멀어지면 fail로 잠금.

### 환경 설정

| 항목 | 값 | 비고 |
|---|---|---|
| 병렬 환경 | 128 (cfg) | |
| 에피소드 길이 | 5s | |
| action_scale | 0.03 | |
| 학습 step | **100K** | `dr`의 1/7로 수렴 |

---

## 4. 결과

![GIF](./gif/03.gif)

보상 그래프 첨부 예정

```
reward=0.665 | idle=-0.203 | lift=0.395 | descend=0.605 | return=0.307
success_motion_pen=0.106 | action_l2=7.560 | joint_vel_l2=1.584 | limit_pen=0.000
left_success_rate=0.964 | right_success_rate=0.986 | episode_success_rate=0.959
left_fail_rate=0.025 | right_fail_rate=0.020
```

메트릭이 결과를 말해준다:

- **성공률 0.96 이상**: 양손 각각 96~98%로 사이클을 완주. 에피소드 단위 성공도 95.9%.
- **실패율 2~3%**: 실패하는 경우도 있지만 매우 낮음.
- **phase별 보상이 모두 양수**: IDLE만 음수(-0.203, 거리 기반)이고 LIFT(0.395) / DESCEND(0.605) / RETURN(0.307)이 모두 안정적으로 활성화. 즉 정책이 모든 phase를 거치고 있다.
- **100K step으로 수렴**: `dr`의 700K 대비 1/7. 같은 자원으로 훨씬 빠른 학습.

겉으로도 **들어올렸다가 내려치고 다시 들어 올리는** 실제 타격 모션이 나옴.

---

## 5. 무엇이 잘 됐는가

- **가설 검증**: "phase × motion shaping" 구조로 보상을 재구조화하면 정적 수렴 문제가 풀린다. `dr`의 `first_hit ≈ 0` 문제가 `ds`에서는 96%+ 성공으로 변환.
- **빠른 수렴**: 100K step으로 안정 정책. dense한 학습 신호가 항상 phase 한 묶음에만 켜져 있어, 정책이 "지금 무엇을 해야 하는가"의 그래디언트를 명확히 받는다.
- **phase 관측의 효과**: state one-hot을 정책에 노출시키니 같은 위치에서도 phase에 따라 다른 행동이 가능. phase 관측 없이는 학습 자체가 불가능했을 것.
- **success/fail 양쪽 신호**: `first_fail = -5`는 정책이 부주의한 이탈을 회피하도록 만들고, `first_success / episode_success`는 사이클 완주를 직접 보상. 두 방향이 정책을 "사이클 안에서 벗어나지 말라"로 묶음.
- **양팔 협응**: 한 손이 fail/success로 잠겨도 다른 손이 계속 사이클을 돌아 episode_success까지 가는 정책이 나옴.

---

## 6. 한계와 막힌 점

- **FSM이 환경 측에 하드코딩됨**. IDLE/LIFT/DESCEND/RETURN의 경계 파라미터(`*_xy_radius`, `*_z_min_above_drum`, `descend_min_impact_velocity` 등)가 cfg에 12개 이상. 새 task로 일반화하려면 이걸 매번 다시 설계해야 함. 즉 "보상 함수에 사람 지식을 박은" 셈이고, 학습이 한 일은 phase 안에서의 motion 최적화에 한정된다.
- **물리적 접촉 없이 z-crossing만으로 성공 판정**. 실제 드럼 메쉬에 닿지 않아도 z 부호가 바뀌면 DESCEND 통과로 본다. 시뮬레이션에서는 충분하지만, 실제 하드웨어로 옮길 때 "친 것처럼 보이는데 안 친 동작"이 나올 위험.
- **하이퍼파라미터 폭증**. `dr`의 보상 항은 9개였는데 `ds`는 phase별 항 10개 + 성공 2개 + 실패 1개 + 정규화 4개로 17개. 가중치 튜닝 부담이 커짐. 다행히 작은 값으로 잘게 쪼개고 성공에 큰 가중을 두는 패턴이 안정적이었음.
- **타이밍/리듬 요소가 빠짐**. 사이클 자체는 학습되지만, "언제" 치는지는 정책이 자유롭게 결정함. 실제 드럼 연주는 악보 타이밍에 묶여야 하므로 이건 다음 task의 몫.
- **DESCEND의 `k_xy=80`, `k_z=120`은 매우 좁은 영역**에서만 보상을 준다. 정책이 그 영역에 도달하지 못하면 DESCEND 보상이 거의 0. 다행히 LIFT가 DESCEND 직전 위치까지 정책을 끌어다 놓아주기 때문에 작동했지만, phase 간 인계가 잘 안 되는 cfg에서는 학습이 막힐 수 있다.

---

## 7. 다음 단계로 넘어간 이유

**단발 타격 동작은 학습 가능함을 확인했다.** 그 다음 자연스러운 단계는 **시간/리듬 요소**다. 드럼 연주는 단순히 "친다"가 아니라 "정해진 시점에 친다"이므로, 사이클 완주에 타이밍 조건을 추가해야 한다.

다음 실험(`rdr`, `rds`, `rrdr`, 04~06): 리듬 타격. 외부에서 주어진 타이밍 신호에 맞춰 사이클이 시작·완료되도록 보상을 시간 의존적으로 확장.

---

## 8. 회고 / 배운 점

- **보상 = 행동 단계 마스킹**이라는 패턴: phase 변수를 곱하면 같은 motion 변수(xy 거리, vz 등)가 단계마다 다른 의미로 작동한다. 이 패턴은 "도달이 아닌 동작"을 학습시키는 데 매우 효과적.
- **phase는 반드시 관측에도 들어가야 한다**: 보상에만 phase 마스킹을 걸고 정책에 phase를 알려주지 않으면, 동일 입력에 다른 출력을 요구하게 되어 학습이 발산한다. 보상이 phase 의존적이면 관측도 phase 의존적이어야 함.
- **dense vs sparse의 비대칭은 가중치가 아니라 구조로 해결한다**: `dr`에서 `down/up` 가중을 올리거나 `xy/z`를 줄이는 식의 튜닝으로는 이 문제가 해결되지 않았을 가능성이 높다. 보상의 시간 구조 자체를 바꿔야 했음.
- **작은 가중 × 많은 항 + 성공에 큰 가중**: `dr`의 큰 단일 보너스(`50.0`)보다 `ds`의 분산된 가중(0.01~0.2) + 성공 15.0이 더 안정적이었다. dense 신호의 분산이 학습 안정성에 기여.
- **exp shaping의 k는 phase에 맞춰 선택**: 멀리서 유도하려면 작은 k(8~10), 정밀 임팩트만 보상하려면 큰 k(80~120). 한 가지 k로 둘 다 하려 들면 둘 다 어정쩡해진다. `dr`은 한 가지 k였고 `ds`는 phase별 분리.
- **사람 지식을 보상에 박는 것의 trade-off**: FSM phase 정의는 분명히 사람이 design한 prior이고, 그만큼 학습이 일반화되지 않는다. 그러나 "어떤 사람 지식을 어디까지 박을지"의 트레이드오프 자체가 RL 실험 설계의 중심 변수임을 체감.