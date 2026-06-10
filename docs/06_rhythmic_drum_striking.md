# Rhythmic Drum Striking

> **한 줄 요약**: 양팔 9-DOF 로봇의 양손 스틱 끝(팁)이 임의로 주어진 3D 목표 점 두 개로 수렴하도록 학습시킨 첫 번째 실험.
>
> **Status**: 부분 성공 (체크포인트 5M steps)
>
> **시기**: 2026-05

---

## 1. 동기 (Why)
왜 이 task를 만들었는가. 이전 단계에서 무엇이 부족했기에 이걸 시도했는가.

## 2. 문제 정의 (What)
- 목표: 구체적으로 무엇을 학습시키나
- 입력 (관측): 어떤 정보를 주는가
- 출력 (행동): 무엇을 결정하나
- 성공 기준: 무엇을 보고 "됐다"고 판단했나

## 3. 설계 (How)
- 관측 벡터 구성과 그 이유
- 행동 공간 구성과 그 이유
- 보상 함수 항목들과 각각의 의도
- 주요 하이퍼파라미터

## 4. 결과
- 정량적 결과 (성공률, 학습 곡선, 체크포인트 step 수)
- 정성적 결과 (영상이 있다면 링크, 또는 텍스트 설명)
reward=0.178 | proximity=0.016 | progress(x100)=0.287 | upward=0.200 | downward=0.138 | action_l2=5.793 | joint_vel_l2=3.755 | limit_pen(x100)=0.007 | tip_limit_pen=0.019 | under_drum_pen=0.132 | success_rate=0.593 | wrong_rate=0.189 | miss_rate=0.407 | snare_success_rate=0.665 | snare_wrong_rate=0.204 | snare_miss_rate=0.335 | floor_success_rate=0.668 | floor_wrong_rate=0.206 | floor_miss_rate=0.332 | mid_success_rate=0.702 | mid_wrong_rate=0.363 | mid_miss_rate=0.298 | high_success_rate=0.623 | high_wrong_rate=0.341 | high_miss_rate=0.377 | hihat_success_rate=0.626 | hihat_wrong_rate=0.102 | hihat_miss_rate=0.374 | ride_success_rate=0.416 | ride_wrong_rate=0.104 | ride_miss_rate=0.584 | crash1_success_rate=0.551 | crash1_wrong_rate=0.082 | crash1_miss_rate=0.449 | crash2_success_rate=0.490 | crash2_wrong_rate=0.108 | crash2_miss_rate=0.510

## 5. 무엇이 잘 됐는가
- 의도한 대로 동작한 부분
- 예상보다 좋았던 부분

## 6. 한계와 막힌 점
- 어디서 막혔는가
- 왜 막혔다고 추정하는가
- 우회/해결 시도와 결과

## 7. 다음 단계로 넘어간 이유
- 이 실험으로 충분하지 않다고 판단한 근거
- 다음 task가 어떻게 이 한계를 해결하려 했는가

## 8. 회고 / 배운 점
- 이 실험에서 얻은 인사이트
- 다시 한다면 어떻게 다르게 할 것인가