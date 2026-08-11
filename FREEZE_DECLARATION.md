# Design Freeze Declaration (pre-holdout)

작성: 2026-08-11. **이 문서 이후 holdout 평가 전까지 아래 항목은 변경 불가.**
변경이 불가피하면 사유·일자를 기록하고 holdout 평가를 다시 유효화해야 함(사실상 재실험).

## 1. 데이터 (configs/data.yaml, 2026-08-10 freeze)
- 분석 기간 2015-01-01~2017-08-15 (buffer 90일), zero-fill(첫 판매일 이후), 음수 클리핑, 지진 dummy(45일), 급여일 특징
- 시계열 선정: first_sale ≤ 2016-06-30, history ≥ 365d, nonzero ≥ 120d(최근 365d ≥ 60d), promo ≥ 10d
- 샘플: 3,000개, promo×CV 3×3 층화, seed 20260810
- 분할: 3 folds × (28일 window, 주간 origin 4개) + holdout(2017-07-18/25, 08-01/08 origins)
- OOF: 2016-03-01~2017-04-18 주간 origin, 월별 expanding 재학습; TCN(≤2016-12-27)/게이트(2017-01-03~) 분리; eval_input(2017-04-25~)

## 2. 모델
- Base: 전역 LightGBM (num_leaves 63, lr 0.05, ff 0.8, mdl 50, L1 목적), 월별 재학습 = 운영 base
- TCN: lookback 28, hidden 32, k3, dropout 0.2, lr 1e-3 (tune_tcn.py, gate_train 선택), Huber(scaled)
- 스케일: rolling_mean_28 + 1.0, clip ±10
- 게이트: 26특징(5그룹), MLP 64-32-1, bias init −2, 손실 = raw L1 + 0.1·mean(g) (λ는 gate_train WAPE로 선택)
- 상수 게이트 A1.5: c* = 0.00 (gate_train WAPE 튜닝 결과)
- TFT 베이스라인: hidden 64, heads 4, bf16, batch 1024, ≤20 epochs, ES patience 3, 폴드별 재학습
- TFT-base 변형: OOF는 단일 TFT(≤2016-02-29, 늦은 시작 137개 시계열 제외), eval은 폴드별 TFT

## 3. 평가
- 지표: WAPE(주), MAE/RMSE/RMSLE/MASE; regime 슬라이스 6종(post-promo 1–3일, 고변동 = train CV_28 상위 25%)
- 검정: series-level Wilcoxon(+Holm), bootstrap 2000 CI, win rate; 총 actual < 1 시계열 제외
- 운영비용: 1:1/2:1/3:1/5:1, 점예측 프록시 명시
- Oracle gate headroom + gate-benefit hit-rate

## 4. Holdout 규율
- 평가 명령: `python scripts/07_make_paper_tables.py --tag final --include-holdout` (+ TFT 통합본 별도)
- **실행 횟수: 1회 — 2026-08-11, `scripts/11_holdout_eval.py` (양 base 동시 평가). 이후 어떤 재튜닝·재평가도 금지.**
- Holdout 결과에 따른 어떤 재튜닝도 금지 (CLAUDE.md §18)

## 5. 검증 폴드에서 내린 설계 반복 이력 (논문 방법론 절에 기술할 것)
1. 스케일 분모 +1 floor (영수요 폭주 방지) — 2026-08-10
2. 게이트 손실을 scale-normalized Huber → raw L1 (평가지표 정렬) — 2026-08-10
3. 게이트 bias −2 초기화 (닫힘 시작) — 2026-08-10
4. λ=0.1 희소성 정규화 (gate_train WAPE 선택, 민감도 부록) — 2026-08-10
모두 folds 1–3 관찰에 근거, holdout 미접촉.
