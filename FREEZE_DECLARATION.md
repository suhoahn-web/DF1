# Design Freeze Declaration (pre-holdout)

Written: 2026-08-11. Nothing below may change between this declaration and the
holdout evaluation. If a change becomes unavoidable, the reason and date must be
recorded here and the holdout evaluation is invalidated (effectively a re-run of
the study).

## 1. Data (configs/data.yaml, frozen 2026-08-10)
- Analysis period 2015-01-01 to 2017-08-15 (90-day feature buffer); zero-filling
  from each series' first observed sale; negative sales clipped; 45-day earthquake
  dummy window; payday features (15th and last day of month)
- Series eligibility: first sale ≤ 2016-06-30, history ≥ 365 days, non-zero days
  ≥ 120 total (≥ 60 in the last 365), promotion days ≥ 10
- Sample: 3,000 series, stratified by promotion-share × CV terciles, seed 20260810
- Splits: 3 development folds × (28-day window, 4 weekly origins) + holdout
  (origins 2017-07-18/25, 08-01/08)
- OOF: weekly origins 2016-03-01 to 2017-04-18, monthly expanding base retraining;
  TCN training origins ≤ 2016-12-27, gate training origins ≥ 2017-01-03
  (strict TCN → gate separation); eval-input origins from 2017-04-25

## 2. Models
- Base: global LightGBM (num_leaves 63, lr 0.05, feature_fraction 0.8,
  min_data_in_leaf 50, L1 objective), monthly retraining = the operational base
- TCN: lookback 28, hidden 32, kernel 3, dropout 0.2, lr 1e-3 (selected by
  scripts/tune_tcn.py on the gate-training period), Huber loss on scaled residuals
- Scaling: rolling 28-day mean + 1.0, clipped to ±10
- Gate: 26 features in 5 groups, MLP 64-32-1, output bias initialized at −2,
  loss = raw-unit L1 + 0.1·mean(g) (λ selected on gate-training WAPE only)
- Constant gate A1.5: c* = 0.00 (tuned on gate-training WAPE)
- TFT baseline: hidden 64, 4 heads, bf16, batch 1024, ≤ 20 epochs, early stopping
  patience 3, retrained per fold
- TFT-base variant: OOF residuals from a single TFT trained through 2016-02-29
  (137 late-starting series excluded); fold/holdout evaluation uses the per-fold
  TFTs (disclosed in the manuscript)

## 3. Evaluation
- Metrics: WAPE (primary), MAE/RMSE/RMSLE/MASE; six regime slices
  (post-promotion = 1–3 calendar days after a promotion ends; high volatility =
  upper quartile of training-period CV_28)
- Tests: series-level Wilcoxon (+ Holm), 2,000-draw bootstrap CIs, win rates;
  series with total actuals < 1 in a window excluded from per-series statistics
- Operational loss: 1:1 / 2:1 / 3:1 / 5:1 ratios, explicitly a point-forecast proxy
- Oracle-gate headroom and gate–benefit alignment analyses

## 4. Holdout discipline
- Evaluation command: `python scripts/11_holdout_eval.py` (both bases at once)
- Executions: 1 — on 2026-08-11, via `scripts/11_holdout_eval.py`. No retuning
  or re-evaluation of any kind is permitted afterwards.

## 5. Design iterations made on the development folds (disclosed in the paper)
1. Scaling denominator floor +1 (prevents zero-demand blow-ups) — 2026-08-10
2. Gate loss changed from scale-normalized Huber to raw-unit L1
   (evaluation-metric alignment) — 2026-08-10
3. Gate output bias initialized at −2 (closed start) — 2026-08-10
4. Sparsity regularizer λ = 0.1 (selected on gate-training WAPE; sensitivity in
   the appendix) — 2026-08-10

All four were based on development-fold observations only; the holdout was
untouched throughout.

## 6. Post-freeze additions (review response; development folds only)
The exploratory comparisons of manuscript Section 5.7 (feature-augmented base,
linear gate, single-signal switching gate, quantile reference) and the round-3
statistical analyses were added after the freeze in response to review. None of
them alter the frozen models; all are evaluated on development folds and labeled
exploratory in the manuscript.
