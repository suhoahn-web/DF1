# Context-Gated Residual Correction (CGRC) for Retail Demand Forecasting

Reproduction code for the paper "When Should a Forecast Be Corrected? Context-Gated
Residual Correction for Retail Demand Forecasting."

CGRC augments a base forecaster (global LightGBM or Temporal Fusion Transformer)
with a causal TCN residual corrector and a small context gate g ∈ [0,1] that learns
*when* to activate the correction from observable context (promotions, calendar,
demand state, recent base errors):

```
final(i,t,h) = max(0, base(i,t,h) + g(i,t,h) · r̂(i,t,h))
```

## Data

The Corporación Favorita Grocery Sales Forecasting dataset is distributed through
Kaggle and cannot be redistributed here. Download it from
https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting (competition
rules must be accepted) and place the archive in `data/raw/`, then run
`python scripts/00_download_data.py` to extract.

## Environment

Python ≥ 3.10. `pip install -r requirements.txt`. The TFT baseline requires a GPU
(`pytorch-forecasting`); all other steps run on CPU.

## Reproduction pipeline

Scripts run in numeric order; every rule (analysis period, cleaning, series
eligibility, splits, hyperparameters) is frozen in `configs/`.

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/01_prepare_data.py --final` | cleaned panel, 3,000-series sample, supervised table, feature audit |
| 2 | `scripts/02_train_baselines.py --tag final` | Seasonal Naive + LightGBM folds, temporal OOF residuals |
| 3 | `scripts/03_train_tft.py` / `03b_tft_oof.py` (GPU) | TFT fold predictions and TFT-base OOF residuals |
| 4 | `scripts/04_train_residual.py --tag final` | TCN corrector, always-on (A1) and constant-gate (A1.5) variants |
| 5 | `scripts/05_train_gate.py --tag final` | context gate (A4), gate diagnostics |
| 6 | `scripts/06_run_ablations.py --tag final` | context-group ablations (A2/A3/A5) |
| 7 | `scripts/07_make_paper_tables.py --tag final` | main tables, statistical tests, operational cost |
| 8 | `scripts/09_build_tftbase.py` → steps 4–6 with `--tag tftbase` | TFT-base variant |
| 9 | `scripts/11_holdout_eval.py` | the single-use holdout evaluation |
| 10 | `scripts/12_review_analyses.py`, `13_p9_baselines.py`, `14_round3_stats.py` | effect sizes, cluster bootstrap, DM tests, exploratory baselines |
| 11 | `scripts/10_make_figures.py` | paper figures |

`FREEZE_DECLARATION.md` documents the design freeze: every modeling choice fixed
before the holdout, the dated log of the four validation-fold design iterations,
and the single holdout run. Leakage is enforced by unit tests
(`pytest tests/`), including a perturbation test asserting that corrupting all
post-origin sales changes no feature.

## Results

`results/metrics/`, `results/statistical_tests/`, and `results/ablations/` contain
the CSVs from which every table in the paper is generated; `results/figures/`
contains the figures. Seeds are fixed (20260810); exact numbers may vary at the
last decimal across library versions.

## License

MIT (code). The dataset remains subject to its Kaggle competition terms.
