# ML challenger reference (Steps 4-5)

All 8 configurations committed in `PREREGISTRATION.md`, run via
`scripts/run_ml_experiment.py --config config.yaml` against the same
universe, factors, walk-forward folds (60mo train / 12mo test / 1mo
embargo), decile long-short construction, and 8bps costs as the baseline —
only the score source changes (a fitted model instead of an equal-weight
average of mom_z/val_z/qual_z).

## Walk-forward OUT-OF-SAMPLE results (109 stitched periods, same dates as the baseline)

| Configuration | Sharpe | Deflated Sharpe (n_trials=9) | Ann. return | Max drawdown | Turnover |
|---|---|---|---|---|---|
| gbm_lr0.03_depth3 | 0.16 | 0.146 | 1.23% | -38.05% | 71.59% |
| gbm_lr0.03_depth5 | -0.04 | 0.052 | -1.20% | -36.98% | 89.21% |
| gbm_lr0.1_depth3 | 0.07 | 0.093 | -0.02% | -37.08% | 79.07% |
| gbm_lr0.1_depth5 | 0.14 | 0.137 | 1.01% | -28.34% | 97.27% |
| rf_depth5_leaf50 | -0.01 | 0.061 | -1.11% | -44.77% | 82.48% |
| rf_depth5_leaf200 | -0.36 | 0.004 | -5.26% | -53.55% | 81.50% |
| **rf_depth10_leaf50** | **0.24** | **0.202** | 2.14% | -32.52% | 100.57% |
| rf_depth10_leaf200 | -0.22 | 0.014 | -2.79% | -39.97% | 100.38% |
| *baseline (linear composite)* | *0.07* | *0.578 (n_trials=1)* | *-0.57%* | *-47.78%* | *37.12%* |

Best of the 8: `rf_depth10_leaf50`, Sharpe 0.24 — modestly above the
baseline's 0.07. But its deflated Sharpe at the honest `n_trials=9` (8 ML
configs + the 1 pre-specified baseline, per `PREREGISTRATION.md`) is only
0.202 — far short of the ~0.95 bar for "likely genuine skill, not noise
picked out of a search." Every ML configuration also runs roughly 2-3x the
baseline's turnover (72-101% vs 37%), which would eat further into any edge
under a less generous cost assumption than the shared 8bps.

Read against the pre-registration's win condition (meaningfully higher
Sharpe AND survives the deflated Sharpe): **ML does not beat the linear
baseline here.** This is consistent with, not contrary to, the null
hypothesis.

## Files in this directory

- `ml_oos_net_returns.csv` — canonical single-file deliverable (Step 5), the first config (`gbm_lr0.03_depth3`)'s stitched OOS return series
- `ml_oos_net_returns_<config>.csv` — stitched OOS return series for each of the 8 configurations individually
- `ml_equity_curve_oos.png` — OOS equity curve for `gbm_lr0.03_depth3` vs SPY

See `../baseline/` for the linear composite's parallel outputs.
