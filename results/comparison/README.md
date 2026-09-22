# Step 6 — Compare

Built by `backtester/scripts/compare_ml_vs_baseline.py`, which reads the
already-saved stitched walk-forward OOS return series from Step 3
(`results/baseline/oos_net_returns.csv`) and Steps 4-5
(`results/ml/ml_oos_net_returns_<config>.csv`) and recomputes Sharpe,
deflated Sharpe, annualized return, max drawdown, and hit rate directly
from those series — a check against, not a copy of, `results/baseline/README.md`
and `results/ml/README.md`. The recomputed numbers matched those READMEs
exactly. Turnover isn't recomputed here (it needs the per-period weights,
which weren't persisted to disk); those figures are carried over verbatim
from the Step 3/5 run and the script documents that source.

## Comparison table

| Config | OOS Sharpe | Deflated Sharpe | n_trials | Ann. return | Max drawdown | Hit rate | Turnover | Periods |
|---|---|---|---|---|---|---|---|---|
| **baseline (linear, equal-weight)** | 0.066 | 0.578 | 1 | -0.57% | -47.78% | 55.96% | 37.12% | 109 |
| gbm_lr0.03_depth3 | 0.160 | 0.146 | 9 | 1.23% | -38.05% | 53.21% | 71.59% | 109 |
| gbm_lr0.03_depth5 | -0.036 | 0.052 | 9 | -1.20% | -36.98% | 46.79% | 89.21% | 109 |
| gbm_lr0.1_depth3 | 0.066 | 0.093 | 9 | -0.02% | -37.08% | 54.13% | 79.07% | 109 |
| gbm_lr0.1_depth5 | 0.145 | 0.137 | 9 | 1.01% | -28.34% | 55.96% | 97.27% | 109 |
| rf_depth5_leaf50 | -0.009 | 0.061 | 9 | -1.11% | -44.77% | 55.05% | 82.48% | 109 |
| rf_depth5_leaf200 | -0.357 | 0.004 | 9 | -5.26% | -53.55% | 47.71% | 81.50% | 109 |
| **rf_depth10_leaf50 (best of 8)** | **0.241** | **0.202** | 9 | 2.14% | -32.52% | 59.63% | 100.57% | 109 |
| rf_depth10_leaf200 | -0.220 | 0.014 | 9 | -2.79% | -39.97% | 46.79% | 100.38% | 109 |

Full precision in `comparison_table.csv`.

## Win condition

Per `PREREGISTRATION.md`: ML wins only if a config's OOS Sharpe beats the
baseline's **and** its deflated Sharpe (n_trials=9) stays above the ~0.95
bar. `rf_depth10_leaf50` clears the first test (0.241 vs 0.066) but not the
second (0.202 vs ~0.95) — **the win condition is not met.** This is
consistent with the pre-registered null hypothesis.

## Equity curve

`equity_overlay_baseline_vs_best_ml.png` — OOS growth of $1 (net of 8bps
costs), baseline vs `rf_depth10_leaf50`. The baseline's swings are much
wider (including the 2020 spike and crash that drives its -47.78% max
drawdown); the best ML config is steadier but never builds a durable edge
over the full window, consistent with its Sharpe being statistically
indistinguishable from noise once the search over 9 configurations is
priced in.

## Reproducing

```bash
cd backtester
PYTHONPATH=. python scripts/compare_ml_vs_baseline.py
```
