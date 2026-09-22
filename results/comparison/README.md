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

## Both n_trials conventions, side by side

The table above uses n_trials=1 for the baseline (a fixed, pre-specified
benchmark) and n_trials=9 for every ML config (the honest cost of the
search). That asymmetry is correct, but shown alone it invites misreading
the baseline as "better" (DSR 0.578 vs 0.202). Here both arms under both
conventions, so nothing is hidden:

| Config | Sharpe | DSR @ n_trials=1 | DSR @ n_trials=9 |
|---|---|---|---|
| baseline | 0.066 | 0.578 | 0.093 |
| rf_depth10_leaf50 (best ML) | 0.241 | 0.754 | 0.202 |

Under either convention the ML config's raw Sharpe is higher, but neither
arm clears 0.95 at n_trials=9 — the honest number for a 9-config search.

## Gross-of-cost vs net-of-cost (is the signal real but untradeable, or not real?)

Built by `backtester/scripts/gross_vs_net.py`, which re-derives weights
exactly as the committed pipeline does, then stitches walk-forward OOS
results with the new `run_backtest_breakdown` (same fold boundaries, same
delisting-exit handling as `run_backtest` — see `src/backtest/engine.py`)
so gross and net come from one pass over the same data. Its net_return
output was cross-checked against `results/baseline/oos_net_returns.csv`
(max abs diff 9.7e-17, i.e. float noise) before trusting the gross column.

| Config | Gross Sharpe | Net Sharpe | Cost drag (Sharpe pts) | Gross DSR | Net DSR | Turnover |
|---|---|---|---|---|---|---|
| baseline | 0.087 | 0.066 | 0.021 | 0.602 | 0.578 | 37.12% |
| gbm_lr0.03_depth3 | 0.209 | 0.160 | 0.049 | 0.180 | 0.146 | 71.59% |
| gbm_lr0.03_depth5 | 0.032 | -0.036 | 0.068 | 0.077 | 0.052 | 89.21% |
| gbm_lr0.1_depth3 | 0.122 | 0.066 | 0.056 | 0.123 | 0.093 | 79.07% |
| gbm_lr0.1_depth5 | 0.222 | 0.145 | 0.078 | 0.193 | 0.137 | 97.27% |
| rf_depth5_leaf50 | 0.048 | -0.009 | 0.057 | 0.084 | 0.061 | 82.48% |
| rf_depth5_leaf200 | -0.297 | -0.357 | 0.060 | 0.007 | 0.004 | 81.50% |
| **rf_depth10_leaf50** | **0.321** | **0.241** | **0.080** | **0.267** | **0.202** | 100.57% |
| rf_depth10_leaf200 | -0.129 | -0.220 | 0.090 | 0.028 | 0.014 | 100.38% |

Full precision in `gross_vs_net.csv`.

**Reading this:** `rf_depth10_leaf50`'s cost drag (0.080 Sharpe points) is
roughly 4x the baseline's (0.021), tracking its ~2.7x turnover. But even
its **gross-of-cost** deflated Sharpe (0.267) falls far short of the ~0.95
bar. Costs make the result worse, but they are not what's driving the
non-result — the finding is "no detectable edge," not "an edge that costs
ate." Every other ML config's gross DSR is lower still.

## Statistical power: could this design have detected a real edge?

Built by `backtester/scripts/dsr_power_table.py`, which inverts the
`deflated_sharpe` formula (root-finds the annualized Sharpe needed to hit a
target DSR, at n_trials=9, given a sample size and skew/kurtosis) rather
than assuming a generic normal-returns approximation. Using
`rf_depth10_leaf50`'s own empirical skew (-1.35) and kurtosis (9.43, vs 3
for normal) — its OOS returns are meaningfully negatively skewed and
fat-tailed, which raises `deflated_sharpe`'s variance term and therefore
the bar:

| OOS months | Ann. Sharpe needed for DSR ≥ 0.50 | for DSR ≥ 0.95 |
|---|---|---|
| 80 | 0.69 | 1.90 |
| 100 | 0.60 | 1.58 |
| **109 (this study)** | **0.57** | **1.48** |
| 118 | 0.55 | 1.39 |
| 130 | 0.52 | 1.30 |
| 150 | 0.48 | 1.17 |

For comparison, the same table under a normal-distribution assumption
(skew=0, kurt=3) is close to but somewhat more lenient than the empirical
one (e.g. at 109 months: 0.51 / 1.08 vs 0.57 / 1.48) — full detail in
`dsr_power_table.csv`.

**Reading this:** at 109 OOS months and n_trials=9, this design needed an
annualized OOS Sharpe of roughly 1.5 to declare a win at DSR ≥ 0.95 — using
this study's own return-distribution shape, not a normal approximation
that would understate the bar. A genuine edge in the Sharpe 0.3-0.5 range
(roughly what the honest ML-vs-linear-factors literature finds after
costs) would be **undetectable** by this design at this sample size and
this multiple-testing correction. The paper's finding is therefore two
things, not one: (1) we fail to reject the null, and (2) the design could
not have rejected it for any realistic effect size — a materially
different and stronger claim than "ML didn't win."

## Reproducing

```bash
cd backtester
PYTHONPATH=. python scripts/compare_ml_vs_baseline.py
PYTHONPATH=. python scripts/gross_vs_net.py
PYTHONPATH=. python scripts/dsr_power_table.py
```
