# Baseline reference (Step 3)

The equal-weight linear composite from `backtester/`, run fresh via
`scripts/run_backtest.py --config config.yaml` (SP500, 2010-01-01 to
2024-12-31, monthly rebalance, decile long/short, 8bps costs, walk-forward
with a 60-month initial train / 12-month test / 1-month embargo). This is
the reference the ML challenger (Step 4+) has to beat, per
`PREREGISTRATION.md`.

## Walk-forward OUT-OF-SAMPLE (the headline numbers)

109 stitched OOS periods across 10 folds.

| Metric | Value |
|---|---|
| Annualized return | -0.57% |
| Annualized volatility | 18.54% |
| Sharpe ratio | 0.07 |
| Max drawdown | -47.78% |
| Hit rate | 55.96% |
| Avg monthly turnover | 37.12% |
| Deflated Sharpe (n_trials=1) | 0.578 |

Deflated Sharpe uses `n_trials=1` here because the baseline is a fixed,
pre-specified benchmark, not the result of a search — it wasn't picked as
the best of several tries. The pre-registration's `n_trials=9` correction
applies to whichever ML configuration comes out ahead in Step 6, not to
this baseline.

## Files in this directory

- `oos_net_returns.csv` — the stitched walk-forward OOS return series (the one to compare against the ML model's OOS series)
- `net_returns.csv` — full-period (in-sample) returns, reference only
- `coverage_report.csv` — per-rebalance-date data coverage
- `equity_curve_oos.png` — OOS equity curve vs SPY
- `equity_curve.png` — full-period equity curve vs SPY

## A note on why these numbers differ slightly from `backtester/BACKTESTER_README.md`

That README's numbers (OOS Sharpe 0.02, deflated Sharpe ~0.53) came from a
run made with a `TIINGO_KEY` set, so delisted/renamed tickers Yahoo Finance
has dropped (e.g. FB, TWTR, XLNX, CELG) fell back to Tiingo instead of
being skipped entirely. This re-run had no `TIINGO_KEY`, so those names are
missing outright — a different (smaller, Yahoo-only) slice of price
history feeding the same code. Coverage here came out to composite mean
95.4% / min 25.4%, value-and-quality mean 83.8% / min 18.6% (see
`coverage_report.csv`) — see that file for the full per-date detail rather
than assuming a direction relative to the original run.

The result is the same story either way — an OOS Sharpe indistinguishable
from zero after costs — but the exact decimals moved (0.02 -> 0.07,
deflated 0.53 -> 0.578). Getting a `TIINGO_KEY` (free signup at tiingo.com)
and re-running would reproduce the original numbers more closely; it's not
required for the ML-vs-baseline comparison, since both sides will run
through this same code and this same data either way.
