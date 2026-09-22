# Pre-registration

Author: Rob Basnet
Date: 2026-07-07

## Why this file exists

I'm writing down the plan for this study before I run any machine-learning model or
look at any result. The idea is simple: if I decide what counts as success ahead of time,
I can't trick myself later by trying a hundred things and keeping the one that looks good.
Everything below is a commitment. If I change any of it later, I'll note the change and
why, rather than quietly editing this file.

## The question

Can a machine-learning model combine momentum, value, and quality into a better long/short
stock signal than a plain equal-weight combination of the same three factors — out of
sample, in US large-cap stocks, after trading costs?

## What I already know (the baseline)

I built a backtester first (that's the other part of this repo). It ranks S&P 500 stocks
each month by an equal-weight blend of momentum, value, and quality, goes long the top
decile and short the bottom decile, and charges realistic costs. Run out of sample with
walk-forward validation, that simple version earns basically nothing after costs:
an out-of-sample Sharpe of about 0.02, slightly negative return, with a deep drawdown.

So the bar is low. The question is whether a smarter, non-linear model does any better
with the exact same information.

## My hypothesis

I'm stating it as a null on purpose, because that keeps me honest:

> Machine learning does NOT beat the equal-weight linear combination out of sample after
> costs.

If the data proves me wrong, great. If it doesn't, that's still a real and useful finding.

## What I'm going to do

- Use the same universe, dates, factors, portfolio rules, and costs as the backtester.
  The ONLY thing that changes is how the three factor scores get combined into one signal.
- Baseline: the equal-weight composite I already have.
- Challenger: an ML model that learns, from past data only, how to combine the factors to
  predict next-month returns.
- Both signals go through the same decile long/short engine with the same 8 basis-point
  cost, so it's a fair comparison.
- I judge everything on the walk-forward, out-of-sample results only. In-sample numbers are
  for reference, never the headline.

## The models and settings I'm committing to

I'll try these, and only these. Counting them now matters, because the deflated Sharpe
ratio (my check against luck) needs to know how many things I tested.

Gradient boosting:
- learning rate: 0.03 and 0.1
- max depth: 3 and 5
- (= 4 combinations)

Random forest:
- max depth: 5 and 10
- min samples per leaf: 50 and 200
- (= 4 combinations)

That's 8 model settings for the challenger, plus the 1 linear baseline = 9 total
configurations. I'll use 9 as the trial count in the deflated Sharpe. If I genuinely need
to test something outside this list, I'll add it here first and update the count.

## How I decide if ML "wins"

ML wins only if BOTH of these are true on the stitched out-of-sample results:

1. Its Sharpe ratio is meaningfully higher than the baseline's (not a rounding-error
   difference), and
2. That result survives the deflated Sharpe — i.e. it's unlikely to be just luck given the
   9 configurations I tried.

If only the in-sample number looks good, that does not count as a win.

## Rules I'm holding myself to

- No look-ahead. The model only ever sees data from before the period it's predicting.
- No leakage. Any scaling or filling-in of missing values is learned from the training
  window only, never the test window. Keep the same purge/embargo gap the backtester uses.
- Same universe, same costs, same rebalance schedule for both the baseline and the ML model.
- Report the honest out-of-sample number no matter what it says.

## What I'll report either way

A short paper with the question, the data, the method, one comparison table (baseline vs
ML), one out-of-sample equity curve, and an honest read of whether ML added anything after
costs — including the limitations. A negative or "no real difference" result gets written
up just as clearly as a positive one.

## Deviations from the original plan

Per the commitment at the top of this file, changes get noted here, not edited quietly into
the text above.

**2026-09-22 — baseline Sharpe restated (0.02 -> ~0.07/0.066), cause identified.**
"What I already know (the baseline)" above states an out-of-sample Sharpe of about 0.02.
That number came from an earlier run of the backtester with a `TIINGO_KEY` set, so Yahoo
Finance's gaps for delisted/renamed names (e.g. FB, TWTR, XLNX, CELG) were filled by Tiingo's
fallback. The committed baseline in `results/baseline/README.md` and everything downstream
(Steps 3-6) comes from a re-run without `TIINGO_KEY`, giving OOS Sharpe 0.066-0.07 instead.
Investigating this (see `results/coverage_gap_note.md`) surfaced something bigger than a
decimal shift: **213 of the 694 point-in-time S&P 500 tickers in the study window (30.7%)
have zero rows anywhere in the panel**, because the Tiingo fallback never engaged. 196 are
explained by `backtester/data_cache/unavailable_prices.json`; 17 are not, and were separately
checked (`results/ticker_identity_check.md`) to rule out the worse failure mode — a base
ticker's price history actually belonging to a different company than the one that was the
point-in-time member (e.g. new "Dow Inc." data attributed to old Dow Chemical's membership
window). That check came back clean: every flagged ticker's price coverage falls entirely
outside its old membership window, so this is a *missing-data* problem, not a *wrong-data*
problem — but it does mean "survivorship-bias-free" is only true of `build_universe`'s
membership matrix, not of what the backtest and ML panel actually trade on, until
`TIINGO_KEY` is set and Steps 3-6 are re-run. `unavailable_fundamentals.json` (216 entries)
is a separate, SEC EDGAR-side coverage gap that a Tiingo key does not fix.

**2026-09-22 — the deflated Sharpe threshold (0.95) was never numerically committed here.**
"How I decide if ML 'wins'" above says the result must "survive the deflated Sharpe" but
never fixes a number. 0.95 is the conventional interpretation used in
`src/analytics/metrics.py`'s docstring ("conventionally >= 0.95 is treated as 'likely genuine
skill'"), and it's what `scripts/compare_ml_vs_baseline.py` checks — but choosing it now,
after seeing results, is a post-hoc threshold pick, however standard. Stated honestly: the
best ML configuration's net-of-cost deflated Sharpe is 0.2022 (n_trials=9); **the "ML does
not win" conclusion is unchanged for any threshold >= 0.21**, i.e. it isn't sensitive to
where exactly the line gets drawn within any conventional range (0.5, 0.75, 0.95 all give
the same answer). See `results/comparison/README.md` for the full comparison and the
statistical-power analysis of what this study's sample size could have detected in the first
place.
