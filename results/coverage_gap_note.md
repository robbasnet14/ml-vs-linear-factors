# Data coverage gap in the ML panel (review finding, verified)

**Verified independently 2026-09-21.** 213 of the 694 point-in-time S&P 500
tickers in the study window (2010-01-01 to 2024-12-31) — 30.7% — have zero
rows anywhere in the ML panel (`build_ml_panel`'s output) or in the linear
baseline's composite (both draw from the same `load_prices` call, so the
gap is identical for both arms).

## Root cause

`TIINGO_KEY` was not set for the run this repo's committed results come
from. `backtester/src/data/loader.py` falls back to Tiingo only for prices
Yahoo Finance can't serve (delisted, acquired, or renamed tickers); without
the key, those names are dropped outright rather than backfilled, and get
recorded in `backtester/data_cache/unavailable_prices.json`.

Of the 213 missing tickers:
- **196** are explained by `unavailable_prices.json` (196 entries total, all
  196 fall inside the universe).
- **17 are unexplained by that skiplist**: `ADT, BEAM, CEG, DELL, DOW, EMC,
  FCPT, JAVA, MHS, MI, MMI, NE, NSM, S, SE, SHLD, TE`. Several of these
  (DOW, DELL, CEG) are tickers that were reused by an unrelated later
  listing after the original company was acquired/split/spun off — worth a
  follow-up look at whether `load_prices`/`build_universe` is silently
  conflating two different securities under one ticker, or just failing
  quietly for a different reason than the logged skiplist. Not yet
  diagnosed.

`unavailable_fundamentals.json` has 216 entries (mostly overlapping the
196 price-unavailable names) and does not by itself remove a ticker from
the panel — a row with a missing `val_z`/`qual_z` but a valid price and
forward return still appears; see `results/ml/README.md`'s coverage
numbers for that separate (smaller) gap.

## Why this matters

Project A's stated claim — a point-in-time, survivorship-bias-free
universe including delisted names — is only true of the *membership
matrix* (`build_universe`'s output, 694 tickers, verified 0 improperly
stripped or leaked). It is **not** true of what the backtest and the ML
panel actually trade on: 30.7% of that universe, concentrated in delisted
and acquired names (exactly where survivorship bias does its damage),
never gets a price and is silently absent from every downstream frame with
no warning either script prints to that effect.

## The two honest ways out (per the review)

1. **Preferred: set `TIINGO_KEY` and re-run both arms** (baseline +
   dataset + all 8 ML configs), so the Tiingo fallback the codebase already
   has actually engages. This needs a free Tiingo signup — outside what
   this session can do unattended. **Action needed from the user.**
2. **If (1) isn't done before publication:** state the exact coverage loss
   in the paper (this file's numbers), and change the Data section's
   framing from "survivorship-bias-free" to "point-in-time universe with
   continuous-Yahoo-coverage names only; 30.7% of point-in-time members
   excluded for missing price history, disproportionately delisted/
   acquired/renamed names — see `results/coverage_gap_note.md`." Keep
   "survivorship-bias-free" only for a description of `build_universe`
   itself (the membership matrix), never for the panel the models actually
   train and trade on.

Do not resolve this with a footnote about the baseline Sharpe moving from
0.02 (the original `TIINGO_KEY` run cited in `backtester/BACKTESTER_README.md`)
to 0.07/0.066 (this repo's `TIINGO_KEY`-less re-run) — that drift is a
symptom of this gap, not a separate, smaller issue.

## Reproducing this check

```bash
cd backtester
PYTHONPATH=. python -c "
from src.data.universe import build_universe
from src.data.loader import load_prices, load_fundamentals
from src.ml.dataset import build_ml_panel
from src.utils.config import load_config
import json
cfg = load_config('config.yaml')
u, d = cfg['universe'], cfg['data']
uni = build_universe(u['name'], u['start_date'], u['end_date'], cache_dir=d['cache_dir'])
members = set(uni.columns[uni.any(axis=0)])
tk = sorted(members)
px = load_prices(tk, u['start_date'], u['end_date'], cache_dir=d['cache_dir'])
fu = load_fundamentals(tk, u['start_date'], u['end_date'], lag_days=d['fundamentals_lag_days'], cache_dir=d['cache_dir'])
panel = build_ml_panel(px, fu, uni, cfg['factors'])
missing = members - set(panel['ticker'])
unavail = set(json.load(open('data_cache/unavailable_prices.json')))
print(len(members), len(missing), len(missing & unavail), sorted(missing - unavail))
"
```
