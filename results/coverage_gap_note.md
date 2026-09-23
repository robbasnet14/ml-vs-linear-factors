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

## Tiingo was tried — free-tier key has no depth for these names (2026-09-22)

The user obtained a `TIINGO_KEY` and we ran the 5-name, 10-minute spot
check the review proposed before committing to a full re-fetch: `ANTM,
APC, CEPH, BNI, BMC` (all in `unavailable_prices.json`), via
`load_prices(..., force_refresh=True)`.

**Result: 0 of 5 returned any price data.** All 5 failed on Yahoo exactly
as before, and Tiingo's `/tiingo/daily/<ticker>/prices` endpoint returned
HTTP 200 with an **empty array** for every one of them — this is not an
auth problem (a control fetch of AAPL on the same key returned real data
immediately) and not a code bug (`_tiingo_headers`/`_fetch_tiingo_prices`
worked as documented). Querying Tiingo's ticker-metadata endpoint
(`/tiingo/daily/<ticker>`, no `/prices`) directly confirms it: for BMC,
BNI, CEPH, ANTM, Tiingo correctly identifies the company ("BMC Software
Inc", "BURLINGTON NORTHERN SANTA FE LLC", "CEPHALON INC" — all flagged
`DELISTED`) but reports `"startDate": null, "endDate": null"` — Tiingo
knows who these companies were but has never ingested price history for
them on this key/tier. This looks like a real, structural gap in Tiingo's
historical depth for names delisted well before its own IEX-era data
backfill, not something a retry or a code fix resolves.

(Bonus, unrelated to the fix: querying `APC` returned a *different*
company — "ARKO Petroleum Corp - Class A", trading since 2026-02-12,
unrelated to the original Anadarko Petroleum this study's `APC` refers to.
Real-world confirmation of the ticker-reuse risk `results/ticker_identity_check.md`
already checked for — harmless here since it postdates the 2010-2024 study
window entirely, but one more point for the "use a permanent identifier,
not a ticker symbol" limitation.)

**Decision, per the pre-agreed protocol ("if it doesn't [return history],
you've learned that cheaply and the reframe is the answer — stop and move
on"): the full 196-name re-fetch was not run.** Spending the API calls on
the rest of the list would almost certainly reproduce the same null result
for names delisted in the same era (2008-2013 for these 5), and the
sample size (5 of 5 failing, spanning different sectors and delisting
reasons — bankruptcy, acquisition, merger) is a reasonable basis for that
call within the 10-minute timebox.

## Resolution: the reframe is the paper's official Data-section language

- **Keep "point-in-time, survivorship-bias-free" only for `build_universe`'s
  membership matrix** (694 tickers, verified correct — no delisted name
  improperly excluded from *membership*).
- **For the panel the models actually train and trade on**, state: "30.7%
  of point-in-time S&P 500 members (213 of 694) are excluded from the
  analysis for lack of available price history, disproportionately
  delisted, acquired, and renamed names. The direction of the resulting
  bias is ambiguous for a dollar-neutral long/short decile book, and we do
  not attempt to sign it: names that deteriorated into bankruptcy would
  mostly have sat in the short leg, so dropping them removes short-leg
  gains a real investor would have earned; names acquired at a premium
  would have produced a sharp adverse move against a short position, so
  dropping them removes short-leg losses a real investor would have taken.
  These pull in opposite directions, and nothing in the available data lets
  us net them out. Coverage was checked against Yahoo Finance and Tiingo;
  Tiingo's free tier had no price history for a 5-name spot check of
  confirmed-delisted names (ANTM, APC, CEPH, BNI, BMC) spanning the
  exclusion period, so a fuller Tiingo re-fetch was not pursued. Results
  are therefore conditional on names with continuous price coverage, not
  the full historical membership — see `results/coverage_gap_note.md`."
  (The instinct that excluding delisted names simply inflates returns is a
  long-only intuition — it does not transfer to a dollar-neutral long/short
  design, where those same names could equally have sat on the short leg.)
- Do **not** resolve this with a footnote about the baseline Sharpe moving
  from 0.02 (the original `TIINGO_KEY` run cited in
  `backtester/BACKTESTER_README.md` — a run made before this gap was
  understood) to 0.07/0.066 (this repo's committed run) — that drift is a
  symptom of this gap, not a separate, smaller issue, and the 0.02 number
  should not be cited in the paper without this caveat attached.
- `unavailable_fundamentals.json`'s 216-name gap is separate (SEC EDGAR
  coverage, not fixable by Tiingo at all) and should be disclosed
  independently, per the review's point 2.

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
