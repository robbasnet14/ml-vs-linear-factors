# Correction facts sheet: EPS split-basis leak

Numbers for writing the correction, each with where it comes from. Tables are in
`results/correction/tables/`, produced by `backtester/scripts/correction_tables.py`
from the re-run outputs in `results/correction/{published,oldformula,corrected}/`.
This is a reference, not prose for the paper.

## 1. The defect

- **What:** the value factor (earnings yield) divided EPS *as filed* by a price adjusted for
  every split and dividend up to the download date. After a split, earlier EPS is on the old
  share basis while the price is on the new one, so earnings yield was inflated by each
  stock's *future* split factor (and, more mildly, by future dividends). A look-ahead leak.
- **Where it entered:** the vendored engine (factor-backtester `31b92b4`). Fixed in
  factor-backtester `d68487a` (EPS restated across later splits; price split-adjusted but not
  dividend-adjusted). Same-basis artifacts the fix also removes: AAPL TTM EPS "falling" 61%
  after its 2020 split; a derived NVDA Q4 FY2022 of −$1.09 (actual +$1.18).
- **Scope in this study:** momentum and quality are unaffected (price ratios; company totals).
  Value is affected in the linear baseline and every ML configuration (both consume `val_z`).

## 2. How it was measured

- **Pinned environment:** Python 3.12.9, scikit-learn 1.5.1, pandas 2.3.2, numpy 2.0.1,
  scipy 1.15.2 (`backtester/requirements.txt`). Under scikit-learn 1.9 the same code and data
  give `rf_depth10_leaf50` 0.147 instead of 0.241 (an unpinned-dependency defect, now pinned).
- **Reproduction first:** the re-run `published` mode matches all nine published return
  series exactly (max absolute difference 0 in every month) and the published factor IC
  table — `tables/reproduction_check.csv`.
- **Three modes,** same pipeline, same pre-registered grid, every model refit per fold:
  `published` (no change), `oldformula` (published formula on re-downloaded data: isolates
  data drift), `corrected` (the fix). Feature built by factor-backtester `d68487a`
  (`inputs/build_value_panels.py`). n_trials stays 9: the same nine configurations recomputed,
  not a new search.
- **The feature swap did only what it should** (`tables/value_panel_agreement.csv`): mean
  monthly rank correlation between leaky and corrected value is 0.956 for the 446 names that
  never split 2008–2024 (only the dividend part of the fix applies) and 0.680 for the 201 that
  split at least once. Coverage: 76,950 non-NaN cells leaky vs 76,624 corrected.
- **Discarded runs:** the first measurement runs were run in parallel with the vendored
  loader online; Yahoo throttling made it skiplist MCHP, KDP, HPQ, COST and HD mid-run
  (stocks with cached prices). Those numbers were thrown away; all numbers here come from
  the offline harness, whose `published` mode reproduces exactly.

## 3. Headline, published vs corrected (`tables/summary.csv`, `tables/sharpe_by_config.csv`)

Walk-forward out-of-sample, net of 8 bps costs.

| | Published | Old formula, fresh data | Corrected |
|---|---|---|---|
| Baseline Sharpe | 0.066 | 0.052 | −0.173 |
| Baseline deflated Sharpe, n=1 | 0.578 | 0.561 | 0.296 |
| Baseline deflated Sharpe, n=9 | 0.093 | 0.086 | 0.020 |
| Best ML config | rf_depth10_leaf50 | rf_depth10_leaf50 | gbm_lr0.03_depth3 |
| Best ML Sharpe | 0.241 | 0.096 | −0.361 |
| Best ML deflated Sharpe, n=9 | 0.202 | 0.108 | 0.005 |
| Best ML max drawdown | −32.5% | −36.5% | −55.8% |
| ML Sharpe range (8 configs) | −0.357 .. 0.241 | −0.353 .. 0.096 | −0.590 .. −0.361 |
| ML configs beating the baseline | 3 of 8 | 1 of 8 | 0 of 8 |

- Every corrected ML configuration is below a baseline that is itself negative.
- Data drift alone (old formula, fresh data) barely moves the baseline (0.066 → 0.052) but
  takes the published best configuration from 0.241 to 0.096.

## 4. Factor IC, full sample (`tables/factor_ic.csv`)

Monthly cross-sectional rank IC vs next-month return.

| Factor | Published mean (t) | Old formula, fresh data | Corrected |
|---|---|---|---|
| val_z | 0.0111 (t = 1.26) | 0.0105 (t = 1.20) | −0.0051 (t = −0.48) |
| mom_z | 0.0036 (t = 0.22) | unchanged | unchanged |
| qual_z | 0.0041 (t = 0.46) | unchanged | unchanged |

Value was the only factor with |t| > 1; corrected, none is.

## 5. Fold 7 (2022-09 to 2023-09)

**Summed monthly net return** (`tables/per_fold.csv`):

| | Published | Old formula, fresh data | Corrected |
|---|---|---|---|
| rf_depth10_leaf50 | +0.280 | +0.244 | +0.001 |
| Baseline | −0.067 | −0.078 | −0.122 |

**Attribution by later forward splits** (`tables/fold7_split_attribution.csv`):
rf_depth10_leaf50, gross (before costs) monthly contribution summed over the fold; "split
after" = a forward split between 2023-09-01 and 2024-12-31, when the leak inflated that name's
value score during fold 7.

| Published | Names held | Mean share of leg weight | Contribution |
|---|---|---|---|
| Long, split after the fold | 8 | 7.2% | +0.046 |
| Long, no split after | 145 | 92.8% | +0.358 |
| Short, split after the fold | 7 | 2.7% | −0.006 |
| Short, no split after | 195 | 97.3% | −0.109 |
| **Total** | | | **+0.289** |

Corrected total: +0.010.

- The test-window channel (holding later-splitters) accounts for +0.046 of +0.289 in the
  published run, so it is not the mechanism. The edge disappears once the feature is
  corrected, with nothing else changed, which leaves contamination of the training data —
  the leaky value→return relationship in the pre-fold history the model was fitted on —
  as the explanation. That last step is an inference from these two measurements, not a
  direct measurement of what the model learned.

## 6. Exploratory only: did the ML configurations lose more than the baseline?

`tables/swing_by_config.csv`. Post hoc (formed after seeing results), n = 8 configurations:
an observation, not a finding.

- Baseline Sharpe change, published → corrected: −0.239.
- Mean ML change −0.454, 1.90× the baseline's; 7 of 8 configurations changed more than the
  baseline. The published best (rf_depth10_leaf50) changed by −0.697 (2.92×): it was selected
  for looking best, so it is the configuration most expected to have exploited the artifact.
- The two most regularised forests changed least (rf_depth5_leaf200 −0.066,
  rf_depth10_leaf200 −0.251), consistent with flexibility mattering but not a test of it.
- **No significant difference in mean monthly return impact** relative to the baseline: paired
  t from −1.57 (gbm_lr0.1_depth5) to +0.78; |t| < 2 for all eight. The paired test is on mean
  monthly return impact, not on the Sharpe swings themselves.

## 7. COL: ticker identity (`tables/col_identity.csv`)

- COL (Rockwell Collins) was an S&P 500 member 2010-01-01 to 2018-11-26 (acquired).
- The study's cached COL series runs 2012-08 to 2020-11; Rockwell Collins (from
  factor-backtester `d68487a`) runs 2008-01 to 2018-11. Monthly returns over the 75
  overlapping months correlate at 0.069: the cache holds a different security under the
  same symbol for COL's membership from 2012-08 on, and nothing for 2010-01 to 2012-07.
- `results/ticker_identity_check.md` examined 20 names: the 3 with non-contiguous membership
  runs (AMD, DXC, GAS) and 17 previously flagged tickers. COL is in neither group. Its method
  states that a non-contiguous run "is the only structural pattern under which one ticker
  symbol could span two different companies"; COL had a single contiguous membership run, so
  it contradicts that premise. (Consistent with a data source attaching the current holder's
  full price history to a reused symbol, which can overlap the old company's window; which
  security the cached series belongs to isn't established here.)
- PAPER.md says "We checked every structural pattern under which this could ... pair the wrong
  entity's data with a valid membership window ... and found no contamination" (Data section,
  "Ticker identity") and "We checked every structural pattern under which this could
  contaminate a valid membership window with the wrong entity's price data and found none"
  (Limitations, item 3). COL is a counterexample to both. Only a permanent security
  identifier (e.g. CRSP PERMNO) would give that guarantee.
- Effect on the numbers: not isolated. COL is one of ~480 names. In every mode its returns
  and momentum come from the study's own (contaminated) price cache; in the oldformula and
  corrected modes its value input comes from the d68487a panel, which uses Rockwell Collins.

## 8. Reproducing

See `results/correction/README.md`.
