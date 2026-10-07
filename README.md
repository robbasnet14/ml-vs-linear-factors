# ML vs. Linear Factors

**Does a machine-learning model that combines momentum, value, and quality beat a plain
equal-weight linear combination of the same three factors, out of sample, in US large-cap
stocks, after trading costs?**

**No.** Every machine-learning configuration in the pre-registered grid loses to the
linear baseline, and the baseline itself loses money: corrected out-of-sample Sharpe
−0.590 to −0.361 across the eight models, against −0.174 for the baseline.

**This is a corrected result.** The version published on 2026-09-23 reported the best
configuration at 0.241 against a baseline at 0.066. Both were artifacts of a look-ahead
leak in the value factor, described below. The pre-registered null held then and holds
now; the correction makes it stronger. Full write-up: [PAPER.md](PAPER.md)
([PDF](paper/ml-vs-linear-factors.pdf)).

## Headline numbers, published and corrected

Walk-forward out-of-sample, net of 8 bps costs.

| | Published | Corrected |
|---|---|---|
| Baseline Sharpe | 0.066 | −0.174 |
| Baseline deflated Sharpe, n=1 / n=9 | 0.578 / 0.093 | 0.294 / 0.020 |
| Best ML configuration | rf_depth10_leaf50 | gbm_lr0.03_depth3 |
| Best ML Sharpe | 0.241 | −0.361 |
| Best ML deflated Sharpe, n=9 (bar ≈0.95) | 0.202 | 0.005 |
| Best ML max drawdown | −32.5% | −55.8% |
| ML configurations beating the baseline | 3 of 8 | 0 of 8 |

*The baseline's deflated Sharpe is shown under both conventions on purpose: n=1 treats it
as a fixed, pre-specified benchmark, n=9 charges it the same multiple-testing cost as the
ML search. Read alone, the n=1 figure invites the wrong conclusion — in the published run,
under the same n=9 convention, the baseline scored 0.093, lower than ML's 0.202. Neither
side clears the ≈0.95 bar, published or corrected. Published-run figures that were not
re-tabulated for the correction: baseline max drawdown −47.78%; average monthly turnover
37.12% (baseline) and 100.57% (rf_depth10_leaf50).*

![Out-of-sample equity curve, published run: baseline vs. rf_depth10_leaf50](results/comparison/equity_overlay_baseline_vs_best_ml.png)

*The chart shows the published run, before the correction.*

## Correction (2026-10-07)

**The defect.** The value factor (earnings yield) divided EPS *as filed* by a price
adjusted for every split and dividend up to the download date. After a split, earlier EPS
is on the old share basis while the price is on the new one, so each stock's earnings
yield was inflated by its *future* split factor (and, more mildly, by future dividends) —
a look-ahead leak. It affected value in both the baseline and every ML configuration;
momentum and quality were unaffected.

**How it was found and corrected.** The defect was in the backtesting engine this study
vendors ([factor-backtester](https://github.com/robbasnet14/factor-backtester) at
`31b92b4`). It surfaced while extending that engine with a size factor, where the same
as-filed-versus-adjusted mismatch made market capitalisation incomputable; the existing
value factor was found to carry it already. It was fixed there in `d68487a`. This study was then re-run in the exact
environment the published results came from (now pinned in `backtester/requirements.txt`):
first unchanged, reproducing all nine published return series exactly; then with the
published formula on re-downloaded data; then with the corrected formula. Same pipeline,
same pre-registered grid, every model refit in every fold, `n_trials` still 9.

**Three separate fragilities.** For `rf_depth10_leaf50`, the published best, with
everything else unchanged:

| Change | Out-of-sample Sharpe |
|---|---|
| As published | 0.241 |
| Same formula, data re-downloaded (data vintage) | 0.096 |
| scikit-learn 1.9 instead of 1.5.1 (unpinned dependency) | 0.147 |
| Leak corrected | −0.457 |

Only the leak is a bug; data vintage and library version are robustness failures. The
same data-vintage change moved the baseline only from 0.066 to 0.052.

**What it changes.** The published edge, which the paper had already traced to a single
12-month window, now has a documented cause: corrected, with nothing else changed, that
window's summed net monthly return falls from +0.280 to +0.001. Value had been the only
factor with a full-sample IC t-statistic above 1 (t = 1.26); corrected, it is −0.48.
Separately, the ticker-identity check was narrower than the published paper claimed: COL
is a counterexample showing ticker matching cannot give the guarantee a permanent security
identifier would.

An exploratory observation, not a finding: the ML configurations' Sharpe ratios fell
further than the baseline's (mean change −0.454 against −0.240, about 1.89×). It was
formed after seeing results, rests on n = 8, and paired tests find no significant
difference in mean monthly return impact for any configuration.

Every number above, with its source table: [`results/correction/FACTS.md`](results/correction/FACTS.md).
Full account: [PAPER.md, Corrections](PAPER.md#corrections).

## Could this design have found a real edge? (probably not, at any size worth caring about)

Independent of whether ML actually wins, we asked whether this study's design — 109
monthly out-of-sample observations, corrected for testing 9 configurations — could have
detected a genuine edge in the first place. Using the published best configuration's own
empirical skew and kurtosis (its returns are meaningfully negatively skewed and fat-tailed,
not close to normal; computed on the published series and not re-run for the correction),
the design needed an annualized out-of-sample Sharpe of roughly **1.5**
to clear the deflated-Sharpe bar. A genuine edge in the Sharpe 0.3–0.5 range — plausible
for a cost-aware, three-factor strategy — would have been **undetectable by this design at
this sample size, regardless of whether one exists.** The study fails to reject the null,
*and* the design could not have rejected it for any realistic effect size. Full derivation:
[PAPER.md, Section 5.4](PAPER.md#54-statistical-power-could-this-design-have-detected-a-real-edge).

## A limitation stated up front, not buried

**30.7% of point-in-time S&P 500 members (213 of 694) are excluded from this analysis for
lack of available price history**, disproportionately delisted, acquired, and renamed
names. **The direction of the resulting bias is ambiguous for this dollar-neutral
long/short design, and we do not attempt to sign it** — names that went bankrupt would
mostly have sat in the short leg (so dropping them removes short-leg gains a real investor
would have earned), while names acquired at a premium would have moved sharply against a
short position (so dropping them removes short-leg losses a real investor would have
taken). These pull in opposite directions. A Tiingo API key was obtained and spot-checked
on 5 confirmed-delisted names spanning the gap; it had no price history for any of them
(a structural gap in that tier's historical depth, not a code or auth issue — verified
before giving up on the fallback). Full detail: [`results/coverage_gap_note.md`](results/coverage_gap_note.md).

## Pre-registered

The question, hypothesis (stated as a null), exact 9-configuration model grid, and win
condition were committed to [`PREREGISTRATION.md`](PREREGISTRATION.md) at commit
[`02647d5`](https://github.com/robbasnet14/ml-vs-linear-factors/commit/02647d5) on
**2026-07-07** — three weeks before any machine-learning model was trained (first ML
commit `eac614c`, 2026-07-28). Deviations from that original plan (a baseline restatement,
the Tiingo outcome, a threshold that was described in words but never numerically fixed,
and the 2026-10-07 correction) are logged, dated, in a `## Deviations from the original
plan` section appended to that same file — the original text was never edited.

## Repo map

```
PREREGISTRATION.md   the pre-registered question, hypothesis, model grid, win condition
PAPER.md             the full paper (abstract through appendix)
paper/               PAPER.md rendered to PDF
backtester/          the backtesting engine + ML pipeline (see note below)
  src/
    data/            price (Yahoo/Tiingo) + fundamentals (SEC EDGAR) loading, point-in-time universe
    features/        factor calculations, cross-sectional standardization
    backtest/        portfolio construction, cost model, engine, walk-forward validation
    ml/              ML dataset builder, walk-forward model training, permutation importance
    analytics/       metrics (Sharpe, deflated Sharpe, factor IC), coverage report, plots
  scripts/           one script per analysis step (baseline, ML training, comparison,
                     gross-vs-net, statistical power, leave-one-out, feature importance,
                     cost sensitivity, factor IC) — see results/comparison/README.md
  tests/             85 pytest tests, network-mocked
results/
  baseline/          Step 3: linear composite walk-forward OOS reference
  ml/                Steps 4-5: all 8 ML configurations' walk-forward OOS results
  comparison/        Steps 6-7: the comparison table + every robustness check above
  correction/        2026-10-07 re-run: published / re-downloaded / corrected, FACTS.md
  coverage_gap_note.md, ticker_identity_check.md   data-quality investigations
```

**`backtester/` is a vendored snapshot** of the standalone backtesting engine, brought in
at commit `eac614c` ("Bring in factor-backtester code as reused baseline"). The
maintained, independently-evolving version of that engine lives at
[github.com/robbasnet14/factor-backtester](https://github.com/robbasnet14/factor-backtester)
— check there for engine-level fixes made after that commit.

## Reproducing

```bash
cd backtester
pip install -r requirements.txt
python -m pytest -q                                          # 85 tests
PYTHONPATH=. python scripts/run_backtest.py --config config.yaml
PYTHONPATH=. python scripts/run_ml_experiment.py --config config.yaml
PYTHONPATH=. python scripts/compare_ml_vs_baseline.py
PYTHONPATH=. python scripts/gross_vs_net.py
PYTHONPATH=. python scripts/dsr_power_table.py
PYTHONPATH=. python scripts/feature_importance.py
PYTHONPATH=. python scripts/subperiod_table.py
PYTHONPATH=. python scripts/cost_sensitivity.py
PYTHONPATH=. python scripts/factor_ic.py
```

The correction re-run (offline, from the local data cache, about 20 minutes per mode) is
documented in [`results/correction/README.md`](results/correction/README.md). The pinned
versions in `requirements.txt` matter: the published results reproduce bit-for-bit only
with them.

Prices come from Yahoo Finance (no key required); fundamentals from SEC EDGAR (no key
required). A `TIINGO_KEY` env var enables an optional price fallback for some
delisted names — see the coverage-gap note above for why it doesn't close the full gap.

## Full write-up

[PAPER.md](PAPER.md) — abstract through appendix, including the statistical power
analysis (what Sharpe this design could and couldn't have detected), the factor-level
rank IC table, and every self-correction made along the way. A rendered PDF is at
[paper/ml-vs-linear-factors.pdf](paper/ml-vs-linear-factors.pdf).
