# Ticker-identity contamination check (review follow-up, verified clean)

**Question:** does any base ticker in the panel carry price data from a
*different real-world entity* than the one that was actually a point-in-time
S&P 500 member — e.g. new "Dow Inc." (spun off 2019) prices attributed to
old Dow Chemical's 2010-2017 membership window? That would be wrong data,
not missing data, and would actively bias returns rather than just shrink
the sample.

## Method

`backtester/scripts/ticker_reuse_check.py`:

1. Find every base ticker with **more than one separate (non-contiguous)
   True-membership run** in the universe matrix — a gap, then a
   reappearance, is the only structural pattern under which one ticker
   symbol could span two different companies. 3 found: **AMD, DXC, GAS**.
2. Add the 17 previously-flagged "missing but not in `unavailable_prices.json`"
   tickers (`ADT, BEAM, CEG, DELL, DOW, EMC, FCPT, JAVA, MHS, MI, MMI, NE,
   NSM, S, SE, SHLD, TE`) — the other place a silent entity-swap could hide,
   since their price fetch didn't fail outright.
3. For each of these 20 names, compare the ticker's actual price-data date
   range against each of its membership window(s), and check for
   month-over-month price-ratio discontinuities anywhere in the series.

## Result: clean

- **AMD and DXC** (real companies that left and rejoined the S&P 500 itself,
  not ticker reuse by a different entity) have real price coverage inside
  *both* of their membership windows (44 + 93 points for AMD, 71 + 92 for
  DXC) — expected and correct, not a bug.
- **GAS** returned no price data at all — already accounted for in
  `unavailable_prices.json`.
- **All 17 of the previously-flagged names — including DOW, DELL, CEG,
  EMC — have ZERO price-data overlap with their old membership window.**
  In every case the earliest available price postdates the membership
  window's end (e.g. `DOW`: membership window ends 2017-09-01, price data
  only starts 2019-03-31 — the real-world gap between old Dow Chemical's
  merger into DowDuPont and new Dow Inc.'s 2019 spinoff-relisting under the
  same ticker). Two names (`MI`, `TE`) show large price-ratio jumps
  (10.9x, 2.1x), but both occur entirely within the *new* entity's own
  post-relisting price history, outside any membership window — so they
  cannot have entered the panel or been attributed to the wrong entity.

**Conclusion:** the reviewer's "expected clean" was correct, and it's now
verified for every ticker where the reuse pattern could structurally occur
— zero cases of a valid membership row paired with a different entity's
price. The earlier 213-ticker gap is entirely a *missing-data* problem
(confirmed again here), not a *wrong-data* problem.

## Scope and limitation (name it properly, per the review)

This check is exhaustive for the two patterns that could produce
contamination (multi-run tickers; tickers whose price fetch quietly
"succeeded" but excluded the old window) — not a brute-force check of all
694 tickers' entire price histories against a ground truth. Base-ticker
string matching fundamentally cannot rule out a hypothetical case where a
reused ticker's new entity started trading *during* the old entity's
window with overlapping dates — implausible for a real delisting/relisting
gap, but not something string matching on the ticker symbol alone can
prove impossible in general. **The correct permanent fix is matching on a
stable identifier (CIK for the SEC-fundamentals side, or CRSP PERMNO for
prices) instead of ticker symbol**, which this codebase does not do. This
is a real limitation to disclose in the paper's Data section, independent
of whether `TIINGO_KEY` gets set.

## Reproducing

```bash
cd backtester
PYTHONPATH=. python scripts/ticker_reuse_check.py
```
