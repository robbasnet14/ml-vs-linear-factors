"""Build the two earnings-yield panels used by backtester/scripts/correction_rerun.py.

Runs inside a checkout of github.com/robbasnet14/factor-backtester at commit
d68487a (the engine version with the split-basis fix), with its price and
fundamentals cache populated, and refuses to run anywhere else:

  cd factor-backtester && git checkout d68487a
  python /path/to/ml-vs-linear-factors/results/correction/inputs/build_value_panels.py \
      --out /path/to/ml-vs-linear-factors/results/correction/inputs

Writes, for the S&P 500 point-in-time universe over 2010-01-01..2024-12-31
(the study's config), month-end x ticker panels of earnings yield:

  value_corrected_d68487a.parquet   each quarter's EPS restated across later
                                    splits, TTM, divided by a split-adjusted
                                    (not dividend-adjusted) close
  value_oldformula_d68487a.parquet  the published (leaky) formula on the same
                                    data: EPS as filed, TTM, divided by the
                                    split- and dividend-adjusted close

Both use a 90-day fundamentals lag, as in the study's config.yaml. Also writes
the two supporting inputs the correction's tables use:

  split_events_d68487a.csv          every split 2008-2024 in that cache,
                                    forward and reverse (ticker, date,
                                    split_ratio) -- for the
                                    fold-7 attribution
  col_monthly_d68487a.csv           month-end adjusted close of COL (Rockwell
                                    Collins) -- for the ticker-identity check
"""
import argparse
import subprocess
import sys
from pathlib import Path

ENGINE_COMMIT = "d68487a"
START, END, LAG_DAYS = "2010-01-01", "2024-12-31", 90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    head = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    if head != ENGINE_COMMIT or dirty:
        sys.exit(f"Run this from a clean factor-backtester checkout at {ENGINE_COMMIT} (found {head!r}, dirty={bool(dirty)})")

    sys.path.insert(0, ".")
    import src.features.factors as factors
    from src.data.loader import load_fundamentals, load_prices
    from src.data.universe import build_universe

    universe = build_universe("SP500", START, END, cache_dir="data_cache")
    tickers = sorted(universe.columns[universe.any(axis=0)])
    history = load_prices(tickers, "2008-01-01", END, "data_cache")  # two years early: covers the oldest TTM quarter
    prices = history[history["date"] >= START].reset_index(drop=True)
    splits = history[["date", "ticker", "split_ratio"]]

    restated = load_fundamentals(tickers, START, END, lag_days=LAG_DAYS, cache_dir="data_cache", splits=splits)
    corrected = factors.value(restated, prices)

    as_filed = load_fundamentals(
        tickers, START, END, lag_days=LAG_DAYS, cache_dir="data_cache", splits=splits.assign(split_ratio=1.0)
    )
    old_formula = factors.value(as_filed, prices.assign(close=prices["adj_close"]))

    out = Path(args.out)
    corrected.to_parquet(out / f"value_corrected_{ENGINE_COMMIT}.parquet")
    old_formula.to_parquet(out / f"value_oldformula_{ENGINE_COMMIT}.parquet")

    events = history[(history["split_ratio"] != 1) & history["split_ratio"].notna()][["ticker", "date", "split_ratio"]]
    events.sort_values(["ticker", "date"]).to_csv(out / f"split_events_{ENGINE_COMMIT}.csv", index=False)
    col = history[history["ticker"] == "COL"].set_index("date")["adj_close"].resample("ME").last()
    col.rename("adj_close").to_csv(out / f"col_monthly_{ENGINE_COMMIT}.csv")
    print(f"corrected {corrected.shape}, old formula {old_formula.shape}, {len(events)} split events -> {out}")


if __name__ == "__main__":
    main()
