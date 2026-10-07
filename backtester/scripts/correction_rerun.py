"""Re-run the study with the value feature swapped, for the EPS split-basis correction.

The published value factor divided EPS as filed by a price adjusted for every
later split and dividend, a look-ahead leak (see results/correction/). This
script re-runs the unchanged published pipeline -- scripts/run_backtest.py,
the full pre-registered grid in scripts/run_ml_experiment.py, and
scripts/factor_ic.py -- with one thing changed: `src.features.factors.value`
returns a precomputed earnings-yield panel instead of computing its own,
reindexed to exactly the shape the original function returns. Every model is
refit per fold on the features it is given; nothing fitted is reused.

Modes:
  published   no swap -- reproduces the published results
  oldformula  the old (leaky) formula, rebuilt from freshly re-downloaded data
              -- isolates data drift from the fix
  corrected   EPS restated across splits, divided by a split-adjusted (not
              dividend-adjusted) price

It runs entirely from the local data cache, read-only, and any failure to read
it stops the run. The vendored loader fails silently in two ways this guards
against, both of which hit this correction's own measurement runs:

- Online, it re-queries Yahoo for every ticker on every run (its cache-miss
  bug, fixed in factor-backtester after the vendored commit) and permanently
  skiplists a ticker whenever a request fails: parallel runs got throttled
  and dropped MCHP, KDP, HPQ, COST and HD despite their cached prices.
- It rewrites each ticker's price file on every load, and its fallback
  catches every Exception: a run reading a file another run was rewriting
  lost that ticker without an error (the first committed corrected baseline
  was affected).

So here prices and fundamentals are served from the cache without writing,
the skiplist is never written, and a cache read error raises an error that is
deliberately not an Exception subclass, so the loader's `except Exception`
can't turn it into a silently skipped ticker. Modes can run in parallel. The
vendored network queries never added or changed a cached row, so this is the
published behaviour without the failure modes; the 'published' mode
reproducing every published series exactly is the check.

The two panels are built by factor-backtester at commit d68487a (see
results/correction/inputs/build_value_panels.py). The published scripts write
to ./outputs and ../results/comparison relative to where they run, so they run
in a temporary directory here and only their outputs are copied to
results/correction/<mode>/ -- nothing published is overwritten.

Run from backtester/, in the pinned environment (requirements.txt):
  PYTHONPATH=. python scripts/correction_rerun.py --mode corrected
"""
import argparse
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

BACKTESTER = Path(__file__).resolve().parent.parent
CORRECTION = BACKTESTER.parent / "results" / "correction"
PANELS = {
    "oldformula": CORRECTION / "inputs" / "value_oldformula_d68487a.parquet",
    "corrected": CORRECTION / "inputs" / "value_corrected_d68487a.parquet",
}


class CacheReadError(BaseException):
    """Not an Exception subclass on purpose: the vendored loader catches
    Exception and quietly drops the ticker, which must not happen here."""


def _read(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise CacheReadError(f"could not read {path}: {e}") from e


def _read_only_price_series(loader):
    """The vendored _load_one_price_series with an empty fetch (offline),
    minus the write: same rows, never touches the file."""
    def load(ticker, start_ts, end_ts, cache_dir, fetch_fn):
        path = loader._cache_path(cache_dir, "prices", ticker)
        cached = _read(path) if path.exists() else loader._empty_price_frame()
        if cached.empty or not (cached["date"].min() <= start_ts and cached["date"].max() >= end_ts):
            cached = cached.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        sliced = cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)].copy()
        sliced.insert(1, "ticker", ticker.upper())
        return sliced[["date", "ticker", "adj_close"]]
    return load


def _read_only_fundamentals_series(loader, original):
    """Cached fundamentals read through `_read` (fatal on error); a ticker with
    no cache file goes to the original, whose SEC fetch fails offline exactly
    as an unreachable SEC would."""
    def load(ticker, start_ts, end_ts, lag_days, cache_dir, ticker_to_cik):
        path = loader._cache_path(cache_dir, "fundamentals", ticker)
        if not path.exists():
            return original(ticker, start_ts, end_ts, lag_days, cache_dir, ticker_to_cik)
        raw = _read(path).copy()
        raw["date"] = raw["report_date"] + pd.Timedelta(days=lag_days)
        raw.insert(1, "ticker", ticker.upper())
        mask = (raw["date"] >= start_ts) & (raw["date"] <= end_ts)
        return raw.loc[mask, ["date", "ticker", "report_date", "earnings", "book_value", "roe"]]
    return load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["published", *PANELS], required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(BACKTESTER))
    import src.data.loader as loader
    import src.features.factors as factors

    # Offline and read-only: cached data only, nothing written, read errors fatal.
    loader._fetch_yfinance_prices = lambda *a, **k: loader._empty_price_frame()
    loader._fetch_tiingo_prices = lambda *a, **k: loader._empty_price_frame()
    loader._sec_get = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline re-run: no network"))
    loader._save_skiplist = lambda *a, **k: None
    loader._load_one_price_series = _read_only_price_series(loader)
    loader._load_one_fundamentals_series = _read_only_fundamentals_series(loader, loader._load_one_fundamentals_series)

    if args.mode in PANELS:
        panel = pd.read_parquet(PANELS[args.mode])
        real_value = factors.value

        def value_from_panel(fundamentals, prices, metric="earnings_yield"):
            original = real_value(fundamentals, prices, metric)
            return panel.reindex(index=original.index, columns=original.columns)

        factors.value = value_from_panel

    out_dir = CORRECTION / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        work.mkdir()
        cfg = yaml.safe_load((BACKTESTER / "config.yaml").read_text())
        cfg["data"]["cache_dir"] = str(BACKTESTER / cfg["data"]["cache_dir"])
        (work / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        os.chdir(work)

        # Keep the baseline's holdings: run_backtest.py imports
        # walk_forward_backtest from this module when it runs, so wrapping the
        # module attribute captures the weights it is called with, unchanged.
        import src.backtest.validation as validation
        (out_dir / "holdings").mkdir(exist_ok=True)
        real_walk_forward = validation.walk_forward_backtest

        def walk_forward_keeping_holdings(weights, *a, **k):
            weights.to_parquet(out_dir / "holdings" / "weights_baseline.parquet")
            return real_walk_forward(weights, *a, **k)

        validation.walk_forward_backtest = walk_forward_keeping_holdings
        sys.argv = ["run_backtest.py", "--config", "config.yaml"]
        runpy.run_path(str(BACKTESTER / "scripts" / "run_backtest.py"), run_name="__main__")
        validation.walk_forward_backtest = real_walk_forward

        # Keep each configuration's holdings too (fold-7 attribution; turnover
        # in scripts/compare_ml_vs_baseline.py).
        ml = runpy.run_path(str(BACKTESTER / "scripts" / "run_ml_experiment.py"), run_name="correction")
        main_globals = ml["main"].__globals__
        original_run = main_globals["run_one_config"]

        def run_and_keep_holdings(model_cfg, *a, **k):
            oos_returns, weights = original_run(model_cfg, *a, **k)
            weights.to_parquet(out_dir / "holdings" / f"weights_{model_cfg['name']}.parquet")
            return oos_returns, weights

        main_globals["run_one_config"] = run_and_keep_holdings
        sys.argv = ["run_ml_experiment.py", "--config", "config.yaml"]
        ml["main"]()

        sys.argv = ["factor_ic.py"]
        runpy.run_path(str(BACKTESTER / "scripts" / "factor_ic.py"), run_name="__main__")

        for f in sorted((work / "outputs").glob("*oos_net_returns*.csv")):
            shutil.copy(f, out_dir / f.name)
        shutil.copy(Path(tmp) / "results" / "comparison" / "factor_ic.csv", out_dir / "factor_ic.csv")

    print(f"\nWrote {args.mode} results to {out_dir}")


if __name__ == "__main__":
    main()
