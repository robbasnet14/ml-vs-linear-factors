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

It runs entirely from the local data cache. The vendored loader would otherwise
re-query Yahoo for every ticker on every run (its cache-miss bug, fixed in
factor-backtester after the vendored commit) and permanently skiplist a ticker
whenever a request fails -- which is what happened during this correction's
own first measurement runs: running modes in parallel got requests throttled,
and MCHP, KDP, HPQ, COST and HD were dropped mid-run despite having cached
prices. Those queries never added or changed a cached row, so serving the cache
directly is the published behaviour without the failure mode; the 'published'
mode reproducing every published series exactly is the check.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["published", *PANELS], required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(BACKTESTER))
    import src.data.loader as loader
    import src.features.factors as factors

    # Offline: cached data only, and the skiplist is never written.
    loader._fetch_yfinance_prices = lambda *a, **k: loader._empty_price_frame()
    loader._fetch_tiingo_prices = lambda *a, **k: loader._empty_price_frame()
    loader._sec_get = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline re-run: no network"))
    loader._save_skiplist = lambda *a, **k: None

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

        sys.argv = ["run_backtest.py", "--config", "config.yaml"]
        runpy.run_path(str(BACKTESTER / "scripts" / "run_backtest.py"), run_name="__main__")

        # Keep each configuration's holdings too (needed for the fold-7 attribution).
        ml = runpy.run_path(str(BACKTESTER / "scripts" / "run_ml_experiment.py"), run_name="correction")
        main_globals = ml["main"].__globals__
        original_run = main_globals["run_one_config"]
        (out_dir / "holdings").mkdir(exist_ok=True)

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
