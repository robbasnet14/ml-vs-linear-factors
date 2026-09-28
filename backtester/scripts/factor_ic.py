"""Full-sample factor IC check (review follow-up): does momentum, value, or
quality, on its own, have a cross-sectional relationship with next-month
returns distinguishable from zero?

Builds the ML panel via the existing `build_ml_panel` (reuses the exact
same factor pipeline and point-in-time universe restriction Steps 2-7 run
on -- nothing here recomputes a factor differently). For each factor,
computes the monthly rank IC via `src.analytics.factor_ic.monthly_rank_ic`
and summarizes it (n, mean, std, t-stat) via `summarize_ic`.

This is a full-sample descriptive statistic about the factors themselves,
including periods the ML models were trained on -- not an out-of-sample
performance claim like the walk-forward results in results/comparison/.
"""
from pathlib import Path

import pandas as pd

from src.analytics.factor_ic import monthly_rank_ic, summarize_ic
from src.data.loader import load_fundamentals, load_prices
from src.data.universe import build_universe
from src.ml.dataset import build_ml_panel
from src.utils.config import load_config

FACTORS = ["mom_z", "val_z", "qual_z"]


def main():
    cfg = load_config("config.yaml")
    uni_cfg, data_cfg, factor_cfg = cfg["universe"], cfg["data"], cfg["factors"]
    start, end, cache_dir = uni_cfg["start_date"], uni_cfg["end_date"], data_cfg["cache_dir"]

    universe = build_universe(uni_cfg["name"], start, end, cache_dir=cache_dir)
    tickers = sorted(universe.columns[universe.any(axis=0)])
    prices = load_prices(tickers, start, end, cache_dir=cache_dir)
    fundamentals = load_fundamentals(tickers, start, end, lag_days=data_cfg["fundamentals_lag_days"], cache_dir=cache_dir)

    # build_ml_panel already restricts every row to point-in-time universe
    # membership (src/ml/dataset.py) -- no separate filtering needed here.
    panel = build_ml_panel(prices, fundamentals, universe, factor_cfg)
    print(f"Panel: {len(panel)} rows, {panel['date'].nunique()} dates, {panel['ticker'].nunique()} tickers")

    rows = []
    for factor in FACTORS:
        ic_series = monthly_rank_ic(panel, factor)
        summary = summarize_ic(ic_series)
        rows.append({"factor": factor, **summary})

    table = pd.DataFrame(rows).set_index("factor")
    out_path = Path("../results/comparison/factor_ic.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path)

    display = table.copy()
    display["mean"] = display["mean"].map(lambda v: f"{v:.4f}")
    display["std"] = display["std"].map(lambda v: f"{v:.4f}")
    display["t_stat"] = display["t_stat"].map(lambda v: f"{v:.2f}")
    print("\nFull-sample monthly cross-sectional rank IC (factor vs. next-month return):")
    print(display.to_string())
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
