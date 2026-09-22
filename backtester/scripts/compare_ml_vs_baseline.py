"""Step 6 — Compare: build the baseline-vs-ML comparison table and the
overlaid OOS equity-curve chart the pre-registration's Step 6 calls for.

Reads the already-saved stitched walk-forward OOS return series from
Step 3 (results/baseline/oos_net_returns.csv) and Steps 4-5 (each
results/ml/ml_oos_net_returns_<config>.csv) rather than re-running the
network-dependent data pipeline, and recomputes Sharpe / deflated Sharpe /
annualized return / max drawdown / hit rate directly from those return
series with the same functions run_backtest.py and run_ml_experiment.py
use — so this table is a check against, not a copy of, results/*/README.md.

Turnover is the one metric that can't be recovered from a saved return
series alone (it needs the per-period weights, which weren't persisted).
Those figures are carried over verbatim from the Step 3/Step 5 run
(results/baseline/README.md, results/ml/README.md) and are not
recomputed here; TURNOVER_PCT below documents that source.
"""
from pathlib import Path

import pandas as pd

from src.analytics.metrics import annualized_return, deflated_sharpe, hit_rate, max_drawdown, sharpe
from src.analytics.plots import plot_equity_curve

PERIODS_PER_YEAR = 12
N_TRIALS = 9  # 8 ML configs + the 1 pre-specified linear baseline, per PREREGISTRATION.md

RESULTS_DIR = Path("../results")
BASELINE_CSV = RESULTS_DIR / "baseline" / "oos_net_returns.csv"
ML_DIR = RESULTS_DIR / "ml"
OUT_DIR = RESULTS_DIR / "comparison"

# Sourced verbatim from results/baseline/README.md and results/ml/README.md
# (the Step 3 / Step 5 run) — see module docstring for why this isn't
# recomputed here.
TURNOVER_PCT = {
    "baseline": 37.12,
    "gbm_lr0.03_depth3": 71.59,
    "gbm_lr0.03_depth5": 89.21,
    "gbm_lr0.1_depth3": 79.07,
    "gbm_lr0.1_depth5": 97.27,
    "rf_depth5_leaf50": 82.48,
    "rf_depth5_leaf200": 81.50,
    "rf_depth10_leaf50": 100.57,
    "rf_depth10_leaf200": 100.38,
}

CONFIG_FILES = {
    "gbm_lr0.03_depth3": ML_DIR / "ml_oos_net_returns_gbm_lr0.03_depth3.csv",
    "gbm_lr0.03_depth5": ML_DIR / "ml_oos_net_returns_gbm_lr0.03_depth5.csv",
    "gbm_lr0.1_depth3": ML_DIR / "ml_oos_net_returns_gbm_lr0.1_depth3.csv",
    "gbm_lr0.1_depth5": ML_DIR / "ml_oos_net_returns_gbm_lr0.1_depth5.csv",
    "rf_depth5_leaf50": ML_DIR / "ml_oos_net_returns_rf_depth5_leaf50.csv",
    "rf_depth5_leaf200": ML_DIR / "ml_oos_net_returns_rf_depth5_leaf200.csv",
    "rf_depth10_leaf50": ML_DIR / "ml_oos_net_returns_rf_depth10_leaf50.csv",
    "rf_depth10_leaf200": ML_DIR / "ml_oos_net_returns_rf_depth10_leaf200.csv",
}


def _load_returns(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["net_return"]


def _row(name: str, returns: pd.Series, n_trials: int) -> dict:
    return {
        "config": name,
        "oos_sharpe": sharpe(returns, PERIODS_PER_YEAR),
        "deflated_sharpe": deflated_sharpe(returns, n_trials),
        "n_trials": n_trials,
        "ann_return_pct": annualized_return(returns, PERIODS_PER_YEAR) * 100,
        "max_drawdown_pct": max_drawdown(returns) * 100,
        "hit_rate_pct": hit_rate(returns) * 100,
        "turnover_pct": TURNOVER_PCT[name],
        "n_periods": len(returns.dropna()),
    }


def build_table() -> pd.DataFrame:
    baseline_returns = _load_returns(BASELINE_CSV)
    rows = [_row("baseline", baseline_returns, n_trials=1)]

    ml_returns = {}
    for name, path in CONFIG_FILES.items():
        r = _load_returns(path)
        ml_returns[name] = r
        rows.append(_row(name, r, n_trials=N_TRIALS))

    table = pd.DataFrame(rows).set_index("config")
    return table, baseline_returns, ml_returns


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, baseline_returns, ml_returns = build_table()

    table_path = OUT_DIR / "comparison_table.csv"
    table.to_csv(table_path)
    print(f"Saved {table_path}\n")

    display = table.copy()
    display["oos_sharpe"] = display["oos_sharpe"].map(lambda v: f"{v:.3f}")
    display["deflated_sharpe"] = display["deflated_sharpe"].map(lambda v: f"{v:.3f}")
    display["ann_return_pct"] = display["ann_return_pct"].map(lambda v: f"{v:.2f}%")
    display["max_drawdown_pct"] = display["max_drawdown_pct"].map(lambda v: f"{v:.2f}%")
    display["hit_rate_pct"] = display["hit_rate_pct"].map(lambda v: f"{v:.2f}%")
    display["turnover_pct"] = display["turnover_pct"].map(lambda v: f"{v:.2f}%")
    print(display.to_string())

    # Best ML config by OOS Sharpe (the pre-registration's headline metric).
    ml_table = table.drop(index="baseline")
    best_name = ml_table["oos_sharpe"].idxmax()
    print(f"\nBest ML config by OOS Sharpe: {best_name}")

    win = (
        ml_table.loc[best_name, "oos_sharpe"] > table.loc["baseline", "oos_sharpe"]
        and ml_table.loc[best_name, "deflated_sharpe"] > 0.95
    )
    print(f"Win condition (Sharpe beats baseline AND deflated Sharpe > 0.95): {'YES' if win else 'NO'}")

    chart_path = OUT_DIR / "equity_overlay_baseline_vs_best_ml.png"
    plot_equity_curve(
        ml_returns[best_name],
        benchmark=baseline_returns,
        path=str(chart_path),
        strategy_label=f"ML: {best_name}",
        benchmark_label="Linear baseline (equal-weight)",
        title="Out-of-sample growth of $1, net of costs: linear baseline vs best ML",
    )
    print(f"Saved {chart_path}")


if __name__ == "__main__":
    main()
