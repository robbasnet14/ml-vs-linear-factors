"""Entry point for the ML challenger — exactly parallel to run_backtest.py,
except the composite score comes from a walk-forward-fit model instead of
an equal-weight average of the same three factors.

Runs every configuration committed in PREREGISTRATION.md (src.ml.model.
MODEL_CONFIGS) by default, since that's the full grid the pre-registration
promises to try — pass --model-config to run just one during development.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.analytics.metrics import (
    annualized_return,
    annualized_volatility,
    average_turnover,
    deflated_sharpe,
    hit_rate,
    max_drawdown,
    sharpe,
)
from src.analytics.plots import plot_equity_curve
from src.backtest.portfolio import decile_portfolios
from src.backtest.validation import walk_forward_backtest
from src.data.loader import load_fundamentals, load_prices
from src.data.universe import build_universe
from src.ml.dataset import build_ml_panel
from src.ml.model import MODEL_CONFIGS, walk_forward_ml_scores
from src.utils.config import load_config

PERIODS_PER_YEAR = 12  # monthly rebalance

# The pre-registration commits to 9 total configurations (8 ML + the 1
# linear baseline) before looking at any result — this is what the deflated
# Sharpe's selection-bias correction is scored against for the ML side.
N_TRIALS = 9


def print_summary(title: str, returns: pd.Series, weights: pd.DataFrame, n_trials: int):
    weights_realized = weights.reindex(returns.index)
    summary = {
        "Annualized return": f"{annualized_return(returns, PERIODS_PER_YEAR):.2%}",
        "Annualized volatility": f"{annualized_volatility(returns, PERIODS_PER_YEAR):.2%}",
        "Sharpe ratio": f"{sharpe(returns, PERIODS_PER_YEAR):.2f}",
        "Max drawdown": f"{max_drawdown(returns):.2%}",
        "Hit rate": f"{hit_rate(returns):.2%}",
        "Avg monthly turnover": f"{average_turnover(weights_realized):.2%}",
        f"Deflated Sharpe (n_trials={n_trials})": f"{deflated_sharpe(returns, n_trials):.3f}",
    }
    label_width = max(len(k) for k in summary)
    print(f"\n{title} ({len(returns)} periods)")
    print("-" * (label_width + 12))
    for label, value_str in summary.items():
        print(f"{label:<{label_width}}  {value_str:>10}")


def run_one_config(model_cfg: dict, panel: pd.DataFrame, forward_returns: pd.DataFrame, monthly_prices: pd.DataFrame, cfg: dict, out_dir: Path):
    uni_cfg, port_cfg, cost_cfg = cfg["universe"], cfg["portfolio"], cfg["costs"]
    start, end = uni_cfg["start_date"], uni_cfg["end_date"]
    wf_cfg = cfg.get("validation", {}).get("walk_forward", {})

    oos_scores = walk_forward_ml_scores(
        panel,
        start=start,
        end=end,
        model=model_cfg["model"],
        params=model_cfg["params"],
        initial_train_months=wf_cfg.get("initial_train_months", 60),
        test_months=wf_cfg.get("test_months", 12),
        embargo_months=wf_cfg.get("embargo_months", 1),
    )

    weights = decile_portfolios(oos_scores, n_deciles=port_cfg["n_deciles"], long_short=port_cfg["long_short"])

    # Reusing walk_forward_backtest (not a raw run_backtest call) here is
    # what makes this "exactly parallel" to the linear pipeline: it costs
    # each fold's test block independently, starting from cash, exactly as
    # it does for the baseline — see src/backtest/validation.py's docstring.
    oos_returns, folds = walk_forward_backtest(
        weights,
        forward_returns,
        cost_bps=cost_cfg["bps_per_trade"],
        start=start,
        end=end,
        initial_train_months=wf_cfg.get("initial_train_months", 60),
        test_months=wf_cfg.get("test_months", 12),
        embargo_months=wf_cfg.get("embargo_months", 1),
        prices=monthly_prices,
    )

    out_path = out_dir / f"ml_oos_net_returns_{model_cfg['name']}.csv"
    oos_returns.to_csv(out_path, header=True)
    print_summary(f"ML challenger '{model_cfg['name']}' — walk-forward OUT-OF-SAMPLE", oos_returns, weights, N_TRIALS)

    return oos_returns, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--model-config",
        default=None,
        help=f"Run only this configuration by name (one of {[c['name'] for c in MODEL_CONFIGS]}); default runs all 8.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    uni_cfg, data_cfg = cfg["universe"], cfg["data"]
    factor_cfg = cfg["factors"]
    start, end, cache_dir = uni_cfg["start_date"], uni_cfg["end_date"], data_cfg["cache_dir"]

    print("Loaded config:", uni_cfg["name"], start, "->", end)

    universe = build_universe(uni_cfg["name"], start, end, cache_dir=cache_dir)
    tickers = sorted(universe.columns[universe.any(axis=0)])
    print(f"Universe: {len(tickers)} distinct tickers over the period")

    prices = load_prices(tickers, start, end, cache_dir=cache_dir)
    fundamentals = load_fundamentals(tickers, start, end, lag_days=data_cfg["fundamentals_lag_days"], cache_dir=cache_dir)

    panel = build_ml_panel(prices, fundamentals, universe, factor_cfg)
    print(f"ML panel: {len(panel)} (date, ticker) rows")

    # Identical computation to scripts/run_backtest.py, so both pipelines
    # trade on exactly the same forward-return panel.
    monthly_prices = prices.pivot(index="date", columns="ticker", values="adj_close").resample("ME").last()
    forward_returns = monthly_prices.pct_change().shift(-1)
    forward_returns = forward_returns.dropna(how="all")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_config:
        configs_to_run = [c for c in MODEL_CONFIGS if c["name"] == args.model_config]
        if not configs_to_run:
            raise ValueError(f"--model-config {args.model_config!r} not found in MODEL_CONFIGS")
    else:
        configs_to_run = MODEL_CONFIGS

    results = {}
    for model_cfg in configs_to_run:
        oos_returns, weights = run_one_config(model_cfg, panel, forward_returns, monthly_prices, cfg, out_dir)
        results[model_cfg["name"]] = oos_returns

    # The literal Step 5 deliverable: one canonical file directly comparable
    # to outputs/oos_net_returns.csv, mirroring the first configuration run
    # (or the single one selected via --model-config).
    primary_name = configs_to_run[0]["name"]
    canonical_path = out_dir / "ml_oos_net_returns.csv"
    shutil.copyfile(out_dir / f"ml_oos_net_returns_{primary_name}.csv", canonical_path)
    print(f"\nCanonical ML result -> {canonical_path} (configuration: {primary_name})")

    plot_equity_curve(results[primary_name], path=str(out_dir / "ml_equity_curve_oos.png"))
    print(f"Saved {out_dir / 'ml_equity_curve_oos.png'} (walk-forward OOS, configuration: {primary_name})")

    if len(results) > 1:
        print("\nAll configurations (walk-forward OOS Sharpe):")
        for name, oos_returns in results.items():
            print(f"  {name:<20} Sharpe {sharpe(oos_returns, PERIODS_PER_YEAR):.3f}")


if __name__ == "__main__":
    main()
