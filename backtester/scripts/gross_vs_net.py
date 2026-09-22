"""Gross-of-cost vs net-of-cost OOS Sharpe for the baseline and all 8 ML
configs (review follow-up: "is the signal real but untradeable at 8bps, or
just not real?").

Mirrors `walk_forward_backtest`'s fold loop exactly (same
`make_walk_forward_folds` boundaries, same "cost each fold's test block as
if starting from cash" convention) but calls the new
`run_backtest_breakdown` per fold instead of `run_backtest`, so gross and
net are stitched from the identical weights/returns/fold splits already
used for the committed results — this is not a different pipeline, just
the same one reporting one more number per period.

net_return here is cross-checked against results/baseline/oos_net_returns.csv
and results/ml/ml_oos_net_returns_<config>.csv (produced by
walk_forward_backtest) and must match to the last basis point; any mismatch
means this script drifted from the committed pipeline and should not be
trusted over the committed results.
"""
from pathlib import Path

import pandas as pd

from src.analytics.metrics import deflated_sharpe, sharpe
from src.backtest.engine import run_backtest_breakdown
from src.backtest.portfolio import decile_portfolios
from src.backtest.validation import make_walk_forward_folds
from src.data.loader import load_fundamentals, load_prices
from src.data.universe import build_universe
from src.features.factors import momentum, quality, value
from src.features.transforms import combine_factors, zscore_cross_section
from src.ml.dataset import build_ml_panel
from src.ml.model import MODEL_CONFIGS, walk_forward_ml_scores
from src.utils.config import load_config

PERIODS_PER_YEAR = 12
N_TRIALS = 9


def walk_forward_breakdown(weights, forward_returns, cost_bps, start, end, initial_train_months, test_months, embargo_months, prices):
    """Same fold loop as `src.backtest.validation.walk_forward_backtest`,
    stitching `run_backtest_breakdown`'s gross/cost/net columns instead of
    just net."""
    folds = make_walk_forward_folds(start, end, initial_train_months, test_months, embargo_months)
    segments = []
    for fold in folds:
        test_dates = weights.index[(weights.index >= fold["test_start"]) & (weights.index < fold["test_end"])]
        if test_dates.empty:
            continue
        fold_prices = prices.reindex(test_dates) if prices is not None else None
        segments.append(
            run_backtest_breakdown(weights.loc[test_dates], forward_returns.reindex(test_dates), cost_bps, prices=fold_prices)
        )
    return pd.concat(segments).sort_index()


def main():
    cfg = load_config("config.yaml")
    uni_cfg, data_cfg = cfg["universe"], cfg["data"]
    factor_cfg, port_cfg, cost_cfg = cfg["factors"], cfg["portfolio"], cfg["costs"]
    wf_cfg = cfg.get("validation", {}).get("walk_forward", {})
    start, end, cache_dir = uni_cfg["start_date"], uni_cfg["end_date"], data_cfg["cache_dir"]
    iw, tw, ew = wf_cfg.get("initial_train_months", 60), wf_cfg.get("test_months", 12), wf_cfg.get("embargo_months", 1)

    universe = build_universe(uni_cfg["name"], start, end, cache_dir=cache_dir)
    tickers = sorted(universe.columns[universe.any(axis=0)])
    prices = load_prices(tickers, start, end, cache_dir=cache_dir)
    fundamentals = load_fundamentals(tickers, start, end, lag_days=data_cfg["fundamentals_lag_days"], cache_dir=cache_dir)

    monthly_prices = prices.pivot(index="date", columns="ticker", values="adj_close").resample("ME").last()
    forward_returns = monthly_prices.pct_change().shift(-1).dropna(how="all")

    rows = []

    # --- Baseline: equal-weight linear composite ---
    factor_frames = {
        "momentum": zscore_cross_section(momentum(prices, factor_cfg["momentum"]["lookback_months"], factor_cfg["momentum"]["skip_months"])),
        "value": zscore_cross_section(value(fundamentals, prices, factor_cfg["value"]["metric"])),
        "quality": zscore_cross_section(quality(fundamentals, factor_cfg["quality"]["metric"])),
    }
    composite = combine_factors(factor_frames)
    monthly_membership = universe.reindex(composite.index, method="ffill").reindex(columns=composite.columns, fill_value=False)
    composite = composite.where(monthly_membership)
    baseline_weights = decile_portfolios(composite, n_deciles=port_cfg["n_deciles"], long_short=port_cfg["long_short"])

    baseline_breakdown = walk_forward_breakdown(
        baseline_weights, forward_returns, cost_cfg["bps_per_trade"], start, end, iw, tw, ew, monthly_prices
    )
    rows.append(("baseline", baseline_breakdown, 1))

    # --- ML configs ---
    panel = build_ml_panel(prices, fundamentals, universe, factor_cfg)
    for model_cfg in MODEL_CONFIGS:
        oos_scores = walk_forward_ml_scores(
            panel, start=start, end=end, model=model_cfg["model"], params=model_cfg["params"],
            initial_train_months=iw, test_months=tw, embargo_months=ew,
        )
        ml_weights = decile_portfolios(oos_scores, n_deciles=port_cfg["n_deciles"], long_short=port_cfg["long_short"])
        breakdown = walk_forward_breakdown(
            ml_weights, forward_returns, cost_cfg["bps_per_trade"], start, end, iw, tw, ew, monthly_prices
        )
        rows.append((model_cfg["name"], breakdown, N_TRIALS))
        print(f"done: {model_cfg['name']}")

    # --- Cross-check net_return against the already-committed results ---
    committed_baseline = pd.read_csv("../results/baseline/oos_net_returns.csv", index_col=0, parse_dates=True)["net_return"]
    max_diff = (rows[0][1]["net_return"] - committed_baseline.reindex(rows[0][1].index)).abs().max()
    print(f"\nBaseline net_return max abs diff vs committed results/baseline/oos_net_returns.csv: {max_diff:.2e}")
    assert max_diff < 1e-9, "gross_vs_net.py's fold loop diverged from the committed walk_forward_backtest pipeline"

    out_rows = []
    for name, breakdown, n_trials in rows:
        gross, net = breakdown["gross_return"], breakdown["net_return"]
        out_rows.append({
            "config": name,
            "gross_sharpe": sharpe(gross, PERIODS_PER_YEAR),
            "net_sharpe": sharpe(net, PERIODS_PER_YEAR),
            "gross_deflated_sharpe": deflated_sharpe(gross, n_trials),
            "net_deflated_sharpe": deflated_sharpe(net, n_trials),
            "cost_drag_sharpe": sharpe(gross, PERIODS_PER_YEAR) - sharpe(net, PERIODS_PER_YEAR),
            "n_trials": n_trials,
            "n_periods": len(net),
        })

    table = pd.DataFrame(out_rows).set_index("config")
    out_dir = Path("../results/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "gross_vs_net.csv")

    display = table.copy()
    for col in ["gross_sharpe", "net_sharpe", "gross_deflated_sharpe", "net_deflated_sharpe", "cost_drag_sharpe"]:
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    print("\n" + display.to_string())
    print(f"\nSaved {out_dir / 'gross_vs_net.csv'}")


if __name__ == "__main__":
    main()
