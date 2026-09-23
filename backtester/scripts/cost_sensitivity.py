"""Step 7: cost-sensitivity sweep -- at what cost (bps/trade) does the best
ML config's OOS Sharpe advantage over the baseline disappear?

Cost is linear in bps (apply_costs = turnover * bps / 1e4), so this needs
only ONE walk-forward pass per arm (at a small reference bps, here 1bp) to
recover each period's gross_return and turnover-implied cost-per-bp; the
Sharpe at any other bps is then a closed-form rescaling, not a re-run.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.metrics import sharpe
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
BEST_CONFIG_NAME = "rf_depth10_leaf50"
REFERENCE_BPS = 1.0
SWEEP_BPS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]


def gross_and_cost_per_bp(weights, forward_returns, start, end, iw, tw, ew, prices):
    folds = make_walk_forward_folds(start, end, iw, tw, ew)
    segments = []
    for fold in folds:
        test_dates = weights.index[(weights.index >= fold["test_start"]) & (weights.index < fold["test_end"])]
        if test_dates.empty:
            continue
        fold_prices = prices.reindex(test_dates) if prices is not None else None
        b = run_backtest_breakdown(weights.loc[test_dates], forward_returns.reindex(test_dates), REFERENCE_BPS, prices=fold_prices)
        segments.append(b)
    stitched = pd.concat(segments).sort_index()
    cost_per_bp = stitched["cost"] / REFERENCE_BPS  # apply_costs is linear in bps
    return stitched["gross_return"], cost_per_bp


def sharpe_at(gross, cost_per_bp, bps):
    net = gross - cost_per_bp * bps
    return sharpe(net, PERIODS_PER_YEAR)


def main():
    cfg = load_config("config.yaml")
    uni_cfg, data_cfg = cfg["universe"], cfg["data"]
    factor_cfg, port_cfg = cfg["factors"], cfg["portfolio"]
    wf_cfg = cfg.get("validation", {}).get("walk_forward", {})
    start, end, cache_dir = uni_cfg["start_date"], uni_cfg["end_date"], data_cfg["cache_dir"]
    iw, tw, ew = wf_cfg.get("initial_train_months", 60), wf_cfg.get("test_months", 12), wf_cfg.get("embargo_months", 1)

    universe = build_universe(uni_cfg["name"], start, end, cache_dir=cache_dir)
    tickers = sorted(universe.columns[universe.any(axis=0)])
    prices = load_prices(tickers, start, end, cache_dir=cache_dir)
    fundamentals = load_fundamentals(tickers, start, end, lag_days=data_cfg["fundamentals_lag_days"], cache_dir=cache_dir)
    monthly_prices = prices.pivot(index="date", columns="ticker", values="adj_close").resample("ME").last()
    forward_returns = monthly_prices.pct_change().shift(-1).dropna(how="all")

    # Baseline
    factor_frames = {
        "momentum": zscore_cross_section(momentum(prices, factor_cfg["momentum"]["lookback_months"], factor_cfg["momentum"]["skip_months"])),
        "value": zscore_cross_section(value(fundamentals, prices, factor_cfg["value"]["metric"])),
        "quality": zscore_cross_section(quality(fundamentals, factor_cfg["quality"]["metric"])),
    }
    composite = combine_factors(factor_frames)
    monthly_membership = universe.reindex(composite.index, method="ffill").reindex(columns=composite.columns, fill_value=False)
    composite = composite.where(monthly_membership)
    baseline_weights = decile_portfolios(composite, n_deciles=port_cfg["n_deciles"], long_short=port_cfg["long_short"])
    baseline_gross, baseline_cost_per_bp = gross_and_cost_per_bp(baseline_weights, forward_returns, start, end, iw, tw, ew, monthly_prices)

    # Best ML
    panel = build_ml_panel(prices, fundamentals, universe, factor_cfg)
    model_cfg = next(c for c in MODEL_CONFIGS if c["name"] == BEST_CONFIG_NAME)
    oos_scores = walk_forward_ml_scores(panel, start=start, end=end, model=model_cfg["model"], params=model_cfg["params"],
                                         initial_train_months=iw, test_months=tw, embargo_months=ew)
    ml_weights = decile_portfolios(oos_scores, n_deciles=port_cfg["n_deciles"], long_short=port_cfg["long_short"])
    ml_gross, ml_cost_per_bp = gross_and_cost_per_bp(ml_weights, forward_returns, start, end, iw, tw, ew, monthly_prices)

    rows = []
    for bps in SWEEP_BPS:
        b_sharpe = sharpe_at(baseline_gross, baseline_cost_per_bp, bps)
        m_sharpe = sharpe_at(ml_gross, ml_cost_per_bp, bps)
        rows.append({"bps": bps, "baseline_sharpe": b_sharpe, "ml_sharpe": m_sharpe, "ml_advantage": m_sharpe - b_sharpe})
    table = pd.DataFrame(rows).set_index("bps")

    out_dir = Path("../results/comparison")
    table.to_csv(out_dir / "cost_sensitivity.csv")
    print(table.round(4).to_string())

    # Find the crossover: finest bps (0.1 resolution) where ml_advantage hits 0
    fine_bps = np.arange(0, 200.1, 0.1)
    advantage = np.array([sharpe_at(ml_gross, ml_cost_per_bp, b) - sharpe_at(baseline_gross, baseline_cost_per_bp, b) for b in fine_bps])
    sign_changes = np.where(np.diff(np.sign(advantage)) != 0)[0]
    if len(sign_changes):
        crossover_bps = fine_bps[sign_changes[0]]
        print(f"\nCrossover (ML advantage hits zero) at approximately {crossover_bps:.1f} bps/trade "
              f"(study uses {cfg['costs']['bps_per_trade']} bps)")
    else:
        direction = "always favors ML" if advantage[0] > 0 else "never favors ML"
        print(f"\nNo crossover found in [0, 200] bps -- {direction} across this whole range")

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(fine_bps, [sharpe_at(baseline_gross, baseline_cost_per_bp, b) for b in fine_bps],
            color="#1baf7a", linewidth=2, linestyle="--", label="Linear baseline")
    ax.plot(fine_bps, [sharpe_at(ml_gross, ml_cost_per_bp, b) for b in fine_bps],
            color="#2a78d6", linewidth=2, label=f"ML: {BEST_CONFIG_NAME}")
    ax.axhline(0.0, color="#898781", linewidth=0.8)
    ax.axvline(cfg["costs"]["bps_per_trade"], color="#52514e", linewidth=1, linestyle=":",
               label=f"Study assumption ({cfg['costs']['bps_per_trade']} bps)")
    if len(sign_changes):
        ax.axvline(crossover_bps, color="#c0392b", linewidth=1, linestyle=":",
                   label=f"Crossover ({crossover_bps:.1f} bps)")
    ax.set_xlabel("Cost per trade (bps)", color="#52514e")
    ax.set_ylabel("Annualized OOS net Sharpe", color="#52514e")
    ax.set_title("Cost sensitivity: OOS Sharpe vs. assumed trading cost", color="#0b0b0b",
                 fontsize=13, fontweight="bold", loc="left")
    ax.tick_params(colors="#898781")
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right", labelcolor="#52514e")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_sensitivity.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Saved {out_dir / 'cost_sensitivity.png'}")

    print(f"\nSaved {out_dir / 'cost_sensitivity.csv'}")


if __name__ == "__main__":
    main()
