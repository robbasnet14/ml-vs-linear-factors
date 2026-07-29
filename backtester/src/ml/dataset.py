"""Build a supervised-learning panel from the backtester's existing factor pipeline.

`build_ml_panel` reuses the exact same loaders, factor formulas, and
cross-sectional standardization as the linear (equal-weight) path in
scripts/run_backtest.py, so the ML challenger sees exactly the same
information the baseline composite does. The only thing that's different
between the two paths is how mom_z/val_z/qual_z get combined into a
signal — a weighted average for the baseline, a fitted model here.
"""
import pandas as pd

from src.features.factors import momentum, quality, value
from src.features.transforms import zscore_cross_section


def build_ml_panel(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    universe: pd.DataFrame,
    factor_cfg: dict,
) -> pd.DataFrame:
    """Return a long (date, ticker, mom_z, val_z, qual_z, fwd_ret) panel.

    `prices`, `fundamentals`, and `universe` are exactly what
    `load_prices`/`load_fundamentals`/`build_universe` return (see
    scripts/run_backtest.py's wiring); `factor_cfg` is the config.yaml
    `factors` section. Each of mom_z/val_z/qual_z is the SAME
    cross-sectionally standardized factor the linear composite uses
    (`src.features.transforms.zscore_cross_section` over the full ticker
    set, same lookback/skip/metric settings) — nothing here recomputes or
    approximates a factor differently for ML.

    `fwd_ret` is the return earned from rebalance date t to t+1
    (`monthly_price.pct_change().shift(-1)`), identical to what
    `src.backtest.engine.run_backtest` pairs with weights — never the
    contemporaneous (t-1 -> t) return the factors at t were computed from.

    A row exists only where the ticker was a point-in-time member of
    `universe` on that date AND has a non-NaN forward return (the same
    two conditions the linear path enforces via its own membership mask and
    its `forward_returns.dropna(how="all")` on the last, return-less date).
    An individual feature (mom_z/val_z/qual_z) can still be NaN in a kept
    row — dropping or imputing that is a training-time decision, not a
    dataset-building one.
    """
    mom_cfg, val_cfg, qual_cfg = factor_cfg["momentum"], factor_cfg["value"], factor_cfg["quality"]

    mom_z = zscore_cross_section(momentum(prices, mom_cfg["lookback_months"], mom_cfg["skip_months"]))
    val_z = zscore_cross_section(value(fundamentals, prices, val_cfg["metric"]))
    qual_z = zscore_cross_section(quality(fundamentals, qual_cfg["metric"]))

    monthly_prices = prices.pivot(index="date", columns="ticker", values="adj_close").resample("ME").last()
    fwd_ret = monthly_prices.pct_change().shift(-1)

    frames = {"mom_z": mom_z, "val_z": val_z, "qual_z": qual_z, "fwd_ret": fwd_ret}
    all_index = sorted(set().union(*(f.index for f in frames.values())))
    all_columns = sorted(set().union(*(f.columns for f in frames.values())))
    frames = {name: f.reindex(index=all_index, columns=all_columns) for name, f in frames.items()}

    # Same restriction the linear composite applies: a ticker only counts on
    # dates it was actually a member of the index (no survivorship bias).
    monthly_membership = universe.reindex(all_index, method="ffill").reindex(columns=all_columns, fill_value=False)
    member_mask = monthly_membership.stack(future_stack=True)
    member_index = member_mask[member_mask].index.set_names(["date", "ticker"])

    panel = pd.DataFrame(index=member_index)
    for name, frame in frames.items():
        panel[name] = frame.stack(future_stack=True).reindex(member_index)

    panel = panel.reset_index().dropna(subset=["fwd_ret"])
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
