"""Sanity checks for Step 2 (ML dataset): src/ml/dataset.py."""
import pandas as pd
import pytest

from src.ml.dataset import build_ml_panel

MONTH_ENDS = pd.date_range("2020-01-31", "2020-06-30", freq="ME")  # 6 months

FACTOR_CFG = {
    "momentum": {"lookback_months": 1, "skip_months": 0},
    "value": {"metric": "earnings_yield"},
    "quality": {"metric": "roe"},
}


def _long_prices(paths: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for ticker, series in paths.items():
        for dt, px in zip(MONTH_ENDS, series):
            rows.append({"date": dt, "ticker": ticker, "adj_close": px})
    return pd.DataFrame(rows)


def _fundamentals_for(tickers: list[str]) -> pd.DataFrame:
    # One report per ticker, filed well before the window, so value/quality
    # have something to carry forward across the whole panel.
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2019-06-15"),
                "ticker": t,
                "report_date": pd.Timestamp("2019-03-31"),
                "earnings": 5.0,
                "book_value": 50.0,
                "roe": 0.1,
            }
            for t in tickers
        ]
    )


def test_fwd_ret_is_the_t_to_t1_return_never_the_prior_period():
    # AAA: flat, then -9.09% (Feb->Mar), then a +200% jump (Mar->Apr), then flat.
    prices = _long_prices(
        {
            "AAA": [100, 110, 100, 300, 300, 300],
            "BBB": [50, 51, 52, 53, 54, 55],
        }
    )
    fundamentals = _fundamentals_for(["AAA", "BBB"])
    universe = pd.DataFrame(True, index=MONTH_ENDS, columns=["AAA", "BBB"])

    panel = build_ml_panel(prices, fundamentals, universe, FACTOR_CFG)

    row = panel[(panel["ticker"] == "AAA") & (panel["date"] == pd.Timestamp("2020-03-31"))].iloc[0]
    # The Mar->Apr forward return (the jump), not the Feb->Mar return that
    # would have driven AAA's momentum score as of March.
    assert row["fwd_ret"] == pytest.approx((300 - 100) / 100)
    assert row["fwd_ret"] != pytest.approx((100 - 110) / 110)


def test_last_month_has_no_forward_return_and_is_dropped():
    prices = _long_prices({"AAA": [100, 110, 100, 300, 300, 300]})
    fundamentals = _fundamentals_for(["AAA"])
    universe = pd.DataFrame(True, index=MONTH_ENDS, columns=["AAA"])

    panel = build_ml_panel(prices, fundamentals, universe, FACTOR_CFG)

    assert panel["date"].max() < MONTH_ENDS[-1]


def test_rows_restricted_to_point_in_time_universe_membership():
    # CCC has perfectly good prices but was never a member of the universe —
    # it must not appear in the panel at all.
    prices = _long_prices(
        {
            "AAA": [100, 110, 100, 300, 300, 300],
            "CCC": [10, 20, 30, 40, 50, 60],
        }
    )
    fundamentals = _fundamentals_for(["AAA", "CCC"])
    universe = pd.DataFrame(True, index=MONTH_ENDS, columns=["AAA"])  # CCC absent

    panel = build_ml_panel(prices, fundamentals, universe, FACTOR_CFG)

    assert set(panel["ticker"]) == {"AAA"}


def test_output_shape_and_columns():
    prices = _long_prices({"AAA": [100, 110, 100, 300, 300, 300]})
    fundamentals = _fundamentals_for(["AAA"])
    universe = pd.DataFrame(True, index=MONTH_ENDS, columns=["AAA"])

    panel = build_ml_panel(prices, fundamentals, universe, FACTOR_CFG)

    assert list(panel.columns) == ["date", "ticker", "mom_z", "val_z", "qual_z", "fwd_ret"]
    assert panel["fwd_ret"].notna().all()  # target dropna already applied
