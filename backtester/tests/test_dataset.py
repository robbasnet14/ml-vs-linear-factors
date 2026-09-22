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


def test_ticker_joining_or_leaving_mid_window_only_appears_while_a_member():
    # DDD has prices for every month but only joins the index at 2020-03-31;
    # EEE has prices for every month but is removed after 2020-02-29. Neither
    # may show up on a date when it wasn't a member, even though its features
    # and forward return are computable there.
    prices = _long_prices(
        {
            "AAA": [100, 110, 100, 300, 300, 300],
            "DDD": [10, 11, 12, 13, 14, 15],
            "EEE": [20, 21, 22, 23, 24, 25],
        }
    )
    fundamentals = _fundamentals_for(["AAA", "DDD", "EEE"])
    universe = pd.DataFrame(True, index=MONTH_ENDS, columns=["AAA", "DDD", "EEE"])
    universe.loc[MONTH_ENDS < pd.Timestamp("2020-03-31"), "DDD"] = False
    universe.loc[MONTH_ENDS > pd.Timestamp("2020-02-29"), "EEE"] = False

    panel = build_ml_panel(prices, fundamentals, universe, FACTOR_CFG)

    dates_by_ticker = panel.groupby("ticker")["date"].agg(["min", "max"])
    # DDD: first row is its join date, and it's present through the last
    # date with a forward return (May; June has none).
    assert dates_by_ticker.loc["DDD", "min"] == pd.Timestamp("2020-03-31")
    assert dates_by_ticker.loc["DDD", "max"] == pd.Timestamp("2020-05-31")
    assert set(panel.loc[panel["ticker"] == "DDD", "date"]) == set(MONTH_ENDS[2:5])
    # EEE: nothing after its removal date.
    assert dates_by_ticker.loc["EEE", "max"] == pd.Timestamp("2020-02-29")
    assert set(panel.loc[panel["ticker"] == "EEE", "date"]) == set(MONTH_ENDS[:2])
    # AAA (a member throughout) is unaffected by the others' churn.
    assert set(panel.loc[panel["ticker"] == "AAA", "date"]) == set(MONTH_ENDS[:5])


def test_output_shape_and_columns():
    prices = _long_prices({"AAA": [100, 110, 100, 300, 300, 300]})
    fundamentals = _fundamentals_for(["AAA"])
    universe = pd.DataFrame(True, index=MONTH_ENDS, columns=["AAA"])

    panel = build_ml_panel(prices, fundamentals, universe, FACTOR_CFG)

    assert list(panel.columns) == ["date", "ticker", "mom_z", "val_z", "qual_z", "fwd_ret"]
    assert panel["fwd_ret"].notna().all()  # target dropna already applied
