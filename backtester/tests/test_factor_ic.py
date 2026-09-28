"""Sanity checks for src/analytics/factor_ic.py."""
import numpy as np
import pandas as pd
import pytest

from src.analytics.factor_ic import monthly_rank_ic, summarize_ic

MIN_NAMES = 5  # small, so tiny synthetic months are usable without hitting the real 20-name floor


def _month(date, factor_vals, target_vals):
    n = len(factor_vals)
    return pd.DataFrame({
        "date": [date] * n,
        "factor": factor_vals,
        "fwd_ret": target_vals,
    })


def test_perfectly_rank_ordered_month_gives_ic_one():
    panel = _month("2020-01-31", [1, 2, 3, 4, 5], [0.01, 0.02, 0.03, 0.04, 0.05])
    ic = monthly_rank_ic(panel, "factor", min_names=MIN_NAMES)
    assert ic["2020-01-31"] == pytest.approx(1.0)


def test_perfectly_reversed_month_gives_ic_negative_one():
    panel = _month("2020-01-31", [1, 2, 3, 4, 5], [0.05, 0.04, 0.03, 0.02, 0.01])
    ic = monthly_rank_ic(panel, "factor", min_names=MIN_NAMES)
    assert ic["2020-01-31"] == pytest.approx(-1.0)


def test_month_below_min_names_is_skipped():
    panel = _month("2020-01-31", [1, 2, 3], [0.01, 0.02, 0.03])  # only 3 names, min_names=5
    ic = monthly_rank_ic(panel, "factor", min_names=MIN_NAMES)
    assert ic.empty


def test_nan_rows_dropped_pairwise_not_propagated_to_the_whole_month():
    # One ticker has a NaN factor value; the other 5 are perfectly rank-ordered.
    # A naive spearmanr on the raw columns would return NaN for the whole month;
    # dropping the NaN pair first should still recover IC = 1.0 on the rest.
    panel = _month("2020-01-31", [1, 2, 3, 4, 5, np.nan], [0.01, 0.02, 0.03, 0.04, 0.05, 0.99])
    ic = monthly_rank_ic(panel, "factor", min_names=MIN_NAMES)
    assert ic["2020-01-31"] == pytest.approx(1.0)


def test_monthly_rank_ic_computed_independently_per_date():
    panel = pd.concat([
        _month("2020-01-31", [1, 2, 3, 4, 5], [0.01, 0.02, 0.03, 0.04, 0.05]),   # IC = 1.0
        _month("2020-02-29", [1, 2, 3, 4, 5], [0.05, 0.04, 0.03, 0.02, 0.01]),   # IC = -1.0
    ])
    ic = monthly_rank_ic(panel, "factor", min_names=MIN_NAMES)
    assert list(ic.index) == ["2020-01-31", "2020-02-29"]
    assert ic["2020-01-31"] == pytest.approx(1.0)
    assert ic["2020-02-29"] == pytest.approx(-1.0)


def test_summarize_ic_matches_hand_computed_stats():
    ic = pd.Series([1.0, -1.0, 0.5, 0.5])  # mean 0.25, sample std ddof=1
    summary = summarize_ic(ic)
    expected_mean = 0.25
    expected_std = float(np.std([1.0, -1.0, 0.5, 0.5], ddof=1))
    expected_t = expected_mean / (expected_std / np.sqrt(4))

    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(expected_mean)
    assert summary["std"] == pytest.approx(expected_std)
    assert summary["t_stat"] == pytest.approx(expected_t)


def test_summarize_ic_handles_empty_series():
    summary = summarize_ic(pd.Series([], dtype=float))
    assert summary["n"] == 0
    assert np.isnan(summary["mean"])
    assert np.isnan(summary["std"])
    assert np.isnan(summary["t_stat"])
