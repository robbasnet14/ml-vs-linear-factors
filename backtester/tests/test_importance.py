"""Sanity checks for src/ml/importance.py."""
import numpy as np
import pandas as pd
import pytest

from src.ml.importance import permutation_importance_per_fold, rank_ic


def test_rank_ic_perfect_rank_agreement_within_each_date():
    dates = pd.Series(["2020-01-31"] * 4 + ["2020-02-29"] * 4)
    scores = np.array([1, 2, 3, 4, 10, 20, 30, 40])
    targets = pd.Series([0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.3, 0.4])

    assert rank_ic(dates, scores, targets) == pytest.approx(1.0)


def test_rank_ic_skips_degenerate_dates():
    # Second date has constant target (no variance) -- must be skipped, not scored as 0.
    dates = pd.Series(["2020-01-31"] * 3 + ["2020-02-29"] * 3)
    scores = np.array([1, 2, 3, 1, 2, 3])
    targets = pd.Series([0.01, 0.02, 0.03, 0.05, 0.05, 0.05])

    assert rank_ic(dates, scores, targets) == pytest.approx(1.0)  # only the first date counts


def _synthetic_panel(n_dates=24, n_tickers=30, seed=0) -> pd.DataFrame:
    # mom_z is genuinely informative (fwd_ret is a noisy function of it);
    # val_z and qual_z are pure noise, uncorrelated with fwd_ret.
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_dates, freq="ME")
    rows = []
    for d in dates:
        mom_z = rng.normal(size=n_tickers)
        val_z = rng.normal(size=n_tickers)
        qual_z = rng.normal(size=n_tickers)
        fwd_ret = 0.05 * mom_z + rng.normal(scale=0.01, size=n_tickers)
        for i in range(n_tickers):
            rows.append({
                "date": d, "ticker": f"T{i}",
                "mom_z": mom_z[i], "val_z": val_z[i], "qual_z": qual_z[i],
                "fwd_ret": fwd_ret[i],
            })
    return pd.DataFrame(rows)


def test_permutation_importance_identifies_the_informative_feature():
    panel = _synthetic_panel()
    result = permutation_importance_per_fold(
        panel, start="2010-01-01", end="2012-01-31",
        model="random_forest", params={"max_depth": 5, "min_samples_leaf": 5},
        initial_train_months=12, test_months=6, embargo_months=1,
        n_repeats=10, random_state=0,
    )

    assert not result.empty
    assert set(result["feature"]) == {"mom_z", "val_z", "qual_z"}

    mean_importance = result.groupby("feature")["importance"].mean()
    # The genuinely informative feature must clearly outrank both noise features.
    assert mean_importance["mom_z"] > mean_importance["val_z"]
    assert mean_importance["mom_z"] > mean_importance["qual_z"]


def test_permutation_importance_one_row_per_fold_and_feature():
    panel = _synthetic_panel(n_dates=30)
    result = permutation_importance_per_fold(
        panel, start="2010-01-01", end="2012-07-31",
        model="random_forest", params={"max_depth": 5, "min_samples_leaf": 5},
        initial_train_months=12, test_months=6, embargo_months=1,
        n_repeats=5, random_state=0,
    )

    n_folds = result["fold"].nunique()
    assert len(result) == n_folds * 3  # 3 features per fold
    assert result.groupby("fold")["test_start"].nunique().eq(1).all()  # one boundary per fold


def test_permutation_importance_flags_a_truncated_trailing_fold():
    # 20 monthly dates, 12-month train + 1-month embargo + nominal 12-month
    # test window -- but the panel only extends 6 months past the embargo,
    # so the one fold that fits is truncated to 6 OOS dates, not 12.
    panel = _synthetic_panel(n_dates=20)
    result = permutation_importance_per_fold(
        panel, start="2010-01-01", end="2011-08-31",
        model="random_forest", params={"max_depth": 5, "min_samples_leaf": 5},
        initial_train_months=12, test_months=12, embargo_months=1,
        n_repeats=5, random_state=0,
    )

    assert "n_test_dates" in result.columns
    last_fold = result["fold"].max()
    assert result.loc[result["fold"] == last_fold, "n_test_dates"].iloc[0] == 6
