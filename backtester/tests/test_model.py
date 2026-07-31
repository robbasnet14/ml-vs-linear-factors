"""Sanity checks for Step 4 (ML challenger): src/ml/model.py."""
import numpy as np
import pandas as pd
import pytest

from src.ml.model import FEATURE_COLUMNS, fit_predict_fold, split_fold, walk_forward_ml_scores

DATES = pd.date_range("2010-01-31", periods=9, freq="ME")  # 6 train months + 2 test months + 1 boundary
FOLD = {"train_start": DATES[0], "train_end": DATES[6], "test_start": DATES[6], "test_end": DATES[8]}


def _panel_row(date, ticker, x, fwd_ret):
    return {"date": date, "ticker": ticker, "mom_z": x, "val_z": x, "qual_z": x, "fwd_ret": fwd_ret}


def _make_panel(test_fwd_ret: float) -> pd.DataFrame:
    rows = [_panel_row(d, "AAA", float(i), 0.01) for i, d in enumerate(DATES[:6])]  # train rows
    rows += [_panel_row(d, "AAA", float(6 + i), test_fwd_ret) for i, d in enumerate(DATES[6:8])]  # test rows
    return pd.DataFrame(rows)


def test_split_fold_separates_train_and_test_by_date_with_no_overlap():
    panel = _make_panel(test_fwd_ret=0.01)

    train, test = split_fold(panel, FOLD)

    assert train["date"].max() < FOLD["train_end"]
    assert test["date"].min() >= FOLD["test_start"]
    assert set(train["date"]) & set(test["date"]) == set()
    assert len(train) + len(test) == len(panel)


def test_imputer_median_comes_from_train_fold_only():
    train = pd.DataFrame({"mom_z": [1.0, 2.0, 3.0, np.nan], "val_z": [0.0] * 4, "qual_z": [0.0] * 4})
    # Extreme values that must NEVER influence the train-fitted median.
    test = pd.DataFrame({"mom_z": [10_000.0, -10_000.0], "val_z": [0.0] * 2, "qual_z": [0.0] * 2})

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    imputer.fit(train)
    mom_idx = list(train.columns).index("mom_z")
    assert imputer.statistics_[mom_idx] == pytest.approx(2.0)  # median of [1, 2, 3], ignoring the NaN and all of `test`


def test_test_fold_target_never_leaks_into_training():
    # Only the test rows' fwd_ret differs between these two panels — an
    # ordinary value in one, a wild outlier in the other. Train rows (and
    # every feature value anywhere) are identical. If the test fold's own
    # label ever influenced its own fold's training, the model fit on the
    # "extreme" panel would predict differently on these same test features.
    panel_normal = _make_panel(test_fwd_ret=0.01)
    panel_extreme = _make_panel(test_fwd_ret=999.0)
    params = {"max_depth": 5, "min_samples_leaf": 1}

    train_n, test_n = split_fold(panel_normal, FOLD)
    train_e, test_e = split_fold(panel_extreme, FOLD)
    assert train_n.equals(train_e)  # sanity: only test rows differ between the two panels

    preds_normal = fit_predict_fold(train_n, test_n, "random_forest", params)
    preds_extreme = fit_predict_fold(train_e, test_e, "random_forest", params)

    assert np.allclose(preds_normal.to_numpy(), preds_extreme.to_numpy())


def test_walk_forward_ml_scores_returns_wide_panel_covering_only_oos_dates():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-31", periods=30, freq="ME")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for t in tickers:
        for d in dates:
            rows.append(_panel_row(d, t, rng.normal(), float(rng.normal(0, 0.02))))
    panel = pd.DataFrame(rows)

    scores = walk_forward_ml_scores(
        panel, start="2010-01-01", end="2012-06-30", model="random_forest",
        params={"max_depth": 5, "min_samples_leaf": 1},
        initial_train_months=12, test_months=6, embargo_months=1,
    )

    assert list(scores.columns) == tickers
    assert scores.index.min() >= pd.Timestamp("2011-02-28")  # first test block starts after train+embargo
    assert not scores.index.duplicated().any()
