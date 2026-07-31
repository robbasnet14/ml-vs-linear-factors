"""Walk-forward ML challenger: the same folds and embargo as the linear
composite (`src.backtest.validation`), a fitted model instead of a
fixed-weight average.

Only `MODEL_CONFIGS` below gets tried — it's the exact grid committed in
PREREGISTRATION.md (2 learning rates x 2 max depths for gradient boosting,
2 max depths x 2 min-samples-leaf for random forest = 8), so the deflated
Sharpe's n_trials=9 (these 8 plus the 1 linear baseline) stays honest.
`n_estimators`/`random_state` etc. are fixed implementation defaults
applied identically to every configuration, not part of that committed
grid.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer

from src.backtest.validation import make_walk_forward_folds

FEATURE_COLUMNS = ["mom_z", "val_z", "qual_z"]
TARGET_COLUMN = "fwd_ret"

_RANDOM_STATE = 0
_RF_N_ESTIMATORS = 300

MODEL_CONFIGS = [
    {"name": "gbm_lr0.03_depth3", "model": "hist_gb", "params": {"learning_rate": 0.03, "max_depth": 3}},
    {"name": "gbm_lr0.03_depth5", "model": "hist_gb", "params": {"learning_rate": 0.03, "max_depth": 5}},
    {"name": "gbm_lr0.1_depth3", "model": "hist_gb", "params": {"learning_rate": 0.1, "max_depth": 3}},
    {"name": "gbm_lr0.1_depth5", "model": "hist_gb", "params": {"learning_rate": 0.1, "max_depth": 5}},
    {"name": "rf_depth5_leaf50", "model": "random_forest", "params": {"max_depth": 5, "min_samples_leaf": 50}},
    {"name": "rf_depth5_leaf200", "model": "random_forest", "params": {"max_depth": 5, "min_samples_leaf": 200}},
    {"name": "rf_depth10_leaf50", "model": "random_forest", "params": {"max_depth": 10, "min_samples_leaf": 50}},
    {"name": "rf_depth10_leaf200", "model": "random_forest", "params": {"max_depth": 10, "min_samples_leaf": 200}},
]


def build_model(model: str, params: dict):
    if model == "hist_gb":
        return HistGradientBoostingRegressor(random_state=_RANDOM_STATE, **params)
    if model == "random_forest":
        return RandomForestRegressor(random_state=_RANDOM_STATE, n_estimators=_RF_N_ESTIMATORS, n_jobs=-1, **params)
    raise ValueError(f"build_model: unknown model {model!r} (expected 'hist_gb' or 'random_forest')")


def split_fold(panel: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `panel` (from `src.ml.dataset.build_ml_panel`) into a fold's
    train/test rows by date, mirroring `walk_forward_backtest`'s own fold
    semantics exactly: train is `[train_start, train_end)`, test is
    `[test_start, test_end)`, and the embargo gap between them belongs to
    neither — no row from it is trained on or scored.
    """
    train = panel[(panel["date"] >= fold["train_start"]) & (panel["date"] < fold["train_end"])]
    test = panel[(panel["date"] >= fold["test_start"]) & (panel["date"] < fold["test_end"])]
    return train, test


def fit_predict_fold(train: pd.DataFrame, test: pd.DataFrame, model: str, params: dict) -> pd.Series:
    """Fit imputation + the model on `train` ONLY, then predict on `test`.

    The median used to fill a missing mom_z/val_z/qual_z comes entirely
    from `train`'s own distribution — `test`'s feature values (and its
    fwd_ret, which is never passed to the model at all) have no way to
    influence what gets learned. Returns predictions indexed exactly like
    `test`'s original index, so the caller can place them back into a
    (date, ticker) panel.
    """
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURE_COLUMNS])
    y_train = train[TARGET_COLUMN].to_numpy()

    estimator = build_model(model, params)
    estimator.fit(x_train, y_train)

    x_test = imputer.transform(test[FEATURE_COLUMNS])
    return pd.Series(estimator.predict(x_test), index=test.index)


def walk_forward_ml_scores(
    panel: pd.DataFrame,
    start: str,
    end: str,
    model: str,
    params: dict,
    initial_train_months: int = 60,
    test_months: int = 12,
    embargo_months: int = 1,
) -> pd.DataFrame:
    """Stitch strictly out-of-sample predicted forward returns across every
    walk-forward fold into a [date x ticker] panel — the ML analogue of the
    linear path's composite score, ready to feed straight into
    `src.backtest.portfolio.decile_portfolios`.

    Fold boundaries come from the exact same `make_walk_forward_folds` the
    linear path's `walk_forward_backtest` uses, so both challengers are
    scored on identical out-of-sample dates. A fold with no train or no test
    rows in `panel` (e.g. too early for any factor history) is skipped.
    """
    folds = make_walk_forward_folds(start, end, initial_train_months, test_months, embargo_months)
    if not folds:
        raise ValueError("walk_forward_ml_scores: no folds fit in the given range")

    predictions = []
    for fold in folds:
        train, test = split_fold(panel, fold)
        if train.empty or test.empty:
            continue
        preds = fit_predict_fold(train, test, model, params)
        predictions.append(test.loc[preds.index, ["date", "ticker"]].assign(score=preds.to_numpy()))

    if not predictions:
        raise ValueError("walk_forward_ml_scores: folds were computed but none contained usable train+test rows")

    stitched = pd.concat(predictions, ignore_index=True)
    if stitched.duplicated(subset=["date", "ticker"]).any():
        raise AssertionError("walk_forward_ml_scores: fold test windows overlapped — this should be impossible")

    return stitched.pivot(index="date", columns="ticker", values="score")
