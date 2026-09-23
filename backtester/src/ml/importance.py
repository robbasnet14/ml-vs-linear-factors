"""Per-fold permutation importance, scored by cross-sectional rank IC.

Why not sklearn's default `permutation_importance` scoring: this strategy
only ever cares about the within-month RANKING of scores (it goes long the
top decile and short the bottom), not the raw magnitude of a predicted
return. Scoring by R^2 (or any pointwise metric) on pooled forward returns
is dominated by the common time-series component shared across all
tickers in a month and would barely move when a feature that matters for
cross-sectional ranking gets shuffled. The rank IC — the average, across
each fold's OOS months, of the Spearman correlation between a config's
predicted score and realized forward return within that month's
cross-section — is what the backtest actually depends on, so it's what
gets used as the scoring function here.

Why per-fold, not one pooled number: pooling collapses 9 independent
walk-forward folds' importances into a single sample-size-1 estimate of
"the" feature ranking. Reporting mean +/- std across folds instead makes
fold-to-fold instability visible — a wide std for the leading feature is
itself a finding (the model isn't leaning on a stable relationship), not
noise to average away.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer

from src.ml.model import FEATURE_COLUMNS, TARGET_COLUMN, build_model, split_fold
from src.backtest.validation import make_walk_forward_folds

_N_REPEATS = 20
_RANDOM_STATE = 0


def rank_ic(dates: pd.Series, scores: np.ndarray, targets: pd.Series) -> float:
    """Mean, across each date's cross-section, of the Spearman correlation
    between `scores` and `targets` — the metric the decile long/short
    engine actually depends on. A date with fewer than 3 names, or with no
    variance in either `scores` or `targets`, contributes no IC (skipped
    rather than treated as 0, so a fold with a few degenerate months isn't
    dragged toward 0 by construction).
    """
    frame = pd.DataFrame({"date": dates.to_numpy(), "score": scores, "target": targets.to_numpy()})
    ics = []
    for _, group in frame.groupby("date"):
        if len(group) < 3 or group["score"].nunique() < 2 or group["target"].nunique() < 2:
            continue
        ic, _ = spearmanr(group["score"], group["target"])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def permutation_importance_per_fold(
    panel: pd.DataFrame,
    start: str,
    end: str,
    model: str,
    params: dict,
    initial_train_months: int = 60,
    test_months: int = 12,
    embargo_months: int = 1,
    n_repeats: int = _N_REPEATS,
    random_state: int = _RANDOM_STATE,
) -> pd.DataFrame:
    """Fit `model`/`params` on each walk-forward fold's train rows (same
    folds `walk_forward_ml_scores` uses), then for each feature, shuffle
    that column alone in the fold's test rows `n_repeats` times and measure
    the drop in rank IC versus the fold's actual (unpermuted) predictions.

    Returns a long DataFrame [fold, test_start, test_end, n_test_dates,
    feature, baseline_ic, mean_permuted_ic, importance] (importance =
    baseline_ic - mean permuted_ic over the repeats) — one row per (fold,
    feature), ready to group by feature for a mean +/- std across folds.
    `n_test_dates` is included so a caller can exclude a truncated trailing
    fold (e.g. the final fold ending at `end` with only 1 OOS month) from
    aggregate stats — a single-date fold's rank IC is a single Spearman
    correlation, not an average, and including it in a mean/std across
    folds gives it the same weight as a fold built from 12 independent
    monthly cross-sections.
    """
    folds = make_walk_forward_folds(start, end, initial_train_months, test_months, embargo_months)
    rng = np.random.default_rng(random_state)

    rows = []
    for fold_idx, fold in enumerate(folds):
        train, test = split_fold(panel, fold)
        if train.empty or test.empty:
            continue

        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train[FEATURE_COLUMNS])
        y_train = train[TARGET_COLUMN].to_numpy()

        estimator = build_model(model, params)
        estimator.fit(x_train, y_train)

        x_test = imputer.transform(test[FEATURE_COLUMNS])
        baseline_pred = estimator.predict(x_test)
        baseline_ic = rank_ic(test["date"], baseline_pred, test[TARGET_COLUMN])
        n_test_dates = test["date"].nunique()

        for j, feature in enumerate(FEATURE_COLUMNS):
            permuted_ics = []
            for _ in range(n_repeats):
                x_perm = x_test.copy()
                x_perm[:, j] = rng.permutation(x_perm[:, j])
                permuted_pred = estimator.predict(x_perm)
                permuted_ics.append(rank_ic(test["date"], permuted_pred, test[TARGET_COLUMN]))
            mean_permuted_ic = float(np.nanmean(permuted_ics))
            rows.append({
                "fold": fold_idx,
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "n_test_dates": n_test_dates,
                "feature": feature,
                "baseline_ic": baseline_ic,
                "mean_permuted_ic": mean_permuted_ic,
                "importance": baseline_ic - mean_permuted_ic,
            })

    return pd.DataFrame(rows)
