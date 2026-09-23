"""Step 7: per-fold permutation importance for the best ML config
(rf_depth10_leaf50), scored by cross-sectional rank IC. See
src/ml/importance.py's module docstring for why per-fold + rank IC rather
than a single pooled R^2-based number.
"""
from pathlib import Path

import pandas as pd

from src.data.loader import load_fundamentals, load_prices
from src.data.universe import build_universe
from src.ml.dataset import build_ml_panel
from src.ml.importance import permutation_importance_per_fold
from src.ml.model import MODEL_CONFIGS
from src.utils.config import load_config

BEST_CONFIG_NAME = "rf_depth10_leaf50"
MIN_TEST_DATES_FOR_SUMMARY = 2  # exclude a truncated trailing fold from aggregate stats


def main():
    cfg = load_config("config.yaml")
    uni_cfg, data_cfg, factor_cfg = cfg["universe"], cfg["data"], cfg["factors"]
    wf_cfg = cfg.get("validation", {}).get("walk_forward", {})
    start, end, cache_dir = uni_cfg["start_date"], uni_cfg["end_date"], data_cfg["cache_dir"]

    universe = build_universe(uni_cfg["name"], start, end, cache_dir=cache_dir)
    tickers = sorted(universe.columns[universe.any(axis=0)])
    prices = load_prices(tickers, start, end, cache_dir=cache_dir)
    fundamentals = load_fundamentals(tickers, start, end, lag_days=data_cfg["fundamentals_lag_days"], cache_dir=cache_dir)
    panel = build_ml_panel(prices, fundamentals, universe, factor_cfg)

    model_cfg = next(c for c in MODEL_CONFIGS if c["name"] == BEST_CONFIG_NAME)
    result = permutation_importance_per_fold(
        panel, start=start, end=end, model=model_cfg["model"], params=model_cfg["params"],
        initial_train_months=wf_cfg.get("initial_train_months", 60),
        test_months=wf_cfg.get("test_months", 12),
        embargo_months=wf_cfg.get("embargo_months", 1),
    )

    out_dir = Path("../results/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "feature_importance_per_fold.csv", index=False)

    print(f"Per-fold detail ({BEST_CONFIG_NAME}), ALL folds including any truncated trailing one:")
    pd.set_option("display.width", 160)
    print(result[["fold", "test_start", "test_end", "n_test_dates", "feature", "baseline_ic", "importance"]].to_string(index=False))

    truncated = sorted(result.loc[result["n_test_dates"] < MIN_TEST_DATES_FOR_SUMMARY, "fold"].unique())
    full = result[result["n_test_dates"] >= MIN_TEST_DATES_FOR_SUMMARY]
    if truncated:
        print(f"\nExcluding fold(s) {truncated} from summary stats below (< {MIN_TEST_DATES_FOR_SUMMARY} OOS "
              f"dates -- a single-date fold's rank IC is one Spearman correlation, not an average, and would "
              f"be over-weighted in a mean/std across folds).")

    summary = full.groupby("feature")["importance"].agg(["mean", "std", "min", "max"])
    summary = summary.sort_values("mean", ascending=False)
    print(f"\nAcross-fold summary on {full['fold'].nunique()} full folds (mean +/- std of importance, higher = more informative):")
    print(summary.to_string())

    print("\nRank stability check -- which feature had the HIGHEST importance in each full fold:")
    winners = full.loc[full.groupby("fold")["importance"].idxmax(), ["fold", "feature", "importance"]]
    print(winners.to_string(index=False))
    print(f"\nWinner counts: {winners['feature'].value_counts().to_dict()}")

    n_negative_baseline_ic = full.drop_duplicates("fold")["baseline_ic"].lt(0).sum()
    print(f"\nFolds with negative baseline (unpermuted) rank IC: {n_negative_baseline_ic} of {full['fold'].nunique()}")

    print(f"\nSaved {out_dir / 'feature_importance_per_fold.csv'}")


if __name__ == "__main__":
    main()
