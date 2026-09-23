"""Step 7: per-fold (not hand-picked pre/post-split) Sharpe for baseline vs
best ML, using the exact walk-forward fold boundaries the pre-registration
already fixed (src.backtest.validation.make_walk_forward_folds) -- no
discretionary split-point choice, so nothing here invites a "you picked the
cut" question.
"""
from pathlib import Path

import pandas as pd

from src.analytics.metrics import sharpe
from src.backtest.validation import make_walk_forward_folds
from src.utils.config import load_config

PERIODS_PER_YEAR = 12


def load_returns(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)["net_return"]


def main():
    cfg = load_config("config.yaml")
    uni_cfg = cfg["universe"]
    wf_cfg = cfg.get("validation", {}).get("walk_forward", {})
    start, end = uni_cfg["start_date"], uni_cfg["end_date"]
    iw, tw, ew = wf_cfg.get("initial_train_months", 60), wf_cfg.get("test_months", 12), wf_cfg.get("embargo_months", 1)

    folds = make_walk_forward_folds(start, end, iw, tw, ew)

    baseline = load_returns("../results/baseline/oos_net_returns.csv")
    best_ml = load_returns("../results/ml/ml_oos_net_returns_rf_depth10_leaf50.csv")

    rows = []
    for i, fold in enumerate(folds):
        b = baseline[(baseline.index >= fold["test_start"]) & (baseline.index < fold["test_end"])]
        m = best_ml[(best_ml.index >= fold["test_start"]) & (best_ml.index < fold["test_end"])]
        rows.append({
            "fold": i,
            "test_start": fold["test_start"].date(),
            "test_end": fold["test_end"].date(),
            "n_periods": len(b),
            "baseline_sharpe": sharpe(b, PERIODS_PER_YEAR),
            "baseline_cum_return_pct": ((1 + b.dropna()).prod() - 1) * 100,
            "ml_sharpe": sharpe(m, PERIODS_PER_YEAR),
            "ml_cum_return_pct": ((1 + m.dropna()).prod() - 1) * 100,
        })

    table = pd.DataFrame(rows).set_index("fold")
    out_path = Path("../results/comparison/per_fold_subperiods.csv")
    table.to_csv(out_path)

    display = table.copy()
    for col in ["baseline_sharpe", "ml_sharpe"]:
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    for col in ["baseline_cum_return_pct", "ml_cum_return_pct"]:
        display[col] = display[col].map(lambda v: f"{v:+.2f}%")
    print(display.to_string())
    print(f"\nSaved {out_path}")

    n_folds_ml_beats_baseline = (table["ml_sharpe"] > table["baseline_sharpe"]).sum()
    print(f"\nML Sharpe beats baseline Sharpe in {n_folds_ml_beats_baseline}/{len(table)} folds")

    # --- Leave-one-out: drop each fold's periods from the POOLED series and
    # recompute Sharpe on what's left, for both arms. A simple per-fold win
    # count treats every fold's win/loss equally regardless of size; this
    # shows whether the pooled 0.241-vs-0.066 result depends on any single
    # fold rather than being a genuinely broad-based edge.
    loo_rows = []
    for i, fold in enumerate(folds):
        mask_b = ~((baseline.index >= fold["test_start"]) & (baseline.index < fold["test_end"]))
        mask_m = ~((best_ml.index >= fold["test_start"]) & (best_ml.index < fold["test_end"]))
        b_sharpe = sharpe(baseline[mask_b], PERIODS_PER_YEAR)
        m_sharpe = sharpe(best_ml[mask_m], PERIODS_PER_YEAR)
        loo_rows.append({"fold_dropped": i, "baseline_sharpe": b_sharpe, "ml_sharpe": m_sharpe, "ml_minus_baseline": m_sharpe - b_sharpe})

    full_b_sharpe, full_m_sharpe = sharpe(baseline, PERIODS_PER_YEAR), sharpe(best_ml, PERIODS_PER_YEAR)
    loo_rows.append({"fold_dropped": "none (full sample)", "baseline_sharpe": full_b_sharpe, "ml_sharpe": full_m_sharpe,
                      "ml_minus_baseline": full_m_sharpe - full_b_sharpe})
    loo_table = pd.DataFrame(loo_rows).set_index("fold_dropped")
    loo_out_path = Path("../results/comparison/leave_one_out_sharpe.csv")
    loo_table.to_csv(loo_out_path)

    loo_display = loo_table.copy()
    for col in loo_display.columns:
        loo_display[col] = loo_display[col].map(lambda v: f"{v:+.3f}")
    print("\nLeave-one-out: drop each fold's periods from the pooled series, recompute Sharpe on the rest")
    print(loo_display.to_string())

    diffs_only_folds = loo_table.drop(index="none (full sample)")["ml_minus_baseline"]
    flips = diffs_only_folds[diffs_only_folds < 0]
    if len(flips):
        worst_fold = diffs_only_folds.idxmin()
        print(f"\nFold(s) whose removal flips ML's advantage negative: {list(flips.index)} "
              f"(most extreme: dropping fold {worst_fold} -> ML-baseline = {diffs_only_folds[worst_fold]:+.3f})")
    else:
        print("\nNo single fold's removal flips ML's advantage negative.")
    print(f"Saved {loo_out_path}")


if __name__ == "__main__":
    main()
