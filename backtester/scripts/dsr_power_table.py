"""Minimum-detectable-Sharpe power table for the deflated Sharpe ratio at
this study's n_trials=9, given the actual walk-forward OOS sample size and
the actual skew/kurtosis of each arm's own return series — not a generic
normal-distribution assumption. Answers: "how large would a genuine OOS
Sharpe have had to be for this study's design to actually detect it?"

Reimplements src.analytics.metrics.deflated_sharpe's formula as a function
of (sr, skew, kurt, T, n_trials) and inverts it numerically (bisection) for
a target DSR — deflated_sharpe itself only goes forward (series -> DSR), so
this is the same math run in reverse, not a different formula.
"""
import numpy as np
import pandas as pd
from scipy import stats, optimize

PERIODS_PER_YEAR = 12
_EULER_MASCHERONI = 0.5772156649015329


def dsr_from_sr(sr: float, skew: float, kurt: float, t: int, n_trials: int) -> float:
    variance_term = max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12)
    sigma_sr = np.sqrt(variance_term / (t - 1))
    if n_trials > 1:
        sr_benchmark = sigma_sr * (
            (1 - _EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n_trials)
            + _EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        sr_benchmark = 0.0
    z = (sr - sr_benchmark) / sigma_sr
    return stats.norm.cdf(z)


def min_sr_for_target_dsr(target_dsr: float, skew: float, kurt: float, t: int, n_trials: int) -> float:
    f = lambda sr: dsr_from_sr(sr, skew, kurt, t, n_trials) - target_dsr
    return optimize.brentq(f, -5.0, 5.0)


def load_returns(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)["net_return"].dropna()


if __name__ == "__main__":
    baseline = load_returns("../results/baseline/oos_net_returns.csv")
    best_ml = load_returns("../results/ml/ml_oos_net_returns_rf_depth10_leaf50.csv")

    print("Actual sample stats (per-period, non-annualized net returns):")
    for name, r in [("baseline", baseline), ("rf_depth10_leaf50", best_ml)]:
        print(f"  {name:<20} T={len(r):<4} skew={stats.skew(r):+.3f}  kurt={stats.kurtosis(r, fisher=False):.3f}  "
              f"sr={r.mean()/r.std():.4f} (ann. {r.mean()/r.std()*np.sqrt(PERIODS_PER_YEAR):.3f})")

    print("\nPower table at n_trials=9 -- using rf_depth10_leaf50's actual skew/kurtosis:")
    skew, kurt = stats.skew(best_ml), stats.kurtosis(best_ml, fisher=False)
    rows = []
    for t in [80, 100, 109, 118, 130, 150]:
        sr50 = min_sr_for_target_dsr(0.50, skew, kurt, t, 9)
        sr95 = min_sr_for_target_dsr(0.95, skew, kurt, t, 9)
        rows.append({
            "oos_months": t,
            "ann_sharpe_needed_dsr_0.50": sr50 * np.sqrt(PERIODS_PER_YEAR),
            "ann_sharpe_needed_dsr_0.95": sr95 * np.sqrt(PERIODS_PER_YEAR),
        })
    table = pd.DataFrame(rows).set_index("oos_months")
    print(table.round(3).to_string())

    print("\nSame table under a normal-distribution assumption (skew=0, kurt=3), for comparison:")
    rows_n = []
    for t in [80, 100, 109, 118, 130, 150]:
        sr50 = min_sr_for_target_dsr(0.50, 0.0, 3.0, t, 9)
        sr95 = min_sr_for_target_dsr(0.95, 0.0, 3.0, t, 9)
        rows_n.append({"oos_months": t, "ann_sharpe_needed_dsr_0.50": sr50 * np.sqrt(PERIODS_PER_YEAR),
                        "ann_sharpe_needed_dsr_0.95": sr95 * np.sqrt(PERIODS_PER_YEAR)})
    print(pd.DataFrame(rows_n).set_index("oos_months").round(3).to_string())

    table.round(4).to_csv("../results/comparison/dsr_power_table.csv")
    print("\nSaved ../results/comparison/dsr_power_table.csv")
