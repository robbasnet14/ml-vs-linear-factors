"""Monthly cross-sectional rank IC: how well a raw factor value, on its own,
predicts forward returns within each rebalance date's cross-section.

This is a full-sample descriptive statistic about a factor's own predictive
power — distinct from `src.ml.importance.rank_ic`, which scores a MODEL's
predictions on out-of-sample fold data and assumes a dense (already-imputed)
score column. A raw factor column (`mom_z`/`val_z`/`qual_z`) can have real
per-ticker NaNs on a given date, and `scipy.stats.spearmanr` propagates a
single NaN into a NaN for the whole month — so this module drops NaNs
pairwise, per date, before scoring, rather than reusing that function as-is.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_NAMES_PER_MONTH = 20


def monthly_rank_ic(panel: pd.DataFrame, factor_col: str, target_col: str = "fwd_ret",
                     date_col: str = "date", min_names: int = MIN_NAMES_PER_MONTH) -> pd.Series:
    """Spearman rank correlation of `factor_col` vs. `target_col`, computed
    separately within each `date_col` cross-section (dropping NaN pairs
    first), returned as a Series indexed by date. A date with fewer than
    `min_names` valid (non-NaN in both columns) rows, or with no variance in
    either column, is skipped entirely rather than included as a degenerate
    0 or NaN — it contributes no information about the factor's rank
    relationship with returns that month.
    """
    ics = {}
    for date, group in panel.groupby(date_col):
        g = group[[factor_col, target_col]].dropna()
        if len(g) < min_names or g[factor_col].nunique() < 2 or g[target_col].nunique() < 2:
            continue
        ic, _ = spearmanr(g[factor_col], g[target_col])
        if not np.isnan(ic):
            ics[date] = ic
    return pd.Series(ics, name=factor_col).sort_index()


def summarize_ic(ic_series: pd.Series) -> dict:
    """Count, mean, std (sample, ddof=1), and t-stat (mean / (std / sqrt(n)))
    of a monthly IC series — the standard descriptive summary of a factor's
    IC track record. Returns NaN for std/t-stat if fewer than 2 observations."""
    n = len(ic_series)
    mean = float(ic_series.mean()) if n else float("nan")
    std = float(ic_series.std(ddof=1)) if n > 1 else float("nan")
    t_stat = mean / (std / np.sqrt(n)) if n > 1 and std else float("nan")
    return {"n": n, "mean": mean, "std": std, "t_stat": t_stat}
