"""Tables for the EPS split-basis correction, from the re-run outputs.

Reads results/correction/{published,oldformula,corrected}/ (written by
scripts/correction_rerun.py), the published results, and
results/correction/inputs/, and writes results/correction/tables/:

  reproduction_check.csv      re-run 'published' mode vs the published return series
  sharpe_by_config.csv        OOS Sharpe of the baseline and all 8 configs, per mode
  summary.csv                 baseline and best-ML Sharpe / deflated Sharpe, per mode
  swing_by_config.csv         published -> corrected change per config, vs the baseline's
  per_fold.csv                summed monthly net return per walk-forward fold
  leave_one_out.csv           Sharpe with each fold dropped, baseline and rf_depth10_leaf50, per mode
  factor_ic.csv               full-sample rank IC per factor, per mode
  fold7_split_attribution.csv fold-7 return of names that split after the fold vs the rest
  value_panel_agreement.csv   rank agreement of leaky vs corrected value, split vs never-split names
  col_identity.csv            the cached COL series vs Rockwell Collins

Run from backtester/, in the pinned environment:
  PYTHONPATH=. python scripts/correction_tables.py
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.metrics import deflated_sharpe, max_drawdown, sharpe
from src.backtest.validation import make_walk_forward_folds
from src.data.universe import build_universe
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parent.parent.parent
CORR = ROOT / "results" / "correction"
INPUTS = CORR / "inputs"
OUT = CORR / "tables"
MODES = ["published", "oldformula", "corrected"]
BASELINE = "oos_net_returns"
CONFIGS = sorted(Path(f).stem.replace("ml_oos_net_returns_", "") for f in glob.glob(str(ROOT / "results/ml/ml_oos_net_returns_*.csv")))
PUBLISHED_BEST = "rf_depth10_leaf50"
N_TRIALS = 9
FOLD7 = ("2022-09-01", "2023-09-01")        # fold 7's test block, [start, end)
SPLIT_WINDOW = ("2023-09-01", "2024-12-31")  # a forward split here inflated the leaky value score during fold 7


def cached_adj_close(tickers: list[str], cache_dir: str, start: str, end: str) -> pd.DataFrame:
    """Daily adj_close straight from the price cache (no network, unlike the
    vendored loader -- see correction_rerun.py), wide date x ticker."""
    frames = {}
    for t in tickers:
        f = Path(cache_dir) / "prices" / f"{t.upper()}.parquet"
        if f.exists():
            p = pd.read_parquet(f)
            p = p[(p["date"] >= start) & (p["date"] <= end)]
            if not p.empty:
                frames[t.upper()] = p.set_index("date")["adj_close"]
    return pd.DataFrame(frames).sort_index()


def series(mode: str, name: str) -> pd.Series:
    return pd.read_csv(CORR / mode / f"{name}.csv", index_col=0, parse_dates=True).iloc[:, 0]


def ml(config: str) -> str:
    return f"ml_oos_net_returns_{config}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "backtester" / "config.yaml"))
    cache_dir = str(ROOT / "backtester" / cfg["data"]["cache_dir"])

    # 1. The re-run reproduces the published series exactly.
    rows = []
    for name in [BASELINE] + [ml(c) for c in CONFIGS]:
        published_file = ROOT / ("results/baseline" if name == BASELINE else "results/ml") / f"{name}.csv"
        published = pd.read_csv(published_file, index_col=0, parse_dates=True).iloc[:, 0]
        rerun = series("published", name)
        rows.append({"series": name, "months": len(published), "max_abs_difference": float((published - rerun).abs().max())})
    pd.DataFrame(rows).to_csv(OUT / "reproduction_check.csv", index=False)

    # 2-3. Sharpe by configuration, and the summary per mode.
    sharpes = pd.DataFrame(
        {m: {"baseline": sharpe(series(m, BASELINE), 12), **{c: sharpe(series(m, ml(c)), 12) for c in CONFIGS}} for m in MODES}
    )
    sharpes.rename_axis("config").to_csv(OUT / "sharpe_by_config.csv")
    summary = []
    for m in MODES:
        base = series(m, BASELINE)
        best = sharpes.loc[CONFIGS, m].idxmax()
        best_r = series(m, ml(best))
        summary.append({
            "mode": m,
            "baseline_sharpe": sharpe(base, 12),
            "baseline_dsr_n1": deflated_sharpe(base, 1),
            "baseline_dsr_n9": deflated_sharpe(base, N_TRIALS),
            "best_ml_config": best,
            "best_ml_sharpe": sharpe(best_r, 12),
            "best_ml_dsr_n9": deflated_sharpe(best_r, N_TRIALS),
            "best_ml_max_drawdown": max_drawdown(best_r),
            "configs_beating_baseline": int((sharpes.loc[CONFIGS, m] > sharpes.loc["baseline", m]).sum()),
        })
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)

    # 4. Published -> corrected, per configuration, against the baseline's change. The
    # paired test is on mean monthly return impact, not on the Sharpe swing itself.
    base_delta = series("corrected", BASELINE) - series("published", BASELINE)
    base_swing = sharpes.loc["baseline", "corrected"] - sharpes.loc["baseline", "published"]
    rows = []
    for c in CONFIGS:
        extra = (series("corrected", ml(c)) - series("published", ml(c))) - base_delta
        swing = sharpes.loc[c, "corrected"] - sharpes.loc[c, "published"]
        rows.append({
            "config": c,
            "published_sharpe": sharpes.loc[c, "published"],
            "corrected_sharpe": sharpes.loc[c, "corrected"],
            "sharpe_swing": swing,
            "swing_vs_baseline_swing": swing / base_swing,
            "paired_t_mean_monthly_return_impact_vs_baseline": extra.mean() / (extra.std(ddof=1) / np.sqrt(len(extra))),
        })
    swing_table = pd.DataFrame(rows)
    swing_table.loc[len(swing_table)] = {"config": "baseline", "published_sharpe": sharpes.loc["baseline", "published"],
                                         "corrected_sharpe": sharpes.loc["baseline", "corrected"], "sharpe_swing": base_swing,
                                         "swing_vs_baseline_swing": 1.0, "paired_t_mean_monthly_return_impact_vs_baseline": np.nan}
    swing_table.to_csv(OUT / "swing_by_config.csv", index=False)

    # 5. Per-fold summed monthly net return.
    wf = cfg["validation"]["walk_forward"]
    folds = make_walk_forward_folds(cfg["universe"]["start_date"], cfg["universe"]["end_date"],
                                    wf["initial_train_months"], wf["test_months"], wf["embargo_months"])
    rows = []
    for m in MODES:
        for label, name in (("baseline", BASELINE), (PUBLISHED_BEST, ml(PUBLISHED_BEST))):
            r = series(m, name)
            rows.append({"mode": m, "series": label,
                         **{f"fold {i} ({f['test_start'].date().isoformat()[:7]})": r[(r.index >= f["test_start"]) & (r.index < f["test_end"])].sum()
                            for i, f in enumerate(folds)}})
    pd.DataFrame(rows).to_csv(OUT / "per_fold.csv", index=False)

    # 5b. Leave-one-fold-out, exactly as scripts/subperiod_table.py computed the
    # published Section 5.2 table: drop one fold's months from the pooled series,
    # Sharpe on the rest. The 'published' rows must reproduce that table.
    rows = []
    for m in MODES:
        base, best = series(m, BASELINE), series(m, ml(PUBLISHED_BEST))
        drops = [(str(i), f) for i, f in enumerate(folds)] + [("none (full sample)", None)]
        for label, f in drops:
            keep_b = base.index == base.index if f is None else ~((base.index >= f["test_start"]) & (base.index < f["test_end"]))
            keep_m = best.index == best.index if f is None else ~((best.index >= f["test_start"]) & (best.index < f["test_end"]))
            b, s = sharpe(base[keep_b], 12), sharpe(best[keep_m], 12)
            rows.append({"mode": m, "fold_dropped": label, "baseline_sharpe": b,
                         f"{PUBLISHED_BEST}_sharpe": s, "ml_minus_baseline": s - b})
    pd.DataFrame(rows).to_csv(OUT / "leave_one_out.csv", index=False)

    # 6. Factor IC per mode.
    ic = pd.concat({m: pd.read_csv(CORR / m / "factor_ic.csv", index_col=0) for m in MODES}, names=["mode"])
    ic.to_csv(OUT / "factor_ic.csv")

    # 7. Fold-7 attribution: names that forward-split after the fold vs the rest, per leg.
    events = pd.read_csv(INPUTS / "split_events_d68487a.csv", parse_dates=["date"])
    later = set(events[(events["split_ratio"] > 1) & (events["date"] >= SPLIT_WINDOW[0]) & (events["date"] <= SPLIT_WINDOW[1])]["ticker"])
    rows = []
    for m in ("published", "corrected"):
        w = pd.read_parquet(CORR / m / "holdings" / f"weights_{PUBLISHED_BEST}.parquet")
        held_names = sorted(w.columns[(w != 0).any()])
        daily = cached_adj_close(held_names, cache_dir, cfg["universe"]["start_date"], cfg["universe"]["end_date"])
        monthly = daily.resample("ME").last()
        fwd = monthly.pct_change().shift(-1)  # exactly as the study's pipeline computes it
        months = w.index[(w.index >= FOLD7[0]) & (w.index < FOLD7[1])]
        wf7 = w.loc[months]
        contrib = wf7 * fwd.reindex(index=months, columns=w.columns).fillna(0.0)
        for leg, sign in (("long", 1), ("short", -1)):
            leg_weight = wf7.where(np.sign(wf7) == sign).abs().sum(axis=1)
            for group, cols in (("forward split after the fold", [c for c in w.columns if c in later]),
                                ("no forward split after the fold", [c for c in w.columns if c not in later])):
                held = np.sign(wf7[cols]) == sign
                rows.append({"mode": m, "config": PUBLISHED_BEST, "leg": leg, "group": group,
                             "names_held": int(held.any().sum()),
                             "mean_share_of_leg_weight": float((wf7[cols].where(held).abs().sum(axis=1) / leg_weight).mean()),
                             "return_contribution": float(contrib[cols].where(held).sum().sum())})
    pd.DataFrame(rows).to_csv(OUT / "fold7_split_attribution.csv", index=False)

    # 8. Where the corrected value differs from the leaky one.
    leaky = pd.read_parquet(INPUTS / "value_oldformula_d68487a.parquet")
    corrected = pd.read_parquet(INPUTS / "value_corrected_d68487a.parquet")
    split_names = set(events["ticker"])
    rows = []
    for group, cols in (("split at least once 2008-2024", [c for c in corrected.columns if c in split_names]),
                        ("never split", [c for c in corrected.columns if c not in split_names])):
        corrs = []
        for d in corrected.index:
            a, b = leaky.loc[d, cols], corrected.loc[d, cols]
            ok = a.notna() & b.notna()
            if ok.sum() > 20:
                corrs.append(a[ok].rank().corr(b[ok].rank()))
        rows.append({"group": group, "names": len(cols), "months": len(corrs), "mean_monthly_rank_correlation": float(np.mean(corrs))})
    rows.append({"group": "all (non-NaN cells, leaky / corrected)", "names": corrected.shape[1], "months": corrected.shape[0],
                 "mean_monthly_rank_correlation": np.nan, "non_nan_leaky": int(leaky.notna().sum().sum()),
                 "non_nan_corrected": int(corrected.notna().sum().sum())})
    pd.DataFrame(rows).to_csv(OUT / "value_panel_agreement.csv", index=False)

    # 9. COL: the study's cached series vs Rockwell Collins.
    universe = build_universe(cfg["universe"]["name"], cfg["universe"]["start_date"], cfg["universe"]["end_date"], cache_dir=cache_dir)
    member = universe["COL"][universe["COL"]]
    cached = pd.read_parquet(Path(cache_dir) / "prices" / "COL.parquet").set_index("date")["adj_close"].resample("ME").last()
    rockwell = pd.read_csv(INPUTS / "col_monthly_d68487a.csv", index_col=0, parse_dates=True)["adj_close"]
    both = pd.concat({"cached": cached.pct_change(), "rockwell_collins": rockwell.pct_change()}, axis=1).dropna()
    pd.DataFrame([{
        "sp500_member_from": member.index.min().date(), "sp500_member_to": member.index.max().date(),
        "cached_series_from": cached.index.min().date(), "cached_series_to": cached.index.max().date(),
        "rockwell_collins_from": rockwell.index.min().date(), "rockwell_collins_to": rockwell.index.max().date(),
        "overlap_months": len(both), "monthly_return_correlation": float(both.corr().iloc[0, 1]),
    }]).to_csv(OUT / "col_identity.csv", index=False)

    for f in sorted(OUT.glob("*.csv")):
        print(f"\n== {f.name}")
        print(pd.read_csv(f).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
