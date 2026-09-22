"""Check for ticker-identity contamination: a base ticker whose price
history spans two different real-world entities (e.g. "DOW" = old Dow
Chemical pre-2017 vs new Dow Inc. post-2019), which would pair a
point-in-time membership window meant for one company with price data from
a different one.

Two checks:
 1. Which tickers have >1 separate (non-contiguous) True-membership run in
    the universe matrix -- a gap-then-reappear pattern is the structural
    signature of ticker reuse and the only place this bug could hide.
 2. For each such ticker, does its price series show a level discontinuity
    at the reappearance boundary consistent with two unrelated
    instruments, and does its price coverage actually overlap each
    membership window (contamination) or fall entirely outside the old
    window (safe failure -- already the case per the missing-ticker check)?
"""
import sys; sys.path.insert(0, ".")
import pandas as pd
from src.data.universe import build_universe
from src.data.loader import load_prices
from src.utils.config import load_config

cfg = load_config("config.yaml")
u, d = cfg["universe"], cfg["data"]
uni = build_universe(u["name"], u["start_date"], u["end_date"], cache_dir=d["cache_dir"])

def runs(col):
    s = uni[col]
    out, start = [], None
    prev = False
    for dt, val in s.items():
        if val and not prev:
            start = dt
        if not val and prev:
            out.append((start, dt))
        prev = val
    if prev:
        out.append((start, s.index[-1]))
    return out

multi_run = {t: runs(t) for t in uni.columns if len(runs(t)) > 1}
print(f"Tickers with >1 separate membership run (reuse candidates): {len(multi_run)}")
for t, rs in sorted(multi_run.items()):
    print(f"  {t}: {[(a.date(), b.date()) for a,b in rs]}")

# Focus on the flagged names plus anything with a multi-run pattern
focus = sorted(set(multi_run) | {"DOW","DELL","CEG","EMC","ADT","BEAM","FCPT","JAVA","MHS","MI","MMI","NE","NSM","S","SE","SHLD","TE"})
tk = sorted(set(focus) & set(uni.columns))
print(f"\nChecking price continuity for {len(tk)} names: {tk}")

px = load_prices(tk, u["start_date"], u["end_date"], cache_dir=d["cache_dir"])
monthly = px.pivot(index="date", columns="ticker", values="adj_close").resample("ME").last()

for t in tk:
    if t not in monthly.columns:
        print(f"{t}: no price data returned at all")
        continue
    series = monthly[t].dropna()
    if series.empty:
        print(f"{t}: price column exists but all-NaN")
        continue
    px_start, px_end = series.index.min(), series.index.max()
    windows = multi_run.get(t, runs(t))
    overlaps = []
    for w_start, w_end in windows:
        window_prices = series[(series.index >= w_start) & (series.index < w_end)]
        overlaps.append((w_start.date(), w_end.date(), len(window_prices)))
    # biggest month-over-month ratio jump anywhere in the series
    ratio = (series / series.shift(1)).dropna()
    max_jump = ratio.max() if len(ratio) else float("nan")
    min_jump = ratio.min() if len(ratio) else float("nan")
    print(f"{t}: price data {px_start.date()} -> {px_end.date()} ({len(series)} pts) | "
          f"membership windows+coverage: {overlaps} | max monthly ratio {max_jump:.2f}x, min {min_jump:.2f}x")
