# EPS split-basis correction: re-run

Re-runs this study with the value feature corrected for a look-ahead leak, beside the
published results (which are not modified). Numbers and their sources: [FACTS.md](FACTS.md).

## Layout

```
inputs/                  built by factor-backtester d68487a (build_value_panels.py)
  value_corrected_d68487a.parquet    corrected earnings-yield panel
  value_oldformula_d68487a.parquet   published formula on the same re-downloaded data
  split_events_d68487a.csv           all splits 2008-2024 (for the fold-7 attribution)
  col_monthly_d68487a.csv            Rockwell Collins month-end prices (COL identity check)
published/  oldformula/  corrected/  return series + factor IC per mode
  holdings/                          rf_depth10_leaf50 holdings (published, corrected)
tables/                  derived by backtester/scripts/correction_tables.py
```

## Reproducing

In the pinned environment (`backtester/requirements.txt`, Python 3.12.9), from `backtester/`:

```bash
PYTHONPATH=. python scripts/correction_rerun.py --mode published    # must match the published series
PYTHONPATH=. python scripts/correction_rerun.py --mode oldformula
PYTHONPATH=. python scripts/correction_rerun.py --mode corrected
PYTHONPATH=. python scripts/correction_tables.py
```

Each mode takes about 20 minutes and runs offline from the local data cache; see
`correction_rerun.py` for why. `tables/reproduction_check.csv` must show a maximum
difference of 0 for every series, or the environment differs from the published one. A
re-run also writes holdings for every configuration under `<mode>/holdings/`; only the two
used by the fold-7 table are committed.

To rebuild `inputs/` instead of using the committed copies, run
`inputs/build_value_panels.py` from a clean factor-backtester checkout at `d68487a` with
its data cache populated.
