# Offline evaluation

```bash
python repro/replay.py --output reproduced
python analysis/table8/reproduce_table8.py --verify --output reproduced/table8.csv
```

The first command verifies the SHA-256 manifest, stages a copy of the frozen
inputs in a new directory, and runs the same scoring and bootstrap functions
used by the benchmark. It checks the recomputed tables against the references
and writes the detailed log to `reproduced/replay.log`. Use `--no-figures` for
tables only. The second command independently checks all 36 Table 8 rows.

- Main, retrospective, frontier and reasoning runs are scored from saved
  forecast quantiles and the bundled weekly observations. The replay
  recomputes relative WIS, coverage, prospective period and phase analyses,
  and the main head-to-head WIS comparisons.
- Forecast-origin swap statistics are rescored and re-bootstrapped from
  saved numerical swap predictions. The mechanism test is re-estimated from
  saved numerical R0 medians, with the original model exclusions for each run.
- The pooled external-information contrasts are re-estimated from saved
  context forecasts. The original travel-signal strata and the corresponding
  input time series require the licensed source data.
- The Stan fit diagnostic summaries are archived in `snapshots/archived_stan/`.
  The full fitted posteriors are produced separately by the R fitting
  workflow. `snapshots/earlier_vintages/` holds earlier exploratory
  sensitivity summaries whose forecast coverage differs from the frozen final
  inputs. These analyses are recalculated for the included data vintage.

`snapshots/manifest.json` lists the relative study source, selection and
SHA-256 for each included snapshot. The replay uses numeric forecasts and
study inputs, so it can run without R, API keys, or a network connection once
the Python environment is installed. It does not regenerate model responses
or the Monte Carlo forecasts that were generated from those responses.
