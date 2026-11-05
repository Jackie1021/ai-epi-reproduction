"""Build runs/main_frontier/ from the saved forecasts of the main study (runs/main/) and the frontier extension
(runs/frontier/): the 11 main-study LLMs and the baselines from runs/main, the two frontier LLMs from runs/frontier.
Then score and plot it with config_main_frontier.yaml:
    .venv/bin/python combine_frontier.py
    bash llm.sh score --config config_main_frontier.yaml --run main_frontier
    .venv/bin/python combine_frontier.py --after-score
    bash llm.sh plot  --config config_main_frontier.yaml --run main_frontier
--after-score keeps the pooled estimates of the main study: the pooled rows of the external-information and phase
tables are those of runs/main, and the phase table keeps the main study's rows for its 11 models (their bootstrap
intervals otherwise shift slightly with the other models in the resampling); only the frontier rows are new.
"""
import sys

import pandas as pd

from llm_bench.runner import runs_root

MAIN, FRONT, OUT = (runs_root() / run for run in ("main", "frontier", "main_frontier"))
FRONTIER = ["gpt61sol", "gpt6astra"]


def combine(name: str) -> None:
    a, b = pd.read_parquet(MAIN / name), pd.read_parquet(FRONT / name)
    out = pd.concat([a[~a["model"].isin(FRONTIER)], b[b["model"].isin(FRONTIER)]], ignore_index=True)
    out.to_parquet(OUT / name, index=False)
    print(f"{name}: {len(a)} + {int(b['model'].isin(FRONTIER).sum())} rows, models: {sorted(out['model'].unique())}")


def keep_main_pooled() -> None:
    for name in ("table_context_experiment.csv", "table_llm_gain_by_phase.csv"):
        a, b = pd.read_csv(MAIN / name), pd.read_csv(OUT / name)
        out = pd.concat([a, b[b["model"].isin(FRONTIER)]], ignore_index=True)[a.columns]
        out.to_csv(OUT / name, index=False)
        print(f"{name}: main-study rows and pooled estimates from runs/main, "
              f"{int(b['model'].isin(FRONTIER).sum())} frontier rows")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--after-score" in sys.argv:
        keep_main_pooled()
    else:
        for name in ("forecasts.parquet", "forecasts_context.parquet", "r0_multipliers.parquet"):
            combine(name)
