# AI-Epi: Mechanistic Grounding for Epidemic Forecasts

AI-Epi compares direct language model forecasts with forecasts that use a language model to adjust transmission in a Bayesian mechanistic model. It evaluates weekly activity for six pathogens in Hong Kong at horizons of one to four weeks. The repository includes the Stan model, rolling-origin fitting and data preparation code, LLM benchmark, and offline reproduction of the reported evaluation.

## Reproduce the evaluation offline

Use Python 3.13. The original Python results were checked with Python 3.13.9.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r bench/requirements.txt
python repro/replay.py --output reproduced
python analysis/table8/reproduce_table8.py --verify --output reproduced/table8.csv
```

The replay starts from the frozen forecasts and other numerical inputs in [`repro/`](repro/) and the weekly surveillance series in [`Data/`](Data/). It writes to a new directory. See [the replay notes](repro/README.md) for which scores, analyses and figures are regenerated and which artifacts are inputs. The [Table 8 implementation](analysis/table8/README.md) reproduces paired absolute log error ratios and their bootstrap intervals, using the forecasts' saved medians. Neither command requires API credentials or a fitted R environment.

## Project layout

| Path | Contents |
| --- | --- |
| `ai_epi.stan`, `ai_epi_core.R`, `export_origins.R` | Mechanistic model, data construction and rolling-origin fits |
| `Data/` | Weekly surveillance data and scripts for building surveillance and contextual inputs |
| `bench/llm_bench/` | Forecast designs, model clients, mechanistic simulator, scoring and plotting |
| `bench/config*.yaml` | Main, retrospective, frontier, and sensitivity experiment definitions |
| `bench/` analysis scripts | Context experiments, forecast-origin swap and research figures |
| `repro/` | Frozen numeric inputs and offline replay |
| `analysis/table8/` | Standalone, verified reproduction of Supplementary Table 8 |

## Generate new forecasts

See the [benchmark guide](bench/README.md) for the R fitting and API workflows. The main study has 24 origins. The retrospective evaluation covers 65 origins from 2015 to 2019. These workflows require a local R/Stan installation. To match the original API experiments, supply the separately licensed OAG passenger workbook and build the importation inputs described in [the contextual-data guide](Data/context/README.md). API providers, model versions, credentials and budgets are separate from the offline evaluation and may affect any newly generated forecasts.

## Run the Python tests

With the same environment activated:

```bash
cd bench
python -m pytest -q
```

The repository contains research code and numerical artifacts for reproduction. The paper's Word files, document-build scripts, raw licensed travel data, API credentials and conversation archives are kept separately.
