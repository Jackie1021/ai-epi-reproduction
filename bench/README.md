# Forecast benchmark

The benchmark compares direct blinded and named LLM forecasts with LLM-guided
mechanistic forecasts. It supports the main, retrospective, frontier,
external-information, reasoning and counterfactual experiments. The archived
design and amendments are in [the protocol](protocol_context_experiment.md).
Model IDs, forecast arms, origin cutoffs, provider settings, and seeds live in
the corresponding `config*.yaml` files.

## Local R and Python setup

From the repository root, install the R packages and the compiler toolchain
required by RStan. R 4.5.3, RStan 2.32.7, and Arrow 23.0.1.2 were used for
the local checks. The other R dependencies are ggplot2, dplyr, tidyr, jsonlite,
patchwork, scales, and ragg.

```bash
Rscript -e 'install.packages(c("rstan","arrow","ggplot2","dplyr","tidyr","jsonlite","patchwork","scales","ragg"), repos="https://cloud.r-project.org")'
bash bench/setup.sh
Rscript export_origins.R --smoke 2025-06-07
bash bench/llm.sh validate-sim --smoke
bash bench/llm.sh all --smoke --mock --run smoke
```

The smoke fit uses short chains to check file formats and the R/Python
connection. Use full runs for research estimates. Run the R entry points from
the repository root, or set `AI_EPI_ROOT` to that directory.

## Full fits and API forecasts

```bash
bash run_export.sh 3
bash run_export_hist.sh 4
bash bench/setup.sh --live
bash bench/llm.sh doctor
bash bench/llm.sh run --dry-run --config config.yaml --origins 2025-06-07
```

Set credentials in the environment or in the ignored `bench/.env` file before
live calls. For the original context arms, provide the separately licensed OAG
workbook and generate the importation series using the
[data guide](../Data/context/README.md). The code accepts missing contextual
inputs for smoke runs. A run with different context is a different experiment.

The corresponding live configuration flags are:

| Study | Configuration |
| --- | --- |
| Main | `config.yaml` |
| Retrospective | `config_hist.yaml` |
| Frontier | `config_frontier.yaml`, `config_frontier_hist.yaml`; see `run_frontier.sh` for the model-specific arm selection |
| Mechanism test | `config_cf.yaml`, `config_cf_reason.yaml`, `config_frontier_cf.yaml` |
| Reasoning sensitivity | `config_reason.yaml`, `config_reason_ext.yaml` |

`bash bench/llm.sh all --config config.yaml` runs forecast generation, scoring
and plotting. `forecasts`, `score` and `plot` run the stages individually.
Live results are written to `bench/runs/<run>/`, and provider responses are
cached locally. The final manuscript evaluation can instead be recalculated
from the saved numerical forecasts using [the offline replay](../repro/README.md).
