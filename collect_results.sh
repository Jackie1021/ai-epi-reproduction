#!/usr/bin/env bash
# Copy the core result figures and tables into results/ (numbered in reading order): PNG for review
# at the top level, vector PDF (embedded fonts) in results/pdf/, CSV tables in results/tables/, the
# information given to the LLMs in results/context_data/, plus an index README.md with journal-style
# figure legends. Called automatically by plot_origins.R and `llm_bench plot`; can also be run by
# hand: bash collect_results.sh
cd "$(dirname "$0")"
R=results
mkdir -p "$R/pdf" "$R/tables" "$R/context_data" "$R/data"
S=output/figures/origins
L=bench/runs/main
H=bench/runs/hist            # historical replication 2015-2019 (protocol amendment 8)
X=Data/context
put() { [ -f "$1" ] && cp -p "$1" "$R/$2"; }
fig() { put "$1.png" "$2.png"; put "$1.pdf" "pdf/$2.pdf"; }   # figure: png + pdf

# files replaced by the composite figures below (older layout)
for f in 04_stan_sampling_diagnostics.csv 07_llm_interval_coverage 08_llm_by_pathogen_direct_blind \
         09_llm_by_pathogen_direct_named 10_llm_by_pathogen_r0_mechanistic 11_llm_relative_wis_table.csv \
         12_llm_relative_wis_log_table.csv 13_llm_interval_coverage_table.csv 14_llm_context_gain \
         15_llm_by_pathogen_r0_context 16_llm_context_gain_table.csv 05_llm_relative_wis_main.pdf \
         05_llm_relative_wis_main 06_llm_mechanistic_scaffold_gain 07_llm_relative_wis_by_pathogen \
         08_llm_calibration_and_robustness 09_llm_external_information; do
  rm -f "$R/$f" "$R/$f.png" "$R/pdf/$f.pdf"
done
rm -f "$R/context_data/hk_influenza_importation_pressure_weekly.csv" "$R/tables/llm_external_information_gain.csv"

fig "$S/fit_origin_val_end"               "00_stan_fit_to_val_end_2025-06-07"
fig "$S/rolling_forecasts"                "01_stan_rolling_forecasts"
fig "$S/rolling_forecasts_log"            "02_stan_rolling_forecasts_log"
put "$S/fits_by_origin.pdf"               "03_stan_fits_by_origin.pdf"
fig "$S/sampling_diagnostics"             "04_stan_sampling_diagnostics"
fig "$S/parameters_prior_posterior"       "05_stan_parameters_prior_posterior"
fig "$L/figures/fig_main_relwis"          "06_llm_relative_wis_main"
fig "$L/figures/fig_scaffold_gain"        "07_llm_mechanistic_scaffold_gain"
fig "$L/figures/fig_by_pathogen_heatmap"  "08_llm_relative_wis_by_pathogen"
fig "$L/figures/fig_calibration"          "09_llm_calibration_and_robustness"
fig "$L/figures/fig_external_information" "10_llm_external_information"
fig "$L/figures/fig_llm_phase"            "11_llm_gain_by_epidemic_phase"
SH=output/figures/origins_hist
fig "$SH/rolling_forecasts"               "12_stan_historical_rolling_forecasts_2015-2019"
fig "$SH/sampling_diagnostics"            "13_stan_historical_sampling_diagnostics"
put "$SH/fits_by_origin.pdf"               "15_stan_historical_fits_by_origin_2015-2019.pdf"
fig "$H/figures/fig_a8_replication"       "14_llm_historical_replication_2015-2019"
put "$SH/diagnostics.csv"                 "tables/stan_historical_sampling_diagnostics.csv"

put "$S/diagnostics.csv"                  "tables/stan_sampling_diagnostics.csv"
put "$S/parameters_prior_posterior.csv"   "tables/stan_parameters_prior_posterior.csv"
put "$L/table_relative_wis.csv"           "tables/llm_relative_wis.csv"
put "$L/table_relative_wis_log.csv"       "tables/llm_relative_wis_log_scale.csv"
put "$L/table_coverage.csv"               "tables/llm_interval_coverage.csv"
put "$L/table_relative_wis_by_period.csv" "tables/llm_relative_wis_by_period.csv"
put "$L/table_llm_gain_by_phase.csv"      "tables/llm_gain_by_epidemic_phase.csv"
put "$L/table_phase_at_origin.csv"        "tables/llm_epidemic_phase_at_origin.csv"
put "$L/table_direction_accuracy.csv"     "tables/llm_direction_accuracy.csv"
put "$L/table_context_experiment.csv"     "tables/llm_external_information_experiment.csv"
put "$L/table_importation_signal.csv"     "tables/llm_importation_signal_by_origin.csv"
put "$L/table_context_gain.csv"           "tables/llm_external_information_stepwise_first_call.csv"
put "$H/table_a8.csv"                     "tables/llm_historical_replication_a8.csv"
put "$H/table_relative_wis.csv"           "tables/llm_relative_wis_2015-2019.csv"
put "$H/table_relative_wis_log.csv"       "tables/llm_relative_wis_log_scale_2015-2019.csv"
put "$H/table_coverage.csv"               "tables/llm_interval_coverage_2015-2019.csv"
put "$H/table_phase_at_origin.csv"        "tables/llm_epidemic_phase_at_origin_2015-2019.csv"
put Data/HK_weekly_6pathogens.csv         "data/HK_weekly_6pathogens.csv"
put Data/HK_weekly_6pathogens_2014-2026.png "data/HK_weekly_6pathogens_2014-2026.png"
put Data/README_HK_weekly.md              "data/README_HK_weekly.md"
put bench/protocol_context_experiment.md  "protocol_external_information_experiment.md"

put "$X/hk_calendar.csv"                  "context_data/hk_school_calendar_and_public_holidays.csv"
put "$X/hk_weather_weekly.csv"            "context_data/hk_weather_weekly.csv"
put "$X/hk_weather_daily.csv"             "context_data/hk_weather_daily.csv"
put "$X/README.md"                        "context_data/README.md"
put "$X/importation/importation_weekly.csv"    "context_data/hk_influenza_importation_weekly.csv"
put "$X/importation/importation_by_source.csv" "context_data/hk_influenza_importation_by_source.csv"
put "$X/importation/README.md"            "context_data/importation_README.md"

A=$(ls -d output/archive/run1_hard_bounds_* 2>/dev/null | head -1)
if [ -n "$A" ]; then
  C="$R/comparison_run1_hard_bounds"
  mkdir -p "$C"
  cp -p "$A/figures/rolling_forecasts.png" "$C/rolling_forecasts_first6_origins.png" 2>/dev/null
  cp -p "$A/figures/fit_2024-07-06.png"    "$C/fit_origin_2024-07-06.png" 2>/dev/null
  cp -p "$A/figures/diagnostics.csv"       "$C/sampling_diagnostics.csv" 2>/dev/null
fi

n_stan=$(find output/origins -name meta.json 2>/dev/null | wc -l | tr -d ' ')
# planned origins (ai_epi_core.R): count, first and last, e.g. "24|6 July 2024|11 April 2026"
plan=$(Rscript -e 'suppressMessages(invisible(capture.output(source("ai_epi_core.R")))); invisible(Sys.setlocale("LC_TIME", "C")); o <- default_origins(); d <- function(x) paste(as.integer(format(x, "%d")), format(x, "%B %Y")); cat(length(o), d(min(o)), d(max(o)), sep = "|")' 2>/dev/null | tail -1)
IFS='|' read -r n_orig first_o last_o <<< "$plan"
left=$(cd bench && bash llm.sh pending --quiet 2>/dev/null | tail -1)
total=$(cd bench && .venv/bin/python -c "
from llm_bench.config import load_config
from llm_bench.runner import all_arms, sample_plan
c = load_config(); m = [x for x in c.models() if x.enabled and x.configured]
a = all_arms(c); f = sample_plan(c, a)
print(${n_orig:-24} * sum(f(arm, x.id) for arm in a for x in m))" 2>/dev/null)

cat > "$R/README.md" <<EOF
# Results (updated $(date '+%Y-%m-%d %H:%M'))

**Status.** Mechanistic model fitted at ${n_stan}/${n_orig} forecast origins. LLM requests still without
an answer: ${left:-?} / ${total:-?} (free-tier daily limits). Figures 06-10 and the LLM tables are
interim until this reaches 0; they are regenerated after every round of requests.

PNG files (400 dpi) are for review; vector PDFs with embedded fonts are in \`pdf/\`; the numbers
behind the figures are in \`tables/\`; the information given to the LLMs is in \`context_data/\`.
All figures are 183 mm wide (double column), Arial 7 pt, with colour-blind-safe colours; in the
heatmaps blue = better and red = worse than the reference, with the values printed in the cells.

## Mechanistic model (Stan)

**00_stan_fit_to_val_end_2025-06-07.** Model fit and 4-week forecast with data up to 7 June 2025
(val_end, the cut-off of the main analysis). Each pathogen is fitted from the start of its own
epidemic (dashed line). Blue: posterior median of expected weekly cases with 50% and 95% credible
intervals. Orange: posterior predictive median with 50% and 95% prediction intervals for the next
4 weeks (grey background), holding R0 at its last fitted value. Black points: observations used for
fitting; open circles: held-out observations; grey points: observations before the fitting window.
Forecast intervals are truncated at 1.5 times the largest observed count.

**01_stan_rolling_forecasts.** Rolling 4-week forecasts with constant R0 from ${n_orig} forecast origins,
every 4 weeks from ${first_o} to ${last_o}. At each origin the model is refitted using only
data up to that origin, starting from each pathogen's own epidemic start (dashed line). Lines and
bands: forecast median with 50% and 95% prediction intervals; points: observed weekly cases.
Intervals are truncated at 1.5 times the largest observed count.

**02_stan_rolling_forecasts_log.** As 01, on a log(1 + cases) scale and without truncation.

**03_stan_fits_by_origin.pdf.** One page per forecast origin, laid out as 00; the page header gives
the maximum R-hat, the number of divergent transitions and the run time.

**04_stan_sampling_diagnostics.** Sampling diagnostics for every forecast origin (columns) and
independently fitted model block (rows). A: maximum R-hat over all parameters. B: divergent
transitions as a percentage of the 2,000 post-warm-up draws (4 chains x 500). C: minimum effective
sample size. Redder cells need more attention.

**05_stan_parameters_prior_posterior.** Prior (grey, dashed outline) and posterior (blue) distributions
of the core epidemiological parameters of the mechanistic model fitted with data up to 7 June 2025 (the
fit of figure 00); vertical lines: posterior medians. A: basic reproduction number R0 at the start of
each pathogen's fitting window; B: initial susceptible fraction; C: mean duration of immunity (years,
log scale); D: ascertainment rate (for HFMD, the proportion of infections hospitalised); E: strength of
cross-immunity between influenza A(H3N2) and A(H1N1), and between influenza A and B. The latent and
infectious periods are fixed from the literature and not shown. Posteriors close to their priors
(duration of immunity, cross-immunity, some ascertainment rates) are not identified by the data and
depend on the prior. Densities: kernel estimates from 100,000 prior draws and 2,000 posterior draws.

## Language models vs the mechanistic model

Models (free tiers; each model is served by a single provider throughout): Gemini 3.8 Flash,
Gemini 2.5 Flash, Gemma 4 31B and Gemma 4 26B A4B (Google AI Studio); Qwen3.8 27B, GLM 5.2 and
DeepSeek V4.1 Flash (Alibaba ModelScope); GPT-OSS 20B and Kimi K3 (NVIDIA NIM); Nemotron 3 Ultra and
Ling 3.0 Flash Sante (OpenRouter). Reasoning ("thinking") was switched off where the provider allows
it and set to low otherwise; Gemini at its default. Models from the same developer share a colour
and are told apart by line style and marker. Designs:
(i) *direct forecast, blinded*: the LLM forecasts weekly counts from rescaled series without dates,
place or pathogen names; (ii) *direct forecast, named*: the same with real dates, Hong Kong and
pathogen names; (iii) *LLM R0 + mechanistic model*: the LLM proposes R0 multipliers for the next
4 weeks (pathogen names and season given, year and city withheld) and the fitted mechanistic model
turns them into case forecasts; (iv) as (iii) plus the school calendar, public holidays and weekly
weather; (v) as (iv) plus the importation pressure of influenza (see \`context_data/\`). Designs
(iii)-(v) add one block of information at a time; figures 06-09 use the first call of every request.
Forecast origins from November 2025 form a prospective test period: their data were not available
when the models, prompts and analysis were specified (\`tables/llm_relative_wis_by_period.csv\`).
Relative WIS is the mean weighted interval score
of a method divided by that of the mechanistic model with constant R0 over the same forecast
targets, combined across pathogens by the geometric mean; values below 1 mean better forecasts than
this reference. Model-design combinations scored at fewer than 5 origins are not shown.

**06_llm_relative_wis_main.** Relative WIS by forecast horizon, one panel per design. Coloured
lines: language models; black line: reference (mechanistic model with constant R0); grey dotted line:
naive persistence.

**07_llm_mechanistic_scaffold_gain.** Paired comparison of relative WIS pooled over pathogens and
horizons (geometric mean) for each language model; a line going down means better forecasts.
Log-scale WIS. A: direct forecast (blinded) vs LLM R0 + mechanistic model. B: direct forecast with a
blinded vs a named prompt. Horizontal lines: reference (black) and baselines (grey). Points are slightly offset
horizontally so that they do not overlap.

**08_llm_relative_wis_by_pathogen.** Relative WIS by pathogen (columns; All = pooled) and model
(rows), one panel per design; the mechanistic and naive baselines are repeated at the bottom of
each panel for reference. Values are geometric means over forecast horizons 1-4 weeks.

**09_llm_calibration_and_robustness.** A, B: empirical coverage (%) of the central 50% and 90%
prediction intervals by model and design (columns (i)-(v) as defined above), pooled over pathogens,
horizons and origins; red cells are below the nominal level (intervals too narrow, overconfident),
blue cells above it; baseline coverage does not depend on the design and spans the row. C: relative
WIS on the natural scale vs on the log scale (WIS of log(1 + counts), as recommended for epidemic
forecasts) for every model-design pair. D: relative WIS in the development period (origins to
October 2025) vs the prospective period (origins from November 2025, whose data were not available
when the models and prompts were specified). In C and D points near the diagonal mean that the
conclusion does not depend on the scale or on the period.

**10_llm_external_information.** Value of the information added to the LLM R0 design: a 2x2
factorial experiment (calendar & weather x influenza importation pressure) with a placebo, run with
Gemma 4 31B, Gemma 4 26B A4B, GPT-OSS 20B, Qwen3.8 27B, DeepSeek V4.1 Flash and Kimi K3, three calls
per request (equal-weight mixture); design and analysis were pre-specified (\`protocol_external_information_experiment.md\`).
Designs: (iii), (iv) and (v) as above; (vi) as (iii) plus the importation pressure only; (vii) as (v)
but with a placebo importation block, the same information taken 13-39 weeks before the origin.
A: the information over the evaluation period (triangles: forecast origins): school and public
holidays, weekly temperature and absolute humidity relative to the 1991-2020 normal, and the
influenza importation pressure by subtype (1 = mean of July 2023 to June 2024); circles: importation
signal at each origin as shown in the prompt (filled: relative pressure above 1 or up more than 30%
over the 4 weeks shown; open: low and flat). B-E: ratio of WIS with / without the information for
the same model and forecast targets, pooled over horizons 1-4 weeks; per pathogen the WIS is summed
over targets and the log ratios are averaged over pathogens. B: main effect of calendar and weather
(mean of (iv)/(iii) and (v)/(vi), all pathogens). C: main effect of the importation pressure (mean
of (vi)/(iii) and (v)/(iv), influenza). D: real vs placebo importation ((v)/(vii), influenza).
E: C within origins with a high or rising (filled) and a low and flat (open) importation signal.
Bars: 95% bootstrap intervals over forecast origins; pooled: equal-weight mean over models with a
two-level bootstrap (origins, then models). Numbers on the right: origins with all five designs
answered. Below 1 = the information improved the forecasts.

**11_llm_gain_by_epidemic_phase.** When does the LLM help? WIS of design (iii) (LLM R0 + mechanistic
model) divided by that of the same mechanistic model with constant R0, for the same targets
(protocol amendment 7; models with at least 20 scored origins). A-C: by epidemic phase at the
forecast origin, from the fitted effective reproduction number Rt in the week before the origin
(growing: Rt at least 1.05; declining: at most 0.95); bars: 95% bootstrap intervals over origins;
pooled: equal-weight mean over models with a two-level bootstrap (origins, then models). D: the
LLM's median R0 multiplier for week 4 by phase (boxes: interquartile range; whiskers: 1.5 x IQR).
E, F: pooled ratio by pathogen and by horizon. Below 1 = the LLM improved the forecast.

**12_stan_historical_rolling_forecasts_2015-2019.** As 01, for the 65 origins of the historical
study (every 4 weeks from 31 January 2015 to 28 December 2019; protocol amendment 8). Each origin
is fitted on the preceding 130 weeks at most. Each pathogen starts at the trough in the first 26
weeks of that window. COVID-19 did not exist.

**13_stan_historical_sampling_diagnostics.** As 04, for the historical origins (one column per
origin; values in \`tables/stan_historical_sampling_diagnostics.csv\`).

**15_stan_historical_fits_by_origin_2015-2019.pdf.** As 03, one page per historical origin (65
pages). For the 14 origins refitted under amendment 8, the page shows the refit; its header gives the
maximum R-hat, the number of divergent transitions and the run time.

**14_llm_historical_replication_2015-2019.** Replication in 2015-2019 (protocol amendment 8; 65
origins, five pathogens, no COVID-19; models with at least 52 scored origins). A-C: WIS of design
(iii) divided by that of the mechanistic model with constant R0, by epidemic phase at the origin, per
model and pooled. A: the primary test of the amendment. D: pooled ratio by phase in 2024-2026 (six
pathogens, amendment 7), in 2015-2019, and in 2015-2026. The 2015-2026 estimate uses both periods
together, five pathogens, and resamples origins within each period. E: every design vs constant R0
in 2015-2019, all phases. F: direct forecast named / blinded in each period. The 2015-2019 seasons
are in the LLMs' training data; a clearly lower ratio there would suggest recall. Bars: 95% bootstrap
intervals; pooled: two-level bootstrap (origins, then models). Below 1 = the LLM (or, in F, naming
the place and dates) improved the forecasts.

## data/
The weekly model input for the six pathogens, 2014-2026, with its construction and checks
(\`README_HK_weekly.md\`) and an overview plot. The main study uses the weeks from 2023; the
historical study uses 2014-2019.

## tables/
Numbers behind the figures: \`stan_sampling_diagnostics.csv\` (04), \`stan_parameters_prior_posterior.csv\` (05; prior and
posterior median and 95% interval of every parameter shown), \`llm_relative_wis.csv\` and
\`llm_relative_wis_log_scale.csv\` (06-09; by model, design, pathogen and horizon, \`n\` = scored
targets), \`llm_interval_coverage.csv\` (09), \`llm_external_information_experiment.csv\` (10; every
pre-specified effect by model and pooled, also by horizon, by pathogen, by importation signal and on
the log scale; \`p_better\` = share of bootstrap samples with a ratio below 1),
\`llm_importation_signal_by_origin.csv\` (the signal strata of 10 E) and
\`llm_external_information_stepwise_first_call.csv\` ((iv) vs (iii) and (v) vs (iv) for all 8 models,
first call only; descriptive).

## context_data/
School calendar and public holidays (EDB example school calendars 2015/16-2019/20 and
2023/24-2025/26, general-holiday lists 2014/15-2019/20 and 2023/24-2025/26), Hong Kong Observatory weekly weather with 1991-2020 normals, and the weekly
influenza importation pressure (air, land and sea arrivals weighted by the positivity at the
travellers' origins: WHO FluNet, Chinese National Influenza Center reports, Taiwan CDC); sources,
methods, references and limitations in \`README.md\` and \`importation_README.md\`.

## comparison_run1_hard_bounds/
Run 1 (hard R0 bounds, adapt_delta 0.95), stopped after 6 origins because of more divergent
transitions and very slow late origins; kept to document why run 2 uses smooth bounds. Figures in
the earlier style.
EOF
echo "results/ updated: $(ls "$R" | grep -c '^[0-9]') files (+ $(ls "$R/pdf" | wc -l | tr -d ' ') pdf, $(ls "$R/tables" | wc -l | tr -d ' ') tables)"
