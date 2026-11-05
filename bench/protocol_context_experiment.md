# Protocol: value of external information for LLM-guided R0 forecasts

**Status:** pre-specified on 25 September 2026, before any request for the new designs (vi, vii) or
for the second and third calls of designs (iii)-(v) was sent. Changes after this date are listed
under *Amendments* with their dates and reasons.

Code: `bench/llm_bench/` (designs `prompts.py`; call plan and forecasts `runner.py`; analysis
`scoring.py: context_effects, import_signal`; figure `plots.py: _fig_external`). Configuration:
`bench/config.yaml`, section `context_experiment`.

## 1. Disclosure: what had been seen before this protocol
Designs (iii) *LLM R0 + mechanistic model*, (iv) *(iii) + calendar & weather* and (v) *(iv) +
importation pressure* had been run with one call per request for 8 models, and the interim results
had been examined:
- (iv) vs (iii) and (v) vs (iv): no clear average effect; every 95% interval included 1.
- An exploratory, post hoc split of (v) vs (iv) by importation signal suggested a benefit where the
  signal was high or rising (19 origin-subtypes) and harm where it was low and flat.

The stratified hypothesis (H4) was therefore **generated from these data**. It is pre-specified here
for the new data, but the first call (sample 0) of designs (iii)-(v) enters the analysis too. H4 is
therefore reported as a secondary, partly confirmatory analysis, not as an independent test.

## 2. Questions
- **H1 (primary).** Does adding the influenza importation pressure to the prompt improve influenza
  forecasts? Estimated as the main effect of importation in the 2x2 factorial design.
- **H2.** Does the benefit depend on the *content* of the importation information, rather than on the
  presence of an extra block of text? Estimated as real vs placebo importation.
- **H3.** Does adding the school calendar, public holidays and weather improve forecasts? This is the
  main effect of calendar & weather, over all six pathogens.
- **H4.** Is the importation effect larger when the importation signal is high or rising than when it
  is low and flat?

## 3. Designs
All designs use the semi-blinded R0 prompt (pathogen names and season given; year, dates and city
withheld; calendar events as day offsets from the origin; importation origins as regions, not
countries). Each design adds or removes whole blocks of the same prompt; everything else is
byte-identical. The prompts of (iii)-(v) were checked to be byte-identical to those already
answered.

| design | id | monthly air passengers | calendar & weather | influenza importation |
|---|---|---|---|---|
| (iii) | `r0_semi` | yes | - | - |
| (iv) | `r0_ctx` | yes | yes | - |
| (vi) | `r0_imponly` | yes | - | real |
| (v) | `r0_imp` | yes | yes | real |
| (vii) | `r0_placebo` | yes | yes | placebo |

- **Calendar & weather.** EDB school calendars, general holidays, and HKO weekly weather: observed for
  the 4 weeks up to the origin; 1991-2020 normals only for the forecast weeks.
- **Importation.** See `Data/context/importation/README.md` for the construction. The prompt shows the
  4 latest weeks published 14 days before the origin.
- **Placebo (vii).** Identical in format, wording, length and value range to (v). The whole importation
  block (all subtypes, coverage, main origin regions, Taiwan) is taken as it would have appeared at
  `origin - s` weeks:
  - `s` is drawn uniformly from 13-39 weeks by
    `numpy.random.default_rng([20260925, YYYYMMDD of the origin])`.
  - The shift goes only into the past, so no information from after the origin enters.
  - The range excludes 48-56 weeks, so the placebo does not match the same season a year earlier.
  - Realised shifts (weeks): 2024-07-06 34, 2024-08-03 22, 2024-08-31 29, 2024-09-28 38,
    2024-10-26 24, 2024-11-23 32, 2024-12-21 38, 2025-01-18 14, 2025-02-15 22, 2025-03-15 16,
    2025-04-12 20, 2025-05-10 37, 2025-06-07 28, 2025-07-05 28, 2025-08-02 23, 2025-08-30 14,
    2025-09-27 24.
  - Agreement between the placebo and the real block, over the latest relative pressure shown:
    - correlation of log pressure within subtype: B -0.47, A(H1N1) -0.43, A(H3N2) -0.23;
    - signal stratum (section 6) agrees in 49% of origin-subtypes.

  The placebo is therefore *mismatched* (it tends to point the wrong way), not neutral. For this
  reason H2 is read together with the effect of the placebo vs no importation, (vii) vs (iv).

## 4. Models, calls and forecasts
- **Models.** Gemma 4 31B and Gemma 4 26B A4B (Google AI Studio), GPT-OSS 20B and Qwen3.8 27B
  (Groq): the four models whose free-tier limits allow the extra calls. The other models of the main
  benchmark keep one call per request and do not enter this experiment.
- **Calls.** Three independent calls per (origin, design, model), with the provider's default sampling
  settings (unchanged from the main benchmark).
  - Parse failures are retried as in the main benchmark.
  - A call that never returns a valid answer leaves that (origin, design, model) incomplete.
- **Forecast.** Each call returns 10th, 50th and 90th percentiles of the R0 multiplier per pathogen and
  week. The three calls form an equal-weight mixture: the 1,000 posterior draws of the fitted
  mechanistic model are split evenly across the calls, and each draw is propagated with multiplier
  paths sampled from its call. Monte Carlo seeds are fixed per (origin, design, model).
- The main benchmark (figures 05-08) continues to use only the first call, so that all 8 models are
  compared on the same footing.
- **Forecast origins.** The 17 origins every 4 weeks from 6 July 2024 to 27 September 2025. Horizons 1-4
  weeks. No origins are added.

## 5. Outcome and effects
- **Outcome.** Weighted interval score (WIS; quantiles 0.05, 0.25, 0.5, 0.75, 0.95) of the weekly
  surveillance indicator.
- **Log WIS.** For a model, design `a` and pathogen `k`, `L[a,k]` = log of the WIS summed over the
  matched targets (origins x horizons).
- **Effect.** An effect is the mean over pathogens of a weighted sum of `L`, reported as a ratio
  `exp(effect)`; below 1 means better forecasts.

| effect | pathogens | contrast | role |
|---|---|---|---|
| importation (main effect) | B, H3, H1 | 1/2 [(vi - iii) + (v - iv)] | **primary (H1)** |
| real vs placebo | B, H3, H1 | (v - vii) | H2 |
| calendar & weather (main effect) | all six | 1/2 [(iv - iii) + (v - vi)] | H3 |
| importation by signal stratum | B, H3, H1 | importation main effect within each stratum | H4 |
| interaction | B, H3, H1 | (v - iv) - (vi - iii) | supporting |
| placebo vs none | B, H3, H1 | (vii - iv) | supporting (reading H2) |
| importation alone / given context | B, H3, H1 | (vi - iii) / (v - iv) | supporting |

- **Analysis set.** For each model, the targets (origin, pathogen, horizon) scored for all five
  designs (complete cases).
- **Per-model intervals.** 95% percentile bootstrap over forecast origins: 4,000 resamples of whole
  origins, keeping all pathogens and horizons of an origin together; seed 20260925.
- **Pooled estimate.** The equal-weight mean of the per-model log effects. Its interval comes from a
  two-level bootstrap:
  - First, origins are resampled. The same draw is used for every model, because the models forecast
    the same targets.
  - Then models are resampled with replacement.

  The interval therefore reflects both origin sampling and between-model heterogeneity (a
  random-effects-type interval) without assuming that the models are independent.
- **Reporting.**
  - We report point estimates, 95% intervals, and `p_better` (the share of bootstrap resamples with a
    ratio below 1).
  - No significance threshold is used. An effect is called clear when its 95% interval excludes 1.
  - H1 is the single primary question; H2-H4 are secondary and are not adjusted for multiplicity.

## 6. Importation signal (H4)
- The signal is defined per origin and influenza subtype, from exactly what the prompt shows: the 4
  latest weeks published 14 days before the origin.
- **High or rising:** latest relative pressure > 1, or latest / oldest of the 4 weeks > 1.3.
- **Low and flat:** otherwise.
- The signal is known at the origin; no outcome data are used.
- Resulting strata: 19 of 51 origin-subtypes are high or rising:
  - A(H1N1) 13 of 17;
  - A(H3N2) 4;
  - B 2.

  The high-or-rising stratum is therefore dominated by A(H1N1).

## 7. Secondary and sensitivity analyses
- Every effect by horizon (1-4 weeks) and by pathogen.
- Every effect with WIS on the log scale, i.e. the WIS of log(1 + counts), as recommended for epidemic
  forecasts (Bosse et al.).
- The same stepwise comparisons with the first call only, for all 8 models, as already computed
  (`table_context_gain.csv`). These are descriptive.

## 8. Completion and interim results
- The analysis is final when every planned call for the four models has an answer (816 new calls plus
  the outstanding first calls).
- Interim figures are regenerated after every round of requests and are labelled interim. A model and
  effect is shown only once it has at least 5 complete origins. No design, model, stratum definition
  or analysis choice will be changed in response to interim results.
- If a model cannot complete, for example because its API is withdrawn, its available origins are
  reported, the pooled estimate uses the models that completed, and the reason is stated here.

## 9. Limitations known in advance
- **Few origins.** 17 origins, and few influenza-active periods, limit precision. With A(H3N2) and B
  largely inactive in the importation data, the importation effect rests mostly on A(H1N1).
- **Placebo not neutral.** The placebo is mismatched rather than neutral (section 3).
- **Revisions.** WHO FluNet data are the current release and may include revisions made after the
  forecast dates.
- **Few, free models.** The four models are free-tier models; results may not generalise to larger
  models.

## 10. Not part of this protocol
The following would be added as dated amendments:
- more forecast origins (every 2 weeks);
- a non-LLM positive control that maps the importation index to R0 multipliers with a rule fitted on
  past origins only.

## Amendments

### Amendment 1 (25 September 2026): corrected and extended data, before any request on them
**Reason.** A data error was found before any result of this experiment was available:
- The previous input file (`Data/HK_ILI_COVID_Sep.csv`) dated every week 7 days early (its date = the
  CHP week end − 7 days). The calendar, weather and importation blocks were therefore one week out of
  step with the case data.
- Its influenza series came from a CHP laboratory table other than the official Flu Express series.

The whole analysis restarts on `Data/HK_weekly_6pathogens.csv` (built from CHP sources by
`Data/build_hk_weekly.py`; formulas and checks in `Data/README_HK_weekly.md`). The runs on the old
file, including the interim answers of this experiment, are archived as a pilot
(`bench/runs/pilot_olddata_20260925`, `output/archive/pilot_olddata_20260925`) and not used.

**Disclosure.** Pilot results on the old data had been seen:
- all designs of the main benchmark;
- the stepwise comparisons (iii)-(v);
- a small number of answers of the experiment.

The designs, models, effects and analysis below were fixed before the pilot of this experiment
produced any result and are not changed.

**Changes.**
1. **Data.** Weeks are dated by their end (Saturday). The pathogen series are as described in
   `Data/README_HK_weekly.md`.
2. **Fitting window.** Each pathogen's fitting start is now determined from data up to the first forecast
   origin only; previously the HFMD start used one later week.
3. **Forecast origins.** There are now 24 origins, every 4 weeks from 2024-07-06 to 2026-04-11. The last
   origin is set by the OAG air-passenger data, which end in March 2026; the prompts show the 6
   complete months before each origin. The context data were extended accordingly:
   - weather to 2026-09;
   - calendar to 2026-08 (EDB 2025/26 example school calendar and 2026 general holidays);
   - importation pressure to 2026-W18 (CNIC reports, WHO FluNet, ImmD arrivals, Taiwan LARS, OAG to
     2026-03).
4. **Prospective period (added, secondary).** Origins from 2025-11-09 on (6 origins: 2025-11-22 to
   2026-04-11) are reported separately from the development period (18 origins):
   - Their forecast targets were not available when the model, the prompts and this protocol were
     specified (the previous data ended on 2025-11-08).
   - They are also likely to lie after the training data of the language models.
5. **Placebo.** The placebo keeps the same rule (a 13-39-week shift into the past, seeded by the origin
   date).
   - Realised shifts for the added origins: 2025-10-25 14, 2025-11-22 22, 2025-12-20 37, 2026-01-17 19,
     2026-02-14 27, 2026-03-14 24, 2026-04-11 33.
   - Agreement with the real block over the 24 origins, on the corrected data:
     - correlation of log relative pressure within subtype: B −0.48, A(H1N1) 0.12, A(H3N2) −0.22;
     - signal stratum agreement 52%.
6. **Importation signal strata (H4).** On the corrected data, 29 of 72 origin-subtypes are high or rising:
   A(H1N1) 13, A(H3N2) 10, B 6.
7. **Request volume.** 24 origins, as follows:
   - main benchmark: 5 designs x 8 models x 1 call = 960 requests;
   - this experiment: 12 further calls per origin for each of the 4 models = 1,152 requests.

### Amendment 2 (26 September 2026): main-benchmark models
- **GLM 5.2 dropped.** OpenRouter withdrew its free version; requests now return "This model is
  unavailable for free". It had no answers on the corrected data.
- **Ling 3.0 Flash Sante added** (inclusionAI, `inclusionai/ling-3.0-flash-sante:free`, OpenRouter free
  tier, reasoning effort low).
- **Second OpenRouter key.** A second key is used once the first key's daily free quota is used up.

The main benchmark therefore has 8 models. The models of this experiment are unchanged (Gemma 4 31B,
Gemma 4 26B A4B, GPT-OSS 20B, Qwen3.8 27B), and each model is served by a single provider throughout.

### Amendment 3 (26 September 2026): providers and refits, before any analysis of the corrected data
- **GPT-OSS 20B moved to NVIDIA NIM.** It is now served by NVIDIA NIM (`openai/gpt-oss-20b`, reasoning
  effort low) instead of Groq, whose free tier allowed only about 40 requests a day. All its requests
  are re-run on NIM; the Groq answers are archived (`runs/main/responses_archive_groq/`) and not used,
  so each model still has a single provider.
- **Refits of poorly converged origins.** Origins with a maximum R-hat above 1.05 were refitted with
  another seed and 8 optimisation starts; the refit is kept when it converges better:

  | origin | before (R-hat / divergent) | after |
  |---|---|---|
  | 2024-10-26 | 1.252 / 297 | 1.018 / 40 |
  | 2025-03-15 | 1.053 / 110 | 1.024 / 104 |
  | 2025-07-05 | 1.072 / 74 | 1.040 / 20 |

  The R0-design answers for these origins, whose prompts contain the fitted estimates, were
  archived (`runs/main/responses_archive_refit/`) and are requested again. 2026-02-14 and
  2026-03-14 are refitted before any request was sent for them.

### Amendment 4 (26 September 2026): providers and models, before any analysis of the corrected data
- **Qwen3.8 27B moved to Alibaba ModelScope** (`Qwen/Qwen3.8-27B`, thinking off, as on Groq). All its
  requests are re-run there; the Groq answers are archived (`runs/main/responses_archive_groq/`).
- **GLM 5.2 restored** through ModelScope (`ZhipuAI/GLM-5.2`, thinking off). The free version on
  OpenRouter had been withdrawn (amendment 2).
- **Two models added to the main benchmark:**
  - DeepSeek V4.1 Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, ModelScope, thinking off);
  - Kimi K3 (`moonshotai/kimi-k3`, NVIDIA NIM, thinking off).
- **Reasoning settings.** Reasoning is off where a provider allows it and set to low where only a
  level can be chosen; Gemini is used at its default. Each model is served by one provider.

The main benchmark now has 11 models. The external-information experiment keeps its 4 models
(Gemma 4 31B, Gemma 4 26B A4B, GPT-OSS 20B, Qwen3.8 27B).

### Amendment 5 (26 September 2026): two more models in this experiment, before any analysis
**Added models.** DeepSeek V4.1 Flash (ModelScope) and Kimi K3 (NVIDIA NIM) join the experiment, with
the same designs, 3 calls per request and analysis. They were chosen because:
- their free-tier throughput lets them finish within the time the main benchmark needs anyway;
- they add two further developers.

**Not added.** GLM 5.2, Nemotron 3 Ultra, Ling 3.0 and the Gemini models were not added: their daily
free quotas would have lengthened the study by several days.

**Consequence.** The experiment now has 6 models from 5 developers, and the pooled estimate averages
over these 6. The choice was made before any result of the experiment on the corrected data had been
examined.

### Amendment 6 (26 September 2026): last refit
**Refits.** The two held-back origins were refitted before any request was sent for them:
- 2026-02-14: 1.099 / 241 → 1.026 / 77 (max R-hat / divergent transitions).
- 2026-03-14: 1.092 / 100 → 1.074 / 153 (first refit) → 1.056 / 135 (second refit, seed 2027, 12
  optimisation starts; minimum ESS 76).

**2026-03-14.** It stays slightly above 1.05 after two refits. The best fit is used and marked in the
sampling-diagnostics figure. As a sensitivity analysis, the main results are repeated without this
origin.

**Final state.** All 24 origins have a maximum R-hat of at most 1.056; 23 of them are at or below
1.05.

### Amendment 7 (27 September 2026): when does the LLM help? Secondary analysis of the main benchmark
**Origin of the question (disclosure).** An interim, exploratory look at the four models with all
origins scored (Gemma 4 31B, Gemma 4 26B A4B, GPT-OSS 20B, Kimi K3; design (iii) vs the mechanistic
model with constant R0) showed ratios that differed by stratum:
- ratio ≈ 1 overall;
- 0.91 in weeks whose *observed* incidence rose at least 1.5-fold over the forecast window;
- 1.06 where incidence fell.

This stratification uses the outcome and cannot be applied in real time. The analysis below is
specified before it is computed. It is secondary and partly confirmatory, because the four models'
data had been seen in the exploratory form.

**Question.** Does the LLM + mechanistic design (iii) forecast better than the same mechanistic model
with constant R0 when the epidemic is growing?

**Strata, known at the forecast origin.** Each (origin, pathogen) is assigned a phase by the fitted
model's effective reproduction number Rt in the last week before the origin (posterior median):
- growing: Rt ≥ 1.05;
- declining: Rt ≤ 0.95;
- stable: otherwise.

**Effect.** For each model, the paired WIS ratio of design (iii) vs constant R0 within each phase:
- per pathogen, summed WIS over matched targets, then the geometric mean over pathogens;
- 95% bootstrap intervals over origins;
- pooled over models with the two-level bootstrap of section 5 (origins shared, then models).

**Primary test of this amendment.** The pooled ratio in the growing phase, on the natural scale.
- **Supported** if its 95% interval lies below 1.
- **Not supported** otherwise; this is reported as such.

**Also reported (descriptive).**
- the same ratio by horizon, by pathogen, on the log scale, for designs (iv) and (v), and by the
  observed-trend strata above;
- directional accuracy: whether the forecast median and the observation move in the same direction
  from the last observed week, up (> ×1.1), down (< ×1/1.1) or flat, for design (iii) and for
  constant R0.

**Models.** All models whose design (iii) has at least 20 scored origins when the main benchmark is
complete.

### Amendment 8 (27 September 2026): historical replication period, 2015-2019
**Status.** Written before any Stan fit, forecast or LLM request for the years before 2023. The data
file was extended back to 2014 on this date (`Data/build_hk_weekly.py`). The only look at those
years was a plot of the raw series (`Data/HK_weekly_6pathogens_2014-2026.png`), made to check the
join between years.

**Origin of the extension (disclosure).**
- The interim amendment-7 estimate, from four complete models, was imprecise: growing phase 0.87
  [0.68, 1.09].
- Most of that growing-phase gain came from three large onsets. Adding origins within 2024-2026
  cannot add onsets.
- The user holds Hong Kong surveillance data from 2014. The decision to extend was therefore taken
  after seeing interim results. To keep it from being a search for significance:
  - every analysis choice below is fixed now;
  - the 2015-2019 period is analysed on its own as the primary test, a replication on data that no
    model has been fitted to;
  - the pooled 2015-2026 estimate is secondary.

**Data.** `HK_weekly_6pathogens.csv`, weeks from 2014-01-04 (same sources and formulas as for
2023-2026).
- COVID-19 does not exist in this period, so five pathogens are analysed.
- 2020-2022 are not used, because of the COVID-19 interventions.
- The main study keeps reading only the weeks from 2023-01-07 (`observed_from`, `data_from`). Its
  inputs are unchanged: 248 weeks x 6 series were checked identical.
  - Five answers requested on 27 September 00:40-00:45 with prompts built from the extended file
    were moved to `runs/main/responses_archive_datachange/` and are requested again with the frozen
    prompts.

**Origins.** The 65 origins on the main study's 4-week grid (anchored at 2025-06-07) from 2015-01-31
to 2019-12-28. The last forecast target is 2020-01-25, before the first school closures. Horizons are
1-4 weeks.

**Mechanistic model.** Same Stan model, priors, fixed periods, knot spacing, sampling settings and
initialisation as the main study. Nothing is tuned on these years. Only the fitting window differs:
- **Window.** Each origin is fitted on a window starting at W0 = max(2014-01-04, origin - 130 weeks),
  close to the longest window of the main study (123 weeks).
- **Start of each pathogen.** A trough, as in the main study: the first week in the first 26 weeks
  of the window whose centred 3-week moving average is within 10% of that range's minimum. Only data
  up to the origin are used.
- **Blocks.** Influenza (B, H3, H1), RSV and HFMD; no COVID block.
- **Poor convergence.** An origin with maximum R-hat > 1.05 or more than 5% divergent transitions is
  refitted once, with seed 2026 and 8 optimisation starts. The refit replaces the original whatever
  its diagnostics. Origins that still fail are dropped in a sensitivity analysis.

**LLM designs and models.**
- **Designs.** (i) direct blind, (ii) direct named, (iii) R0 + mechanistic model, (iv) (iii) +
  calendar & weather, with the same prompts as the main study except that COVID-19 is absent.
  - The calendar gives the EDB example school calendars and general holidays for 2014/15-2019/20.
    *Note added the same day, before any request:* the 2014/15 example school calendar is not
    archived, so January-August 2015 lists general holidays only. `Data/context/add_calendar_2015_2020.py`
    lists the sources.
  - Weather comes from the same HKO series.
  - Air passengers come from OAG 2011-2021.
- **Not run.** Designs (v)-(vii): weekly land arrivals and mainland positivity do not exist in the
  same construction before 2023.
- **Models and calls.** All models of the main benchmark, one call per request.
- **Order of requests.** Main-study requests go first. Historical requests follow in the order (iii),
  (iv), (i), (ii). Within a design, origins are taken in a fixed random order (seed 20260927), so that
  any incomplete set is a random subset of origins, not the earliest years.
- **End of collection.** Collection ends when all requests have answers or on 31 October 2026,
  whichever comes first.
- **Models analysed.** A model enters a design's analysis if at least 52 of the 65 origins are
  scored.

**Primary test of this amendment (A8-1).** The amendment-7 test, replicated in 2015-2019: the
pooled ratio of design (iii) vs constant R0 in the growing phase.
- Phase: Rt ≥ 1.05 in the last week before the origin (posterior median).
- WIS on the natural scale; geometric mean over the five pathogens.
- Two-level bootstrap of section 5, with 4,000 resamples and seed 20260925.
- **Supported** if the 95% interval lies below 1; **not supported** otherwise.

**Secondary analyses (not adjusted for multiplicity).**
- **A8-2.** Design (iii) vs constant R0 in 2015-2019 over all phases, and in each phase.
- **A8-3.** 2015-2026 pooled: the historical and main origins together, over the five pathogens
  common to both periods (COVID-19 excluded). Origins are resampled within each period, then models
  are resampled. Reported overall and by phase.
- **A8-4.** Designs (i), (ii) and (iv) vs constant R0, and the other mechanistic baselines (last-year
  R0, scaled last-year R0, naive), for 2015-2019. Also log-scale WIS, results by horizon and pathogen,
  coverage, and directional accuracy.
- **Contamination check.** The 2015-2019 seasons are in the models' training data.
  - Designs (i), (iii) and (iv) withhold the year and the city; (ii) gives both.
  - The ratio (ii)/(i) is compared between 2015-2019 and 2024-2026. A clearly lower ratio in
    2015-2019 would suggest recall of the past seasons. It would be reported as such, and design (ii)
    of that period would be discounted.

**Limitations known in advance.**
- Pre-2020 influenza epidemics were larger than those of 2023-2026, so the two periods differ in
  intensity.
- The mechanistic model's priors were chosen with 2023-2026 in view.
- The historical fitting windows start at a fixed lag rather than at a common season onset.

### Amendment 9 (27 September 2026): mechanistic forecasts use constant R0 only

**Change.** The mechanistic forecast is the fitted model extrapolated with R0 held at its mean over
the last 7 days of the fit (constant R0). The last-year R0 and scaled last-year R0 variants are removed
from every analysis, table and figure of the main study and of the 2015-2019 replication. A8-4 now
compares designs (i), (ii) and (iv) with constant R0 and naive persistence only.

**Reason.** The fitted R0 trajectory already uses all data before the origin, including past seasons.
Copying last year's R0 path at the forecast stage adds a seasonal rule that is not part of the model,
whose R0(t) is a spline without seasonal structure. For horizons of 1-4 weeks, holding the latest R0
is the usual mechanistic extrapolation, and it remains the reference for all comparisons.

**Unchanged.**
- The Stan program still computes the two variants; they are kept in the raw output and not used.
- The prompts of designs (iii)-(vii) still show the model's R0 in the same weeks one year earlier.
  This is information the LLM may use, and the prompts of the main study are frozen.
- All other analyses, the primary tests and the decision rules.

**What had been seen.** Interim results of the main study, in which the last-year R0 variant had lower
natural-scale error than the constant-R0 reference. This amendment was made after those results were
seen, and is disclosed as such.

### Amendment 10 (29 September 2026): how the LLMs use seasonal information, and a free test of reasoning mode (exploratory)

**What had been seen.** Interim results of the main study, including the amendment-7 finding that the
LLMs' week-4 multipliers tend towards a return to typical levels. The prompts of designs (iii)-(vii)
show the model's R0 for the same weeks one year earlier. No analysis of how the LLMs use that
information had been run, and no answer from the reasoning versions below had been requested.
Both parts are exploratory and are reported as such.

**A10-1. Use of last-year R0.** For every answer of designs (iii)-(v) in the main study whose prompt
showed last-year R0 (origins with at least one year of fitted data), each pathogen k and week h:
- y = log of the LLM's median multiplier (p50);
- x = log(R0 in week h one year earlier / current R0), both as printed in the prompt.
- The slope β of y = α + βx is estimated per model by least squares. 95% interval: bootstrap over
  origins (4,000 resamples, seed 20260929). Pooled: mean of per-model slopes, two-level bootstrap.
- β = 1: the LLM follows last year's pattern; β = 0: it ignores it.
- Also reported by epidemic phase (amendment 7 definition) and by pathogen.

**A10-2. Reasoning test.** Two models, through the same free NVIDIA API and the same model versions:
- GPT-OSS 20B with reasoning effort "high" (main study: "low");
- Kimi K3 with thinking on (main study: off).
- Designs (iii) and (i), the 24 main-study origins, one answer per request.
- Prompts are identical to the main study; only the request parameters differ (maximum output
  raised to 16,000 tokens to leave room for reasoning).
- Run separately (`bench/config_reason.yaml`, results in `bench/runs/reason/`); these answers do not
  enter any main-study estimate, table or figure.
- Comparisons, paired within model on the same targets: WIS ratio reasoning / non-reasoning
  (natural and log scale), relative WIS against constant R0, interval coverage, and β (A10-1).
- Two models only: results are descriptive, with no decision rule. They are reported in the appendix;
  if informative, the Discussion mentions them in one sentence.
- **Update (29 September 2026, before any GPT-OSS request was repeated).** In the first round, 10 of
  30 GPT-OSS answers stopped at the 16,000-token limit while still reasoning, without an answer. The
  limit is raised to 32,000 tokens for GPT-OSS. Answers already received stopped well below 16,000
  tokens and are kept. Kimi K3 (all 48 answers received, none at the limit) is unchanged.

### Amendment 11 (29 September 2026): pre-specified extension of the reasoning test

**What had been seen.** The A10-2 results for GPT-OSS 20B and Kimi K3. With reasoning on, design
(iii) had relative WIS against constant R0 of 0.96 and 0.92 (natural scale; pooled 0.94, 95% interval
0.85-1.09). In growing epidemics the values were 0.94 and 0.85 (0.79-0.99). Direct forecasts improved
more with reasoning but remained overconfident. No answer with reasoning on had been requested from
the models below. This amendment tests the hypothesis raised by A10-2 on new models.

**Models.** Reasoning versions of seven main-study models, with the same model versions and prompts;
only the request parameters change (maximum output 32,000 tokens):
- Nemotron 3 Ultra and Ling 3.0 Flash Sante: reasoning effort "high" (main study: "low"), OpenRouter;
- Qwen3.8 27B, GLM 5.2 and DeepSeek V4.1 Flash: thinking on (main study: off), ModelScope;
- Gemini 2.5 Flash and Gemini 3.8 Flash: reasoning effort "high" (main study: provider default,
  which already includes thinking), Google AI Studio.

**Design.** Design (iii) only, the 24 main-study origins, one answer per request. Results in
`bench/runs/reason_ext/`; they do not enter any main-study estimate.

**Order of requests.** The main study has priority. Requests for a provider start only when all
main-study requests of the models on that provider are answered, because models on one provider share
its free quota.

**Hypotheses** (natural scale, design iii):
- **R1.** Reasoning improves on the same model without it: pooled paired WIS ratio (reasoning / main
  study) below 1.
- **R2.** With reasoning, LLM-guided forecasts beat the mechanistic model with constant R0 in growing
  epidemics (amendment-7 phase definition): pooled relative WIS below 1.
- **R3.** As R2, over all phases.

**Analysis.** Each hypothesis is tested first on the new models alone, then on all reasoning versions
(new models plus the two of A10-2). Pooled estimates use the two-level bootstrap of section 5 (4,000
resamples, seed 20260929). A model enters if at least 20 of the 24 origins are scored. A hypothesis is
supported if the 95% interval lies below 1. Log-scale WIS, coverage and the use of last-year R0
(A10-1) are also reported. The results are reported in the Results section as a pre-specified
secondary analysis; the two A10-2 models are labelled as the exploratory pilot that generated the
hypothesis.
- **Update (29 September 2026, before any answer of this amendment was received).** Order of requests
  changed: the reasoning requests now go first. Until a provider's reasoning requests are answered,
  the main-study and historical requests of the models that share its quota (OpenRouter: Nemotron,
  Ling; ModelScope: Qwen, GLM, DeepSeek; Google: the Gemini model concerned) wait. Models on other
  providers continue as before. This changes only the timing; the design, hypotheses and analysis are
  unchanged. The main-study data lock moves later accordingly.

### Amendment 12 (30 September 2026): longer chains for five historical origins (robustness check)

**What had been seen.** After the pre-specified refit (amendment 8), five of the 14 refitted
historical origins still had a maximum R-hat above 1.05: 2016-01-02 (1.125), 2016-11-05 (1.212),
2016-12-03 (1.144), 2016-12-31 (1.065) and 2017-02-25 (1.08). The problem lies in the influenza block
and, at some origins, the RSV block; divergent transitions stayed below 5%. No forecast from these
refits had been analysed.

**Change.** These five origins are fitted once more with longer chains: 1,000 warm-up and 1,000
sampling iterations per chain (main setting: 500 and 500), seed 2027, eight optimisation starts, and
all other settings unchanged. The fits are written to `output/origins_hist_long/` and do not replace
anything.

**Use.** The main analysis is unchanged: it uses the amendment-8 refits. Two sensitivity analyses of
the 2015-2019 replication are reported:
- **S1 (pre-specified):** without the five origins.
- **S2 (this amendment):** with the longer-chain fits for the five origins. The mechanistic forecasts
  are recomputed from the new posterior, and the LLM-guided forecasts propagate the multipliers already
  given by each LLM through the new posterior draws. No new LLM request is sent.
Convergence of the longer fits is reported whatever it shows.

### Amendment 13 (1 October 2026): answers that could not be read

**What had been seen.** 41 main-study requests had failed all three attempts for several days:
Gemma 4 26B (15, external-information designs), Kimi K3 (10) and Ling 3.0 Flash Sante (16, design i).
Failed answers are cached, so later rounds replayed the same three failures instead of asking again.
The causes were: Gemma wrote reasoning text containing braces before its JSON answer, or left out a
bracket; Kimi returned degenerate output ("!!!…"); Ling reached the 32,768-token output limit while
reasoning, before writing an answer. No forecast exists for these requests.

**Changes** (all models, designs and runs; prompts and request parameters unchanged):
- **Parsing.** If the text from the first to the last brace is not valid JSON, the parser also accepts
  the last complete JSON object that starts with the requested key ("multipliers" or "forecasts").
  On all 5,710 cached answers, the 5,149 that parsed before parse identically; 9 failed answers now
  parse.
- **Retries.** An answer that cannot be parsed is requested up to five more times (six attempts in
  total; previously three). The new attempts are new requests, not cached replays.

### Amendment 14 (1 October 2026): still longer chains and a smaller step size (robustness check)

**What had been seen.** With the longer chains of amendment 12, three historical origins still had a
maximum R-hat above 1.05: 2016-11-05 (1.13), 2016-12-31 (1.079) and 2017-02-25 (1.057). Comparing chain
means showed that the chains disagree mainly on the H3N2 basic reproduction number and susceptible
fraction, whose product (Rt) the data pin down: the posterior is a narrow ridge that the chains cross
slowly, not a set of separate modes. In the main study, one origin (2026-03-14) had R-hat 1.056 (RSV).

**Change.** These four origins are fitted again with 2,000 warm-up and 4,000 sampling iterations per
chain, target acceptance 0.99 (main setting: 0.98) and maximum tree depth 13 (main setting: 12); seed
2028, eight optimisation starts, all else unchanged. Fits are written to `output/origins_hist_longer/`
and `output/origins_longer/` and replace nothing.

**Use** (fixed before any of these fits is run):
- Main analyses unchanged (historical: amendment-8 refits; main study: the amendment-6 fit).
- Historical sensitivity analysis S2 uses, for each of the five origins of amendment 12, the fit with
  the longest chains: this amendment for the three origins above, amendment 12 for 2016-01-02 and
  2016-12-03.
- Main study: a further sensitivity analysis replaces 2026-03-14 with this fit; the pre-specified
  analysis without that origin is also kept.
- The R-hat of every fit of these origins (original, refit, longer, longest) is reported.

### Amendment 15 (13 October 2026): mechanism test with counterfactual prompts

Written before any request of this test was sent.

- Question: do the LLM judgements follow transmission mechanisms, or only statistical patterns of the series?
- Design (config_cf.yaml, results only in runs/cf/): 12 of the 24 main-study origins (every second origin,
  2024-07-06 to 2026-03-14), the 11 main-study models, one request per prompt. Three pairs of prompts; within
  a pair the epidemic state and every other input are identical, and one input is changed:
  1. School calendar (base: design iv): school term throughout the forecast window (cf_term) vs school
     holidays from day +1 to the end of the window (cf_holiday).
  2. Influenza importation (base: design v): the A(H3N2) block set to low and flat levels (weighted positivity
     1% each week, relative pressure 0.20; cf_h3_low) vs high and rising levels (8, 11, 15, 20%, relative
     pressure 2.50; cf_h3_high). A(H1N1), B and all other inputs are the real values.
  3. News (base: design iii): no notable changes (cf_news_none) vs a new immune-escape SARS-CoV-2 variant
     spreading in neighbouring regions (cf_news_variant), with "no notable changes for the other pathogens".
- Outcome: for each model, origin, pathogen and week, the log ratio of the median multipliers of the two
  prompts of a pair (second / first). Pooled as the equal-weight mean over models, 95% interval by the
  two-level bootstrap used elsewhere (origins, then models).
- Expected direction (mechanism) and specificity:
  1. Holidays lower the multipliers of the pathogens transmitted mainly among children (influenza B, A(H3N2),
     A(H1N1), RSV, HFMD) more than that of SARS-CoV-2.
  2. High A(H3N2) importation raises the A(H3N2) multiplier more than those of A(H1N1), B and the
     non-influenza pathogens.
  3. The variant raises the SARS-CoV-2 multiplier more than those of the other pathogens.
- Primary summary per pair: the effect on the target pathogen(s), the effect on the other pathogens and their
  difference (specificity); secondary: the share of model-origin pairs with a change in the expected direction,
  by model and by week.
- The answers are not used for forecasts or for any accuracy estimate; requests of the main study and of the
  2015-2019 study keep priority for shared free quotas only through the normal pacing of each run.
- Note (14 October 2026): OpenRouter withdrew the free endpoint of Ling 3.0 Flash Sante on 13 October (HTTP 404,
  "unavailable for free"); no request of this test had been answered for it. The test therefore uses the other
  10 models; the paid endpoint was not used because it may not be the same model version.
- Extension (14 October 2026, before any request with reasoning was sent): the same prompts with reasoning switched
  on (config_cf_reason.yaml; GPT-OSS 20B and Nemotron 3 Ultra at high effort; Kimi K3, Qwen3.8, GLM 5.2 and DeepSeek
  V4.1 Flash with thinking on; Gemini 2.5 and 3.8 Flash at high effort once their requests of the main test are
  answered). Question: does reasoning change the direction or the specificity of the responses? Analysed in the same
  way; results reported separately from the main test, whose models use the main-study settings.
- Note (18 October 2026): requests of Gemini 3.8 Flash were stopped on 15 October so that its free quota served the
  2015-2019 study; Gemini 2.5 Flash answered the news pair (12 origins) and 9 of 12 school-term prompts before Google
  returned HTTP 404 ("no longer available to new users"). The main test therefore has 9 models for the news pair and 8
  for the calendar and importation pairs. With reasoning on, Kimi K3 (endpoint withdrawn, HTTP 410) and GPT-OSS 20B
  (repeated connection errors) did not complete; the analysis uses the four complete models (DeepSeek V4.1 Flash,
  GLM 5.2, Nemotron 3 Ultra, Qwen3.8 27B). Gemini models were not run with reasoning on.
- Reporting (18 October 2026): Fig. 4E-G and Supplementary Tables 5-6; prompts in Supplementary Note 7.
