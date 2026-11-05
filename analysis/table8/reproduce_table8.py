#!/usr/bin/env python3
"""Reproduce Supplementary Table 8 from frozen median forecasts and observations."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parent
DEFAULT_INPUT = PACKAGE / "data" / "table8_inputs.csv.gz"
REFERENCE = PACKAGE / "reference" / "table8_reference.csv"
BOOTSTRAP_SEED = 20260927
BOOTSTRAP_REPLICATES = 4000
MAIN_MODELS = {
    "gem38f": "Gemini 3.8 Flash",
    "gem25f": "Gemini 2.5 Flash",
    "gemma31": "Gemma 4 31B",
    "gemma26": "Gemma 4 26B A4B",
    "qwen": "Qwen3.8 27B",
    "glm52": "GLM 5.2",
    "dsv41f": "DeepSeek V4.1 Flash",
    "gptoss20": "GPT-OSS 20B",
    "kimik3": "Kimi K3",
    "nemo3u": "Nemotron 3 Ultra",
    "ling3s": "Ling 3.0 Flash Sante",
}
FRONTIER_MODELS = {"gpt61sol": "GPT-6.1 Sol", "gpt6astra": "GPT-6 Astra"}
COMPLETE_MAIN = tuple(model for model in MAIN_MODELS if model != "ling3s")
COHORTS = {"main": MAIN_MODELS, "frontier": FRONTIER_MODELS}
ARMS = {"direct_blind": "blind", "direct_named": "named", "r0_semi": "grounded"}
MATCH_KEYS = ["origin", "target_date", "horizon", "strain", "model"]
INPUT_FIELDS = ["run", *MATCH_KEYS, "arm", "forecast_median", "observed"]
METRIC_FIELDS = ["n_origins", "n_cells", "ratio", "lo95", "hi95", "p_better"]
OUTPUT_FIELDS = [
    "group", "model", "model_label", "horizon", "comparison",
    "origins_blind", "origins_named", "origins_grounded", *METRIC_FIELDS,
]
FLOAT_FIELDS = {"ratio", "lo95", "hi95", "p_better"}


def read_input(path: Path) -> pd.DataFrame:
    """Load the frozen medians and retain their original row and float order."""
    data = pd.read_csv(
        path,
        parse_dates=["origin", "target_date"],
        date_format="%Y-%m-%d",
        dtype={"horizon": "int64", "forecast_median": "float64", "observed": "float64"},
        float_precision="round_trip",
    )
    if list(data.columns) != INPUT_FIELDS:
        raise ValueError(f"Expected columns {', '.join(INPUT_FIELDS)}")
    if data.empty or data.isna().any().any():
        raise ValueError("The input must be nonempty and contain no missing values")
    if not all(pd.api.types.is_datetime64_any_dtype(data[c]) for c in ("origin", "target_date")):
        raise ValueError("origin and target_date must contain dates in YYYY-MM-DD format")
    if not np.isfinite(data[["forecast_median", "observed"]]).all().all() or (data["observed"] < 0).any():
        raise ValueError("Medians must be finite and observations finite and nonnegative")
    expected = {(run, model, arm) for run, models in COHORTS.items() for model in models for arm in ARMS}
    actual = set(data[["run", "model", "arm"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError("Input runs, models and forecast designs must match the Table 8 cohorts")
    if set(data["horizon"]) != {1, 2, 3, 4}:
        raise ValueError("Expected forecast horizons 1–4 weeks")
    if data.duplicated(["run", *MATCH_KEYS, "arm"]).any():
        raise ValueError("Multiple medians for the same run, forecast target, model and design")
    return data


def mean_log_ratio(grounded: np.ndarray, direct: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Average the pathogen log ratios after summing each design's errors.

    A pathogen enters the mean only when it has matched targets and both total
    errors are positive. Exponentiating the mean gives their geometric ratio.
    """
    ratios = np.log(np.where(grounded > 0, grounded, np.nan)) - np.log(np.where(direct > 0, direct, np.nan))
    return np.nanmean(np.where(present > 0, ratios, np.nan), axis=-1)


def stratified_gain(paired: pd.DataFrame, by: str) -> pd.DataFrame:
    """Compute paired ratios within a horizon or across all four horizons."""
    # Each comparison/grouping starts from the same seed. Within a call, use
    # one stream across levels in their sorted order, including pooled draws.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for level, sub in paired.groupby(by):
        models = sorted(sub["model"].unique())
        origins = np.sort(sub["origin"].unique())
        strains = sorted(sub["strain"].unique())
        model_index = {value: i for i, value in enumerate(models)}
        origin_index = {value: i for i, value in enumerate(origins)}
        strain_index = {value: i for i, value in enumerate(strains)}

        # One cell per model, origin and pathogen. For the all-horizons
        # estimate, a cell sums absolute log errors across its matched horizons.
        grounded = np.zeros((len(models), len(origins), len(strains)))
        direct = np.zeros_like(grounded)
        matched = np.zeros_like(grounded, dtype=bool)
        sums = sub.groupby(["model", "origin", "strain"])[["ae_log", "ae_log_base"]].sum()
        for (model, origin, strain), errors in sums.iterrows():
            cell = (model_index[model], origin_index[origin], strain_index[strain])
            grounded[cell] = errors["ae_log"]
            direct[cell] = errors["ae_log_base"]
            matched[cell] = True

        # Resample forecast origins with replacement. The same origin
        # multiplicities apply to both designs, every pathogen and every model.
        origin_draws = rng.integers(0, len(origins), size=(BOOTSTRAP_REPLICATES, len(origins)))
        origin_weights = np.zeros((BOOTSTRAP_REPLICATES, len(origins)))
        np.add.at(origin_weights, (np.arange(BOOTSTRAP_REPLICATES)[:, None], origin_draws), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            point = mean_log_ratio(grounded.sum(axis=1), direct.sum(axis=1), matched.sum(axis=1))
            bootstrap = mean_log_ratio(
                np.einsum("mok,bo->bmk", grounded, origin_weights),
                np.einsum("mok,bo->bmk", direct, origin_weights),
                np.einsum("mok,bo->bmk", matched.astype(float), origin_weights),
            )
            n_origins = matched.any(axis=2).sum(axis=1)
            for index, model in enumerate(models):
                values = bootstrap[:, index]
                values = values[np.isfinite(values)]
                if not np.isfinite(point[index]) or not len(values):
                    continue
                lo, hi = np.percentile(values, [2.5, 97.5])
                rows.append((
                    level, model, int(n_origins[index]), int(matched[index].sum()),
                    float(np.exp(point[index])), float(np.exp(lo)), float(np.exp(hi)), float((values < 0).mean()),
                ))

            # The pooled estimate weights models equally on the log scale.
            # Its second bootstrap level resamples the models after origins.
            valid = np.isfinite(point)
            if valid.sum() >= 2:
                model_draws = rng.integers(0, len(models), size=(BOOTSTRAP_REPLICATES, len(models)))
                sampled = np.take_along_axis(bootstrap, model_draws, axis=1)
                sampled_valid = np.take_along_axis(np.broadcast_to(valid, bootstrap.shape), model_draws, axis=1)
                pooled = np.nanmean(np.where(sampled_valid, sampled, np.nan), axis=1)
                pooled = pooled[np.isfinite(pooled)]
                lo, hi = np.percentile(pooled, [2.5, 97.5])
                rows.append((
                    level, "pooled", int(n_origins[valid].min()), int(matched[valid].sum()),
                    float(np.exp(point[valid].mean())), float(np.exp(lo)), float(np.exp(hi)),
                    float((pooled < 0).mean()),
                ))
    return pd.DataFrame(rows, columns=["level", "model", *METRIC_FIELDS])


def reproduce(data: pd.DataFrame) -> list[dict]:
    # This table uses absolute error of the median on the log(1 + x) scale.
    # The forecast median is floored at zero before applying the transform.
    medians = data.assign(
        ae_log=np.abs(np.log1p(data["observed"]) - np.log1p(data["forecast_median"].clip(lower=0))),
    )
    origins = medians.groupby(["run", "model", "arm"])["origin"].nunique()
    rows = []
    for comparator in ("direct_blind", "direct_named"):
        comparison = f"grounded_vs_{ARMS[comparator]}"
        for run, labels in COHORTS.items():
            run_medians = medians[medians["run"] == run]
            grounded = run_medians.loc[run_medians["arm"] == "r0_semi", [*MATCH_KEYS, "ae_log"]]
            direct = run_medians.loc[run_medians["arm"] == comparator, [*MATCH_KEYS, "ae_log"]]
            paired = grounded.merge(direct, on=MATCH_KEYS, suffixes=("", "_base"), validate="one_to_one")
            paired["all"] = "all"
            per_model = stratified_gain(paired, "all").set_index("model")
            for model, label in labels.items():
                rows.append({
                    "group": f"{run}_model", "model": model, "model_label": label,
                    "horizon": "all", "comparison": comparison,
                    **{f"origins_{label}": int(origins[run, model, arm]) for arm, label in ARMS.items()},
                    **per_model.loc[model, METRIC_FIELDS].to_dict(),
                })

            if run == "main":
                # Ling is included in the per-model rows. Pool only the ten
                # models with complete main-study coverage for both designs.
                complete = paired[paired["model"].isin(COMPLETE_MAIN)]
                for by, group in (("all", "pooled_complete"), ("horizon", "pooled_by_horizon")):
                    results = stratified_gain(complete, by)
                    for result in results[results["model"] == "pooled"].to_dict("records"):
                        rows.append({
                            "group": group, "model": "pooled", "model_label": "Pooled, 10 complete models",
                            "horizon": "all" if by == "all" else int(result["level"]),
                            "comparison": comparison, **{field: result[field] for field in METRIC_FIELDS},
                        })
    return rows


def verify(rows: list[dict]) -> None:
    """Compare every computed field to the bundled full-precision reference."""
    with REFERENCE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != OUTPUT_FIELDS:
            raise ValueError("Reference CSV has unexpected columns")
        reference_rows = list(reader)
    if len(rows) != len(reference_rows):
        raise ValueError(f"Reference has {len(reference_rows)} rows; computed {len(rows)}")
    for number, (row, reference) in enumerate(zip(rows, reference_rows), start=1):
        for field in OUTPUT_FIELDS:
            actual = row.get(field, "")
            expected = reference[field]
            if field in FLOAT_FIELDS:
                match = math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-14)
            else:
                match = str(actual) == expected
            if not match:
                raise ValueError(f"Reference mismatch in row {number}, {field}: expected {expected}, got {actual}")


def write_csv(out, rows: list[dict]) -> None:
    writer = csv.DictWriter(out, fieldnames=OUTPUT_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="median inputs CSV or CSV.gz (default: packaged data)")
    parser.add_argument("--output", type=Path, help="new results CSV (default: standard output)")
    parser.add_argument("--verify", action="store_true", help="compare results to the bundled full-precision reference")
    args = parser.parse_args()
    try:
        rows = reproduce(read_input(args.input.expanduser()))
        if args.verify:
            verify(rows)
        if args.output is None:
            write_csv(sys.stdout, rows)
        else:
            with args.output.expanduser().open("x", encoding="utf-8", newline="") as out:
                write_csv(out, rows)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(1, f"Unable to reproduce Supplementary Table 8: {exc}\n")
    if args.verify:
        print(f"Verified all {len(rows)} data rows against the bundled reference.", file=sys.stderr)


if __name__ == "__main__":
    main()
