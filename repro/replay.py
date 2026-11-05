"""Recompute forecast evaluation from frozen numeric artifacts, without model calls."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "repro" / "snapshots"
RUNS = ("main", "hist", "frontier", "frontier_hist", "reason", "reason_ext", "main_frontier")
SWAPS = {"main": 20, "hist": 52, "frontier": 20, "frontier_hist": 52}
MECHANISMS = {"cf": {"gem38f", "ling3s"}, "cf_reason": {"kimik3r", "gptoss20r"}, "frontier_cf": set()}


def verify_inputs() -> None:
    manifest = json.loads((SNAPSHOTS / "manifest.json").read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {name}")


def verify_results(output: Path) -> int:
    count = 0
    for expected in sorted((SNAPSHOTS / "expected").rglob("*.csv")):
        name = expected.relative_to(SNAPSHOTS / "expected")
        if name.name == "table_context_experiment_all.csv":
            actual = pd.read_csv(output / name.with_name("table_context_experiment.csv"))
            actual = actual[actual["stratum"] == "all"]
        else:
            actual = pd.read_csv(output / name)
        try:
            pd.testing.assert_frame_equal(
                actual.reset_index(drop=True), pd.read_csv(expected),
                check_dtype=False, rtol=1e-8, atol=1e-10,
            )
        except AssertionError as error:
            raise ValueError(f"Reference mismatch: {name}\n{error}") from error
        count += 1
    return count


def headline_scores(output: Path) -> None:
    """The primary paired comparison uses models eligible for all three designs."""
    from llm_bench.scoring import stratified_gain

    scores = pd.read_parquet(output / "main" / "scores.parquet")
    arms = ["direct_blind", "direct_named", "r0_semi"]
    n = scores[scores["arm"].isin(arms)].groupby(["model", "arm"])["origin"].nunique().unstack(fill_value=0)
    models = n.index[n[arms].ge(20).all(axis=1)]
    selected = scores[scores["model"].isin(models)]
    keys = ["origin", "target_date", "horizon", "strain", "model"]
    outputs = []
    for control in arms[:2]:
        paired = selected[selected["arm"] == "r0_semi"].merge(
            selected[selected["arm"] == control], on=keys, suffixes=("", "_control"), validate="one_to_one"
        ).assign(all="all")
        for loss in ("wis", "wis_log"):
            for grouping in ("all", "horizon"):
                frame = stratified_gain(paired, grouping, loss, f"{loss}_control")
                outputs.append(frame[frame["model"] == "pooled"].assign(control=control, loss=loss))
    pd.concat(outputs, ignore_index=True).to_csv(output / "main" / "table_head_to_head_wis.csv", index=False)


def run(output: Path, figures: bool) -> int:
    os.environ.update(AI_EPI_RUNS_DIR=str(output), AI_EPI_OFFLINE="1", MPLBACKEND="Agg")
    sys.path.insert(0, str(ROOT / "bench"))

    from cf_analysis import analyse as mechanism
    import cf_analysis
    from combine_frontier import combine, keep_main_pooled
    from internal_multiplier_placebo import analyse as swap
    from llm_bench import amendment10, cli, historical, sensitivity
    from llm_bench.config import load_config

    (output / "main_frontier").mkdir()
    for filename in ("forecasts.parquet", "forecasts_context.parquet", "r0_multipliers.parquet"):
        combine(filename)
    shutil.copyfile(output / "main" / "table_phase_at_origin.csv", output / "main_frontier" / "table_phase_at_origin.csv")
    for run_name in RUNS:
        config = load_config("config.yaml" if run_name == "main" else f"config_{run_name}.yaml")
        cli.cmd_score(config, SimpleNamespace(run=run_name, smoke=False))
    keep_main_pooled()
    historical.a8_tables()
    headline_scores(output)

    for run_name, threshold in SWAPS.items():
        swap(run_name, None if run_name == "main" else f"config_{run_name}.yaml", threshold, replay=True)
    for run_name, exclude in MECHANISMS.items():
        cf_analysis.RUN, cf_analysis.EXCLUDE = output / run_name, exclude
        mechanism().to_csv(output / run_name / "table_mechanism.csv", index=False)
    sensitivity.run_sensitivity("main", None, 20)
    sensitivity.run_sensitivity("hist", "config_hist.yaml", 52)
    amendment10.reasoning_comparison().to_csv(output / "reason" / "table_a10_reasoning.csv", index=False)
    amendment10.amendment11()
    amendment10.replay_reasoning_beta()

    if figures:
        from edfig3_hist import figure as retrospective_figure
        from fig4_combined import figure as mechanism_figure

        cli.cmd_plot(load_config("config_main_frontier.yaml"), SimpleNamespace(run="main_frontier", log=False))
        plot_dir = output / "main_frontier" / "figures"
        mechanism_figure(plot_dir / "fig_swap_and_mechanisms")
        retrospective_figure(plot_dir / "fig_retrospective")
    return verify_results(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for replay outputs")
    parser.add_argument("--no-figures", action="store_true", help="only produce and verify tables")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if output.is_relative_to(SNAPSHOTS):
        parser.error("Output must be outside the frozen snapshots")
    verify_inputs()
    shutil.copytree(SNAPSHOTS / "runs", output)
    with (output / "replay.log").open("x", encoding="utf-8") as log, redirect_stdout(log):
        count = run(output, not args.no_figures)
    print(f"Verified {count} evaluation tables. Outputs: {output}")


if __name__ == "__main__":
    main()
