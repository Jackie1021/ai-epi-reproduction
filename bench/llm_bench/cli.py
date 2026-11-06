"""命令行入口（在 bench/ 目录下运行）

  python -m llm_bench doctor                    检查密钥 / SDK / 数据 / Stan 导出
  python -m llm_bench models                    列出各家账号可用的模型 ID
  python -m llm_bench validate-sim              Python 模拟器 vs Stan 逐样本对比
  python -m llm_bench run --dry-run             只生成 prompt、统计调用量，不花钱
  python -m llm_bench run [--origins 2025-06-07] [--models claude,gpt] [--arms r0_semi]
  python -m llm_bench forecasts                 用已收回的回答重建预测（不调用 API）
  python -m llm_bench ensemble                  LLM 与季节性机制基线的组合（探索性分析）
  python -m llm_bench score | plot | all | cost
  加 --smoke --mock：用冒烟测试的 Stan 导出和离线 mock 模型跑通全流程（不联网）
"""
from __future__ import annotations

import argparse
import json
import os
from importlib.util import find_spec

import numpy as np
import pandas as pd

from .config import KEY_ENV, MOCK_MODEL, load_config
from .data import list_bundles, load_observed, select_bundles


def _models(cfg, args):
    if args.mock:
        return [MOCK_MODEL]
    models = cfg.models()
    if args.models:
        wanted = set(args.models.split(","))
        models = [m for m in models if m.id in wanted]
    ready = []
    for m in models:
        if not m.enabled:
            continue
        if not m.configured:
            print(f"skip {m.id}: model is FILL_ME in config.yaml")
        elif not m.has_key and not args.dry_run:
            print(f"skip {m.id}: {KEY_ENV[m.provider]} not set in bench/.env")
        else:
            ready.append(m)
    return ready


def _arms(cfg, args):
    from .runner import all_arms

    configured = all_arms(cfg)
    if args.arms is None:
        return configured
    selected = list(dict.fromkeys(arm.strip() for arm in args.arms.split(",") if arm.strip()))
    if not selected:
        raise SystemExit("--arms must name at least one configured experiment arm")
    missing = [arm for arm in selected if arm not in configured]
    if missing:
        raise SystemExit(
            f"Experiment arms not enabled in the configuration: {', '.join(missing)}. "
            "Add them to arms or context_experiment.arms in the YAML before selecting them."
        )
    return selected


def cmd_doctor(cfg, args):
    print("== API keys / SDKs (providers used in config.yaml) ==")
    for provider in dict.fromkeys(m.provider for m in cfg.models()):
        env = KEY_ENV[provider]
        sdk = "anthropic" if provider == "anthropic" else "openai"
        print(f"  {provider:10s} key {'set' if os.environ.get(env) else 'MISSING':8s} "
              f"sdk {sdk} {'ok' if find_spec(sdk) else 'NOT INSTALLED (run bench/setup.sh)'}")
    print("== models in config.yaml ==")
    for m in cfg.models():
        state = ("disabled" if not m.enabled else "ready" if m.configured and m.has_key
                 else "FILL_ME" if not m.configured else "no key")
        print(f"  {m.id:10s} {m.provider:10s} {m.model:28s} {state}")
    print("== data ==")
    for key in ("ili_csv", "oag_xlsx", "calendar_csv"):
        p = cfg.data_path(key)
        print(f"  {key:13s} {'ok' if p and p.exists() else 'missing (optional)' if key == 'calendar_csv' else 'MISSING'}  {p}")
    for smoke in (False, True):
        bundles = list_bundles(cfg, smoke)
        label = "smoke" if smoke else "main"
        if not bundles:
            print(f"  stan origins ({label}): none")
            continue
        worst = max(b.meta["diagnostics"]["max_rhat"] for b in bundles)
        div = sum(b.meta["diagnostics"]["n_divergent"] for b in bundles)
        print(f"  stan origins ({label}): {len(bundles)} "
              f"({bundles[0].origin:%Y-%m-%d} .. {bundles[-1].origin:%Y-%m-%d}), "
              f"worst Rhat {worst:.3f}, divergent total {div}")


def cmd_models(cfg, args):
    from .providers import get_provider
    for provider in dict.fromkeys(m.provider for m in cfg.models()):
        env = KEY_ENV[provider]
        if not os.environ.get(env):
            print(f"== {provider}: no key")
            continue
        try:
            ids = get_provider(provider).list_models()
            print(f"== {provider} ({len(ids)} models)")
            for i in ids:
                print("  ", i)
        except Exception as e:
            print(f"== {provider}: {type(e).__name__}: {e}")


def cmd_validate_sim(cfg, args):
    from .simulator import MechanisticForecaster
    strains = cfg.strains
    for b in select_bundles(cfg, args.origins, args.smoke):
        f = MechanisticForecaster(b, strains)
        mu = f.expected_cases(np.ones((f.n_draws, len(strains), cfg.horizons)))
        st = b.stan_mu_fc[b.stan_mu_fc["fc_mode"] == 1].pivot_table(
            index=["draw", "horizon"], columns="strain", values="mu")
        py = pd.DataFrame(mu.reshape(-1, len(strains)), columns=strains,
                          index=pd.MultiIndex.from_product([f.draw_ids, range(1, cfg.horizons + 1)]))
        st = st.loc[py.index, strains]
        err = float(((py - st).abs() / st.abs().clip(lower=1e-9)).to_numpy().max())
        print(f"{b.origin:%Y-%m-%d}: max relative difference vs Stan mu_fc = {err:.2e} "
              f"{'OK' if err < 1e-6 else 'MISMATCH'}")


def cmd_run(cfg, args):
    from .runner import Runner
    bundles = select_bundles(cfg, args.origins, args.smoke, args.max_origins)
    if not bundles:
        raise SystemExit("没有可用的 Stan 导出：先运行 bash run_export.sh（或 Rscript export_origins.R）")
    models = _models(cfg, args)
    if not models:
        raise SystemExit("没有可调用的模型：检查 config.yaml 的 model 和 bench/.env 的 key")
    runner = Runner(cfg, args.run, models, _arms(cfg, args), n_samples=args.samples)
    runner.all_bundles = select_bundles(cfg, "all", args.smoke, None)
    runner.run(bundles, dry_run=args.dry_run)


def cmd_pending(cfg, args):
    """Print how many requests still lack a successful answer (0 = experiment complete)."""
    from .runner import pending_tasks, sample_plan
    bundles = select_bundles(cfg, args.origins, args.smoke, args.max_origins)
    arms = _arms(cfg, args)
    todo = pending_tasks(args.run, bundles, _models(cfg, args), arms, sample_plan(cfg, arms, args.samples))
    if args.quiet:
        print(len(todo))
        return
    print(f"{len(bundles)} origins with Stan output; {len(todo)} requests still without an answer")
    if len(todo):
        print(todo.groupby(["model", "arm"]).size().unstack(fill_value=0).to_string())


def cmd_forecasts(cfg, args):
    """Rebuild forecasts*.parquet from the saved answers (no API calls)"""
    from .runner import Runner
    runner = Runner(cfg, args.run, [], _arms(cfg, args))
    runner.write_forecasts(select_bundles(cfg, "all", args.smoke, None))


def phase_analysis(cfg, out, scores, min_origins: int = 20):
    """protocol amendment 7: design (iii)-(v) vs the mechanistic model with constant R0, by epidemic phase
    at the origin (fitted Rt), horizon, pathogen and observed trend; plus directional accuracy"""
    from .data import list_bundles
    from .runner import load_forecasts
    from .scoring import direction_accuracy, phase_at_origin, stratified_gain
    obs = load_observed(cfg)
    phase_file = out / "table_phase_at_origin.csv"
    if os.environ.get("AI_EPI_OFFLINE") == "1" and phase_file.exists():
        ph = pd.read_csv(phase_file, parse_dates=["origin"])
    else:
        ph = phase_at_origin(list_bundles(cfg), cfg.strains)
        ph.to_csv(phase_file, index=False)
    k = ["origin", "target_date", "horizon", "strain"]
    base = scores[scores["model"] == cfg.get("reference_model", "stan_const")][k + ["wis", "wis_log"]]
    o = obs.stack().rename("y").reset_index()
    o.columns = ["date", "strain", "y"]
    parts = []
    eligible = {}
    for arm in ("r0_semi", "r0_ctx", "r0_imp"):
        a = scores[scores["arm"] == arm]
        n = a.groupby("model")["origin"].nunique()
        eligible[arm] = sorted(n[n >= min_origins].index)
        a = a[a["model"].isin(eligible[arm])]
        if a["model"].nunique() < 1:
            continue
        d = a[k + ["model", "wis", "wis_log"]].merge(base, on=k, suffixes=("", "_base")) \
            .merge(ph[["origin", "strain", "phase"]], on=["origin", "strain"]) \
            .merge(o.rename(columns={"date": "origin", "y": "y0"}), on=["origin", "strain"]) \
            .merge(o.rename(columns={"date": "target_date", "y": "yt"}), on=["target_date", "strain"])
        r = (d["yt"] + 1) / (d["y0"] + 1)
        d["obs_trend"] = np.select([r >= 1.5, r <= 1 / 1.5], ["rising", "falling"], "flat")
        d["all"] = "all"
        for col, scale in (("wis", "natural"), ("wis_log", "log")):
            for by in ("all", "phase", "horizon", "strain", "obs_trend"):
                parts.append(stratified_gain(d, by, col, col + "_base").assign(arm=arm, scale=scale))
    if parts:
        tab = pd.concat(parts, ignore_index=True)
        tab["level"] = tab["level"].astype(str)
        tab.to_csv(out / "table_llm_gain_by_phase.csv", index=False)
        v = tab[(tab["arm"] == "r0_semi") & (tab["by"].isin(["all", "phase"])) & (tab["model"] == "pooled")]
        print("\nLLM + mechanistic (iii) / mechanistic (constant R0), pooled over", eligible["r0_semi"])
        print(v[["scale", "level", "n_origins", "ratio", "lo95", "hi95", "p_better"]].round(3).to_string(index=False))
    # directional accuracy, design (iii) vs constant R0, same targets
    fc = load_forecasts(args_run_name(out))
    mods = eligible.get("r0_semi", [])
    fc = fc[((fc["arm"] == "r0_semi") & fc["model"].isin(mods)) | (fc["model"] == "stan_const")]
    da = direction_accuracy(fc, obs).merge(ph[["origin", "strain", "phase"]], on=["origin", "strain"])
    ref = da[da["model"] == "stan_const"][k + ["correct"]].rename(columns={"correct": "correct_ref"})
    da = da[da["model"].isin(mods)].merge(ref, on=k)
    if len(da):
        rng = np.random.default_rng(20260927)
        rows = []
        for phase, g in list(da.groupby("phase")) + [("all", da)]:
            origins = np.sort(g["origin"].unique())
            per = g.groupby("origin")[["correct", "correct_ref"]].agg(["sum", "count"])
            cs, cr, cn = per[("correct", "sum")].reindex(origins).to_numpy(), \
                per[("correct_ref", "sum")].reindex(origins).to_numpy(), per[("correct", "count")].reindex(origins).to_numpy()
            idx = rng.integers(0, len(origins), size=(4000, len(origins)))
            diff = (cs[idx].sum(1) - cr[idx].sum(1)) / cn[idx].sum(1)
            rows.append((phase, len(g), cs.sum() / cn.sum(), cr.sum() / cn.sum(), (cs.sum() - cr.sum()) / cn.sum(),
                         *np.percentile(diff, [2.5, 97.5])))
        dt = pd.DataFrame(rows, columns=["phase", "targets", "acc_llm", "acc_constant_R0", "difference", "lo95", "hi95"])
        dt.to_csv(out / "table_direction_accuracy.csv", index=False)
        print("\ndirectional accuracy (design iii vs constant R0):"); print(dt.round(3).to_string(index=False))


def args_run_name(out):
    return out.name


def cmd_score(cfg, args):
    from .runner import load_forecasts, run_dir
    from .scoring import coverage_table, relative_wis, score_forecasts
    out = run_dir(args.run)
    fc = load_forecasts(args.run)
    scores = score_forecasts(fc, load_observed(cfg), cfg.quantiles)
    scores.to_parquet(out / "scores.parquet", index=False)
    ref = cfg.get("reference_model", "stan_const")
    for col, suffix in (("wis", ""), ("wis_log", "_log")):
        rel = relative_wis(scores, ref, col)
        rel.to_csv(out / f"table_relative_wis{suffix}.csv", index=False)
    cov = coverage_table(scores)
    cov.to_csv(out / "table_coverage.csv", index=False)
    # 开发期 vs 前瞻期（起点在 prospective_from 之后：设定模型和 prompt 时还没有这些数据）
    if cfg.get("prospective_from"):
        cut = pd.Timestamp(str(cfg["prospective_from"]))
        parts = []
        for period, sub in (("development", scores[scores["origin"] < cut]), ("prospective", scores[scores["origin"] >= cut])):
            if len(sub):
                for col, scale in (("wis", "natural"), ("wis_log", "log")):
                    parts.append(relative_wis(sub, ref, col).assign(period=period, scale=scale,
                                                                   n_origins=sub["origin"].nunique()))
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / "table_relative_wis_by_period.csv", index=False)
    # 逐级加信息的价值：r0_semi -> r0_ctx（校历、天气；全部病原体）-> r0_imp（输入压力；只看流感）
    from .scoring import paired_gain
    parts = []
    for a, b, label, strains in (("r0_semi", "r0_ctx", "calendar and weather", None),
                                 ("r0_ctx", "r0_imp", "influenza importation pressure", ["B", "H3", "H1"])):
        if {a, b} <= set(scores["arm"]):
            g = paired_gain(scores, a, b, strains=strains)
            g.insert(0, "strain", "all")
            per = []                                       # 分病原体（合并所有时域）
            for s in (strains or cfg.strains):
                x = paired_gain(scores, a, b, strains=[s])
                x = x[x["horizon"].astype(str) == "all"]
                x.insert(0, "strain", s)
                per.append(x)
            g = pd.concat([g] + per, ignore_index=True)
            g.insert(0, "added", label)
            g.insert(1, "base_arm", a)
            g.insert(2, "new_arm", b)
            g.insert(3, "pathogens", "influenza" if strains else "all")
            parts.append(g)
    if parts:
        gain = pd.concat(parts, ignore_index=True)
        gain.to_csv(out / "table_context_gain.csv", index=False)
    # LLM 什么时候有用（protocol 修订 7）：按起点时模型估计的 Rt 分阶段，LLM + 机制模型 vs 固定 R0
    try:
        phase_analysis(cfg, out, scores, min_origins=int(cfg.get("phase_min_origins", 20)))
    except Exception as e:                      # 次要分析失败不影响主结果
        print("phase analysis skipped:", type(e).__name__, e)
    # 外部信息的析因实验（3 次调用混合；bench/protocol_context_experiment.md）
    ctx_path = out / "forecasts_context.parquet"
    if ctx_path.exists():
        from .data import load_importation
        from .prompts import IMPORT_DELAY_DAYS
        from .scoring import context_effects, import_signal
        fc_ctx = pd.read_parquet(ctx_path)
        if len(fc_ctx):
            sc = score_forecasts(fc_ctx, load_observed(cfg), cfg.quantiles)
            sc.to_parquet(out / "scores_context.parquet", index=False)
            imp = load_importation(cfg)
            signal = import_signal(imp, sc["origin"].unique(), IMPORT_DELAY_DAYS) if imp is not None else None
            if signal is not None:
                signal.to_csv(out / "table_importation_signal.csv", index=False)
            eff = context_effects(sc, cfg.strains, signal)
            if len(eff):                           # sensitivity: WIS on the log scale
                eff = pd.concat([eff.assign(scale="natural"),
                                 context_effects(sc, cfg.strains, signal, col="wis_log").assign(scale="log")],
                                ignore_index=True)
                eff.to_csv(out / "table_context_experiment.csv", index=False)
                view = eff[(eff["horizon"].astype(str) == "all") & (eff["pathogens"].isin(["all", "influenza"]))
                           & (eff["scale"] == "natural")]
                print("\nexternal-information experiment (WIS ratio, < 1 = better; all horizons):")
                print(view.pivot_table(index=["effect", "stratum"], columns="model", values="ratio")
                      .round(3).to_string())
    if cfg.get("request_order") and not args.smoke and os.environ.get("AI_EPI_OFFLINE") != "1":  # 历史期（修正案 8）：与主研究合并的分析和图 12
        try:
            _cmd_a8(cfg, args)
        except Exception as e:
            print("amendment 8 analysis skipped:", type(e).__name__, e)
    rel = pd.read_csv(out / "table_relative_wis.csv")
    view = rel[rel["strain"] == "all"].pivot_table(index=["model", "arm"], columns="horizon",
                                                  values="rel_wis")
    print(f"relative WIS vs {ref}, geometric mean over pathogens (lower is better):")
    print(view.round(3).to_string())
    print("\ncoverage:")
    print(cov.round(3).to_string(index=False))


def cmd_ensemble(cfg, args):
    """LLM forecasts combined with the seasonal mechanistic baseline (exploratory, sample 0 only)"""
    from .ensemble import build, summarise
    from .runner import experiment, load_forecasts, run_dir
    out = run_dir(args.run)
    fc = load_forecasts(args.run)
    llm = [m.id for m in cfg.models() if m.enabled]
    sc, weights = build(fc, load_observed(cfg), cfg.quantiles, llm, list(cfg["arms"]),
                        ens_members=experiment(cfg)[1])
    tab = summarise(sc)
    tab.to_csv(out / "table_ensemble.csv", index=False)
    weights.to_csv(out / "table_ensemble_weights.csv", index=False)
    print(tab.round(3).to_string(index=False))


def cmd_plot(cfg, args):
    from .plots import plot_all
    from .runner import run_dir
    out = run_dir(args.run)
    rel = pd.read_csv(out / ("table_relative_wis_log.csv" if args.log else "table_relative_wis.csv"))
    cov = pd.read_csv(out / "table_coverage.csv")
    order = [m.id for m in cfg.models() if m.enabled] + ["mock"]
    labels = {m.id: m.label for m in cfg.models() if m.label}
    note = "log scale" if args.log else ""
    families = {m.id: m.family for m in cfg.models() if m.family}
    gain_path = out / "table_context_experiment.csv"
    gain = pd.read_csv(gain_path) if gain_path.exists() else None
    if gain is not None and "scale" in gain:
        gain = gain[gain["scale"] == "natural"]
    log_path = out / "table_relative_wis_log.csv"
    rel_log = pd.read_csv(log_path) if log_path.exists() and not args.log else None
    from .data import list_bundles, load_calendar, load_importation, load_weather
    from .runner import experiment
    context = {"calendar": load_calendar(cfg), "weather": load_weather(cfg),
               "importation": load_importation(cfg),
               "origins": [b.origin for b in list_bundles(cfg)],
               "experiment_models": experiment(cfg)[1], "experiment_samples": experiment(cfg)[2]}
    if not context["origins"] and os.environ.get("AI_EPI_OFFLINE") == "1":
        forecasts = pd.read_parquet(out / "forecasts.parquet", columns=["origin"])
        context["origins"] = sorted(pd.to_datetime(forecasts["origin"].unique()))
    ph_path, ph0 = out / "table_llm_gain_by_phase.csv", out / "table_phase_at_origin.csv"
    if ph_path.exists() and ph0.exists() and (out / "r0_multipliers.parquet").exists():
        mult = pd.read_parquet(out / "r0_multipliers.parquet")
        mult = mult[(mult["arm"] == "r0_semi") & (mult["sample"] == 0) & (mult["week"] == 4)].copy()
        mult["origin"] = pd.to_datetime(mult["origin"])
        mult = mult.merge(pd.read_csv(ph0, parse_dates=["origin"])[["origin", "strain", "phase"]], on=["origin", "strain"])
        context["phase"], context["mult"] = pd.read_csv(ph_path), mult
    per_path = out / "table_relative_wis_by_period.csv"
    if per_path.exists() and not args.log:
        context["rel_period"] = pd.read_csv(per_path)
    sc_path = out / "scores.parquet"
    if sc_path.exists():                     # Fig. 3: which models gain significantly from grounding (log scale)
        from .scoring import stratified_gain
        sc = pd.read_parquet(sc_path)
        key = ["origin", "target_date", "horizon", "strain", "model"]
        d = sc[sc["arm"] == "r0_semi"][key + ["wis_log"]].merge(
            sc[sc["arm"] == "direct_blind"][key + ["wis_log"]], on=key, suffixes=("", "_base"))
        if len(d):
            d["all"] = "all"
            ext = set(cfg.get("extension_models") or ())        # bootstrap the extension models on their own, so the
            sig = {}                                            # main-study models keep their intervals
            for part in (d[~d["model"].isin(ext)], d[d["model"].isin(ext)]):
                if len(part):
                    g = stratified_gain(part, "all", "wis_log", "wis_log_base")
                    sig.update({r.model: bool(r.hi95 < 1) for r in g.itertuples() if r.model != "pooled"})
            context["scaffold_sig"] = sig
    context["extension_models"] = list(cfg.get("extension_models") or ())
    for p in plot_all(rel, cov, order, cfg.strains, cfg.horizons, out / "figures", note, labels,
                      families=families, gain=gain, rel_log=rel_log, context=context):
        print("wrote", p)
    if args.run == "main" and os.environ.get("AI_EPI_OFFLINE") != "1":
        import subprocess
        subprocess.run(["bash", str(cfg.project_dir / "collect_results.sh")], check=False)


def cmd_cost(cfg, args):
    from .runner import run_dir
    path = run_dir(args.run) / "calls.jsonl"
    if not path.exists():
        raise SystemExit("no calls yet")
    calls = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines()])
    paid = calls[calls.get("cache_hit") == False]  # noqa: E712
    agg = paid.groupby("model").agg(calls=("status", "size"), input_tokens=("input_tokens", "sum"),
                                    output_tokens=("output_tokens", "sum"), usd=("cost_usd", "sum"),
                                    mean_latency_s=("latency_s", "mean"))
    print(agg.round(2).to_string())
    print(calls.groupby(["model", "status"]).size().unstack(fill_value=0).to_string())


def _cmd_a8(cfg, args):
    from .historical import cmd_a8
    cmd_a8(cfg, args)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="llm_bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["doctor", "models", "validate-sim", "run", "score", "plot",
                                        "all", "cost", "pending", "forecasts", "ensemble", "a8"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", default=None, help="run name (default: main, or smoke with --smoke)")
    ap.add_argument("--origins", default=None, help="all | first | last | 2025-06-07,2025-07-05")
    ap.add_argument("--max-origins", type=int, default=None)
    ap.add_argument("--models", default=None, help="comma-separated model ids from config.yaml")
    ap.add_argument("--arms", default=None, help="comma-separated arms (default: config.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and count calls only")
    ap.add_argument("--samples", type=int, default=None, help="override n_samples from config.yaml")
    ap.add_argument("--quiet", action="store_true", help="pending: print only the number")
    ap.add_argument("--smoke", action="store_true", help="use output/origins_smoke")
    ap.add_argument("--mock", action="store_true", help="offline mock model instead of real APIs")
    ap.add_argument("--log", action="store_true", help="plot relative WIS on the log1p scale")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.run is None:
        args.run = "smoke" if args.smoke else cfg.get("run_name", "main")
    commands = {"doctor": cmd_doctor, "models": cmd_models, "validate-sim": cmd_validate_sim,
                "run": cmd_run, "score": cmd_score, "plot": cmd_plot, "cost": cmd_cost,
                "pending": cmd_pending, "forecasts": cmd_forecasts,
                "ensemble": cmd_ensemble, "a8": _cmd_a8}
    if args.command == "all":
        cmd_run(cfg, args)
        if not args.dry_run:
            cmd_score(cfg, args)
            cmd_plot(cfg, args)
    else:
        commands[args.command](cfg, args)
