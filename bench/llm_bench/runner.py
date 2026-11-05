"""实验主流程：prompt → API（缓存 / 重试 / 预算 / 并发）→ 解析 → 预测分位数 → 写盘

同一个请求（模型、参数、prompt、第几次采样）只会付费一次：结果缓存在 bench/cache/，
中断后重跑会直接读缓存。每次调用都记在 runs/<run>/calls.jsonl。

调用计划（sample_plan）：config 的 arms 每个模型调用 n_samples 次（主结果只用第 1 次，sample 0）；
context_experiment 的组对它列出的模型各调用 context_experiment.n_samples 次（外部信息实验，图 09）。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import persistence_quantiles
from .config import BENCH_DIR, Config, ModelSpec, key_chain
from .data import (OriginBundle, load_calendar, load_importation, load_oag_monthly, load_observed,
                   load_weather)
from .parse import ParseError, extract_json, parse_direct, parse_r0
from .prompts import ARMS, R0_MULT_RANGE, PromptSpec, build_prompt
from .providers import get_provider
from .simulator import MechanisticForecaster, sample_multiplier_paths

CACHE_DIR = BENCH_DIR / "cache"
# 临时性错误（过载、按分钟限流、网络）等一会儿再试；当天额度用完 / 没有余额 / 模型不可用时
# 这个模型本轮熔断（后面的请求直接跳过，下一轮再试）；其余错误只让这一条请求失败
RETRYABLE = {"RateLimitError", "InternalServerError", "APIConnectionError", "APITimeoutError",
             "ServiceUnavailableError", "OverloadedError", "APIStatusError"}
EXHAUST_HINTS = ("perday", "per_day", "per-day", "insufficient_quota", "insufficient balance", "no credits", "credits or licenses",
                 "request too large", "invalid api key", "incorrect api key")
EXHAUST_CLASSES = {"NotFoundError", "PermissionDeniedError", "AuthenticationError"}
QUOTA_HINTS = ("perday", "per_day", "per-day", "insufficient_quota", "nocredits")   # 换一个账号的 key 可能有用


class ModelExhausted(RuntimeError):
    pass


def classify_error(e: Exception) -> tuple[str, float | None]:
    """-> ("retry", wait_s or None) | ("exhaust", None) | ("fail", None)"""
    name, msg = type(e).__name__, str(e)
    low = msg.lower().replace(" ", "")
    if name in EXHAUST_CLASSES or any(h.replace(" ", "") in low for h in EXHAUST_HINTS):
        return "exhaust", None
    if name in RETRYABLE:
        m = re.search(r"retry in ([0-9.]+)s", msg)
        return "retry", (float(m.group(1)) + 2 if m else None)
    return "fail", None
# Stan 的外推方式只用 1 = R0 固定在拟合结束时的值（修正案 9）。Stan 仍会算 2 = 去年同期 R0、3 = 缩放后的
# 去年同期 R0，但那是模型之外另加的季节规则，不作为机制模型的预测，不进入任何评分和结果
STAN_MODES = {1: "stan_const"}
DROPPED_BASELINES = {"stan_lastyear", "stan_scaled"}


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, limit_usd: float | None):
        self.limit = limit_usd
        self.spent = 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self.limit is not None and self.spent >= self.limit:
                raise BudgetExceeded(f"budget {self.limit} USD reached")

    def add(self, usd: float | None) -> None:
        if usd:
            with self._lock:
                self.spent += usd


@dataclass
class Task:
    bundle: OriginBundle
    arm: str
    model: ModelSpec
    sample: int
    prompt: PromptSpec


def runs_root() -> Path:
    """Location for forecasts and analysis outputs (override for offline replay)."""
    return Path(os.environ.get("AI_EPI_RUNS_DIR", str(BENCH_DIR / "runs"))).resolve()


def run_dir(name: str) -> Path:
    d = runs_root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _request(model: ModelSpec, prompt: PromptSpec, sample: int, attempt: int) -> tuple[str, dict]:
    payload = {"provider": model.provider, "model": model.model, "params": model.params,
               "system": prompt.system, "user": prompt.user, "sample": sample, "attempt": attempt}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return key, payload


def experiment(cfg: Config) -> tuple[list[str], list[str], int]:
    """(designs, model ids, calls per request) of the external-information experiment"""
    exp = cfg.get("context_experiment") or {}
    return list(exp.get("arms", [])), list(exp.get("models", [])), int(exp.get("n_samples", 1))


def all_arms(cfg: Config) -> list[str]:
    """main designs, then the designs used only in the external-information experiment"""
    return list(dict.fromkeys(list(cfg["arms"]) + experiment(cfg)[0]))


def sample_plan(cfg: Config, arms: list[str], n_main: int | None = None):
    """-> f(arm, model_id) = number of calls. Main designs: n_samples for every model; designs of the
    external-information experiment: its n_samples for its models (the larger of the two applies)."""
    main = set(cfg["arms"])
    n_main = int(n_main or cfg.get("n_samples", 1))
    ex_arms, ex_models, ex_n = experiment(cfg)

    def f(arm: str, model_id: str) -> int:
        if arm not in arms:
            return 0
        n = n_main if arm in main else 0
        if arm in ex_arms and model_id in ex_models:
            n = max(n, ex_n)
        return n
    return f


class Runner:
    def __init__(self, cfg: Config, run_name: str, models: list[ModelSpec], arms: list[str],
                 n_samples: int | None = None):
        self.cfg = cfg
        self.dir = run_dir(run_name)
        self.models = models
        self.arms = arms
        self.all_bundles: list[OriginBundle] | None = None   # every origin (set by the CLI)
        self.obs = load_observed(cfg)
        self.oag = load_oag_monthly(cfg)
        self.calendar = load_calendar(cfg)
        self.weather = load_weather(cfg)
        self.importation = load_importation(cfg)
        self.budget = Budget(cfg.get("budget_usd"))
        self.n_samples = int(n_samples or cfg.get("n_samples", 1))
        self.samples_for = sample_plan(cfg, arms, self.n_samples)
        conc = int(cfg.get("concurrency_per_model", cfg.get("concurrency_per_provider", 1)))
        self.conc = {m.id: (m.concurrency or conc) for m in models}
        self.sema = {m.id: threading.Semaphore(self.conc[m.id]) for m in models}
        self.pace = {m.id: 60.0 / m.rpm for m in models if m.rpm}      # 两次请求之间的最小间隔（秒）
        self.next_slot = {m.id: 0.0 for m in models}
        self.exhausted: dict[str, str] = {}
        # 每轮最多发新请求的时间（分钟）：到点后不再发新请求（已发出的照常完成），让下一轮尽快开始，
        # 当天额度刚恢复的模型不必等慢模型跑完整轮
        rm = cfg.get("round_minutes")
        self.deadline = time.time() + 60 * float(rm) if rm else None
        # 备用 key 按顺序换用；按模型分别切换（Gemini 的免费额度按项目 x 模型计算，一个模型用完不代表其他模型也用完）
        self.keys = {m.id: key_chain(m.provider) for m in models}
        self.key_pos = {m.id: 0 for m in models}
        self._lock = threading.Lock()
        self._log_lock = threading.Lock()

    # ---------------------------------------------------------------- prompts
    def prompts_for(self, bundle: OriginBundle) -> dict[str, PromptSpec]:
        out = {}
        pdir = self.dir / "prompts" / f"{bundle.origin:%Y-%m-%d}"
        pdir.mkdir(parents=True, exist_ok=True)
        for arm in self.arms:
            p = build_prompt(arm, self.cfg, self.obs, bundle, self.oag, self.calendar, self.weather,
                             self.importation)
            (pdir / f"{arm}.txt").write_text(f"[system]\n{p.system}\n\n[user]\n{p.user}\n", encoding="utf-8")
            out[arm] = p
        return out

    # ---------------------------------------------------------------- calls
    def _log(self, rec: dict) -> None:
        with self._log_lock, open(self.dir / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _call(self, model: ModelSpec, prompt: PromptSpec, sample: int, attempt: int) -> tuple[dict, bool]:
        key, payload = _request(model, prompt, sample, attempt)
        path = CACHE_DIR / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), True
        self.budget.check()
        tries = int(self.cfg.get("api_retries", 6))
        for k in range(tries + 1):
            if model.id in self.exhausted:
                raise ModelExhausted(self.exhausted[model.id])
            keys = self.keys.get(model.id) or [None]
            pos = self.key_pos.get(model.id, 0)
            try:
                with self.sema[model.id]:
                    self._wait_turn(model.id)
                    res = get_provider(model.provider, keys[pos]).complete(model, prompt.system, prompt.user, prompt)
                break
            except Exception as e:
                kind, wait = classify_error(e)
                quota = any(h in str(e).lower().replace(" ", "") for h in QUOTA_HINTS)
                if kind == "exhaust" and quota and pos + 1 < len(keys):
                    with self._lock:                      # this account is used up today: next key
                        if self.key_pos[model.id] == pos:
                            self.key_pos[model.id] = pos + 1
                            self._log({"model": model.id, "status": "key_switch", "to_key": pos + 1})
                    continue
                if kind == "exhaust":
                    with self._lock:
                        self.exhausted.setdefault(model.id, f"{type(e).__name__}: {str(e)[:160]}")
                    raise ModelExhausted(self.exhausted[model.id]) from e
                if kind == "fail" or k == tries:
                    raise
                wait = wait or min(600, 30 * 2 ** k) * random.uniform(0.8, 1.2)
                self._log({"model": model.id, "status": "retry_wait", "wait_s": round(wait),
                           "error": f"{type(e).__name__}: {str(e)[:200]}"})
                time.sleep(wait)
        cost = model.cost(res.input_tokens, res.output_tokens)
        self.budget.add(cost)
        rec = {"key": key, "request": payload, "response": asdict(res), "cost_usd": cost,
               "created_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec, False

    def _wait_turn(self, model_id: str) -> None:
        """Proactive pacing: at most rpm request starts per minute for this model."""
        gap = self.pace.get(model_id)
        if not gap:
            return
        with self._lock:
            start = max(time.time(), self.next_slot[model_id])
            self.next_slot[model_id] = start + gap
        time.sleep(max(0.0, start - time.time()))

    def execute(self, task: Task) -> dict:
        if self.deadline and time.time() > self.deadline:      # 本轮到点：留到下一轮，不写结果文件
            return {"origin": f"{task.bundle.origin:%Y-%m-%d}", "arm": task.arm, "model": task.model.id,
                    "sample": task.sample, "status": "deferred"}
        H, n_q = self.cfg.horizons, len(self.cfg.quantiles)
        retries = int(self.cfg.get("max_parse_retries", 2))
        base = {"origin": f"{task.bundle.origin:%Y-%m-%d}", "arm": task.arm, "model": task.model.id,
                "model_name": task.model.model, "sample": task.sample}
        result = {**base, "status": "failed", "attempts": 0}
        for attempt in range(retries + 1):
            result["attempts"] = attempt + 1
            log = {**base, "attempt": attempt}
            try:
                rec, hit = self._call(task.model, task.prompt, task.sample, attempt)
                resp = rec["response"]
                log.update(cache_hit=hit, input_tokens=resp["input_tokens"],
                           output_tokens=resp["output_tokens"], cost_usd=rec["cost_usd"],
                           latency_s=resp["latency_s"], stop_reason=resp["stop_reason"],
                           model_reported=resp["model_reported"])
                obj = extract_json(resp["text"])
                if task.prompt.kind == "direct":
                    arr, flags = parse_direct(obj, task.prompt.keys, H, n_q)
                    if task.prompt.scale is not None:
                        arr = arr * task.prompt.scale[:, None, None]
                    result.update(status="ok", quantiles=arr.tolist(), flags=flags)
                else:
                    arr, rationale, flags = parse_r0(obj, task.prompt.keys, H, *R0_MULT_RANGE)
                    result.update(status="ok", multipliers=arr.tolist(), rationale=rationale, flags=flags)
                log["status"] = "ok"
                self._log(log)
                break
            except ParseError as e:
                log.update(status="parse_error", error=str(e))
                result["error"] = f"parse: {e}"
                self._log(log)
            except ModelExhausted as e:
                log.update(status="skipped", error=str(e)[:200])
                result.update(status="skipped", error=str(e)[:200])
                self._log(log)
                break
            except BudgetExceeded as e:
                log.update(status="budget", error=str(e))
                result.update(status="budget", error=str(e))
                self._log(log)
                break
            except Exception as e:  # API errors: the SDKs already retried 429/5xx
                log.update(status="api_error", error=f"{type(e).__name__}: {e}")
                result.update(status="api_error", error=f"{type(e).__name__}: {e}")
                self._log(log)
                break
        rdir = self.dir / "responses" / base["origin"] / task.arm
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / f"{task.model.id}_s{task.sample}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        return result

    # ---------------------------------------------------------------- main
    def run(self, bundles: list[OriginBundle], dry_run: bool = False) -> pd.DataFrame | None:
        tasks = []                       # origin by origin, so that paired designs fill up together
        for b in bundles:
            prompts = self.prompts_for(b)
            for arm, p in prompts.items():
                for m in self.models:
                    tasks += [Task(b, arm, m, s, p) for s in range(self.samples_for(arm, m.id))]
        order = self.cfg.get("request_order")
        if order:                        # 历史期（修正案 8）：按设计的优先顺序，设计内按固定随机顺序的起点
            rank = {a: i for i, a in enumerate(order["arms"])}
            origins = sorted({t.bundle.origin for t in tasks})
            perm = np.random.default_rng(int(order["origin_seed"])).permutation(len(origins))
            pos = {o: int(np.where(perm == i)[0][0]) for i, o in enumerate(origins)}
            tasks.sort(key=lambda t: (rank.get(t.arm, len(rank)), pos[t.bundle.origin], t.sample))
        chars = sum(len(t.prompt.system) + len(t.prompt.user) for t in tasks)
        print(f"{len(bundles)} origins, {len(self.arms)} arms, {len(self.models)} models: {len(tasks)} calls "
              f"(main designs x {self.n_samples}, external-information experiment x {experiment(self.cfg)[2]}); "
              f"~{chars / 4 / 1e6:.2f}M input tokens (prompts saved under {self.dir / 'prompts'})")
        if dry_run:
            return None

        # 每个模型一个独立的线程池（大小 = 它的并发数）：慢的或在等限速的模型不会占住别的模型的线程
        results = []
        pools = {m.id: ThreadPoolExecutor(max_workers=self.conc[m.id]) for m in self.models}
        futures = [pools[t.model.id].submit(self.execute, t) for t in tasks]
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if i % 25 == 0 or i == len(tasks):
                    ok = sum(x["status"] == "ok" for x in results)
                    print(f"  {i}/{len(tasks)} done, {ok} ok, spent ${self.budget.spent:.2f}", flush=True)
        finally:
            for pool in pools.values():
                pool.shutdown(wait=True)
        for mid, why in self.exhausted.items():
            print(f"  {mid}: stopped for this round ({why[:120]})")
        status = pd.DataFrame([{k: r.get(k) for k in ("origin", "arm", "model", "sample", "status",
                                                      "attempts", "error")} for r in results])
        status.to_csv(self.dir / "call_status.csv", index=False)
        print(status.groupby(["model", "arm"])["status"].value_counts().unstack(fill_value=0).to_string())

        return self.write_forecasts(self.all_bundles or bundles)

    def write_forecasts(self, bundles: list[OriginBundle]) -> pd.DataFrame:
        """forecasts*.parquet always cover every saved answer and every origin, so that a partial
        run (--arms / --models / --origins) never drops the other designs or models"""
        saved = self._all_saved()
        fc = self.build_forecasts(bundles, saved)
        for name, df in fc.items():
            path = self.dir / ("forecasts.parquet" if name == "main" else f"forecasts_{name}.parquet")
            df.to_parquet(path, index=False)
            print(f"forecasts ({name}): {len(df)} rows -> {path}")
        mult = self.multiplier_table(saved)
        if len(mult):
            mult.to_parquet(self.dir / "r0_multipliers.parquet", index=False)
        return fc["main"]

    # ---------------------------------------------------------------- forecasts
    def _all_saved(self) -> list[dict]:
        """Every successful answer saved for this run (all designs and models)."""
        out = []
        for f in sorted((self.dir / "responses").glob("*/*/*.json")):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if r.get("status") == "ok" and r.get("arm") in ARMS:
                out.append(r)
        return out

    def multiplier_table(self, results: list[dict]) -> pd.DataFrame:
        """Every R0 multiplier answered (all designs, models and calls)"""
        rows = []
        for r in results:
            if "multipliers" not in r:
                continue
            arr = np.array(r["multipliers"])
            for i, s in enumerate(self.cfg.strains):
                for h in range(arr.shape[1]):
                    rows.append((pd.Timestamp(r["origin"]), r["model"], r["arm"], r["sample"], s, h + 1,
                                 *arr[i, h], (r.get("rationale") or {}).get(s)))
        return pd.DataFrame(rows, columns=["origin", "model", "arm", "sample", "strain", "week",
                                           "p10", "p50", "p90", "rationale"])

    def build_forecasts(self, bundles: list[OriginBundle], results: list[dict]) -> dict[str, pd.DataFrame]:
        """main: the designs in config `arms` for every model, first call only (sample 0), plus the
        baselines. context: the designs of the external-information experiment for its models, an
        equal-weight mixture of all its calls, only where every planned call has an answer.
        Monte Carlo draws are seeded per (origin, design, model), so a forecast does not change
        when other answers arrive."""
        cfg, strains, levels, H = self.cfg, self.cfg.strains, self.cfg.quantiles, self.cfg.horizons
        seed = int(cfg.get("seed", 1))

        def rng_for(*key):
            return np.random.default_rng([seed, zlib.crc32("|".join(map(str, key)).encode())])

        ex_arms, ex_models, ex_n = experiment(cfg)
        main_arms = [a for a in ARMS if a in set(cfg["arms"])]
        by_key = defaultdict(list)
        for r in results:
            if r["status"] == "ok":
                by_key[(r["origin"], r["arm"], r["model"])].append(r)
        model_ids = sorted({k[2] for k in by_key})
        rows = {"main": [], "context": []}

        def emit(dest, origin, model, arm, q):          # q: (N, H, Q)
            for i, s in enumerate(strains):
                for h in range(H):
                    target = origin + pd.Timedelta(weeks=h + 1)
                    for j, lev in enumerate(levels):
                        rows[dest].append((origin, target, h + 1, s, model, arm, lev, float(q[i, h, j])))

        def llm_forecast(sim, ok, rng):
            if "quantiles" in ok[0]:
                return np.mean([np.array(r["quantiles"]) for r in ok], axis=0)
            mults = [np.array(r["multipliers"]) for r in ok]              # each (N, H, 3)
            paths = np.empty((sim.n_draws, len(strains), H))
            for j, qm in enumerate(mults):                               # equal-weight mixture
                idx = np.arange(j, sim.n_draws, len(mults))
                paths[idx] = sample_multiplier_paths(qm, len(idx), rng)
            pred = sim.predictive(sim.expected_cases(paths), rng)
            return np.quantile(pred, levels, axis=0).transpose(2, 1, 0)

        for b in bundles:
            origin, okey = b.origin, f"{b.origin:%Y-%m-%d}"
            # Stan's own extrapolations
            sf = b.stan_fc.copy()
            sf["quantile"] = sf["quantile"].round(4)
            sf = sf[sf["quantile"].isin([round(q, 4) for q in levels])]
            for mode, name in STAN_MODES.items():
                part = sf[sf["fc_mode"] == mode]
                for r in part.itertuples():
                    rows["main"].append((origin, r.target_date, int(r.horizon), r.strain, name, "baseline",
                                         float(r.quantile), float(r.value)))
            # naive persistence
            hist = self.obs.loc[:origin]
            rng = rng_for(okey, "naive")
            q = np.stack([persistence_quantiles(hist[s].to_numpy(float), H, levels, rng) for s in strains])
            emit("main", origin, "naive", "baseline", q)
            # mechanistic simulator with multiplier 1 (should reproduce stan_const)
            sim = MechanisticForecaster(b, strains, cfg.get("sim_draws"))
            rng = rng_for(okey, "sim_const")
            ones = np.ones((sim.n_draws, len(strains), H))
            pred = sim.predictive(sim.expected_cases(ones), rng)
            emit("main", origin, "sim_const", "check", np.quantile(pred, levels, axis=0).transpose(2, 1, 0))
            # LLM designs, main benchmark: first call only
            for arm in main_arms:
                for mid in model_ids:
                    ok = [r for r in by_key.get((okey, arm, mid), []) if r["sample"] == 0]
                    if ok:
                        emit("main", origin, mid, arm, llm_forecast(sim, ok, rng_for(okey, arm, mid, "main")))
            # external-information experiment: all planned calls
            for arm in ex_arms:
                for mid in ex_models:
                    ok = sorted((r for r in by_key.get((okey, arm, mid), []) if r["sample"] < ex_n),
                                key=lambda r: r["sample"])
                    if len(ok) == ex_n:
                        emit("context", origin, mid, arm,
                             llm_forecast(sim, ok, rng_for(okey, arm, mid, "context")))
        out = {}
        for name, rr in rows.items():
            fc = pd.DataFrame(rr, columns=["origin", "target_date", "horizon", "strain", "model", "arm",
                                           "quantile", "value"])
            fc["origin"] = pd.to_datetime(fc["origin"])
            fc["target_date"] = pd.to_datetime(fc["target_date"])
            out[name] = fc
        return out


def pending_tasks(run_name: str, bundles: list[OriginBundle], models: list[ModelSpec], arms: list[str],
                  samples_for) -> pd.DataFrame:
    """Tasks that do not yet have a successful response in runs/<run>/responses/.
    samples_for(arm, model_id) -> number of calls (sample_plan)."""
    rows = []
    rdir = runs_root() / run_name / "responses"
    for b in bundles:
        for arm in arms:
            for m in models:
                for smp in range(samples_for(arm, m.id)):
                    f = rdir / f"{b.origin:%Y-%m-%d}" / arm / f"{m.id}_s{smp}.json"
                    ok = f.exists() and json.loads(f.read_text(encoding="utf-8")).get("status") == "ok"
                    if not ok:
                        rows.append((f"{b.origin:%Y-%m-%d}", arm, m.id, smp))
    return pd.DataFrame(rows, columns=["origin", "arm", "model", "sample"])


def load_forecasts(run_name: str) -> pd.DataFrame:
    path = runs_root() / run_name / "forecasts.parquet"
    if not path.exists():
        raise SystemExit(f"{path} 不存在：先运行 python -m llm_bench run")
    fc = pd.read_parquet(path)
    return fc[~fc["model"].isin(DROPPED_BASELINES)]               # 旧文件里可能还有，统一去掉
