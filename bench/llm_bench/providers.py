"""各家 API 的统一调用接口

anthropic                      官方 anthropic SDK（流式 + get_final_message，避免长思考超时）
openai / gemini / xai / deepseek / groq / openrouter  官方 openai SDK；后几家用各自文档给出的 OpenAI 兼容地址
mock                           离线测试：不联网，按简单规则生成合法 JSON
密钥只从环境变量读取（bench/.env），不会出现在日志或缓存里。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import KEY_ENV, ModelSpec

OPENAI_COMPAT_BASE = {
    "openai": None,
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",          # NVIDIA NIM（build.nvidia.com 开发者免费档）
    "modelscope": "https://api-inference.modelscope.cn/v1",   # 阿里魔搭 API-Inference（免费，需实名）
}


@dataclass
class LLMResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_reported: str | None = None
    stop_reason: str | None = None
    latency_s: float | None = None
    request_id: str | None = None


class AnthropicProvider:
    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=4)   # ANTHROPIC_API_KEY from the environment

    def complete(self, spec: ModelSpec, system: str, user: str, prompt=None) -> LLMResult:
        params = {"max_tokens": 64000, **spec.params}
        t0 = time.time()
        with self.client.messages.stream(model=spec.model, system=system,
                                         messages=[{"role": "user", "content": user}],
                                         **params) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResult(text=text, input_tokens=msg.usage.input_tokens,
                         output_tokens=msg.usage.output_tokens, model_reported=msg.model,
                         stop_reason=msg.stop_reason, latency_s=time.time() - t0,
                         request_id=getattr(msg, "_request_id", None))

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list()]


class OpenAICompatProvider:
    """OpenAI uses the Responses API by default; the others use Chat Completions.
    Override per model with params: {api: chat} or {api: responses}."""

    def __init__(self, provider: str, key_env: str | None = None):
        import openai

        kwargs = {"api_key": os.environ.get(key_env or KEY_ENV[provider]), "max_retries": 4, "timeout": 1800.0}
        if OPENAI_COMPAT_BASE[provider]:
            kwargs["base_url"] = OPENAI_COMPAT_BASE[provider]
        self.client = openai.OpenAI(**kwargs)
        self.provider = provider

    def complete(self, spec: ModelSpec, system: str, user: str, prompt=None) -> LLMResult:
        params = dict(spec.params)
        api = params.pop("api", "responses" if self.provider == "openai" else "chat")
        t0 = time.time()
        if api == "responses":
            r = self.client.responses.create(model=spec.model, instructions=system, input=user, **params)
            u = r.usage
            return LLMResult(text=r.output_text or "", input_tokens=getattr(u, "input_tokens", None),
                             output_tokens=getattr(u, "output_tokens", None), model_reported=r.model,
                             stop_reason=r.status, latency_s=time.time() - t0,
                             request_id=getattr(r, "_request_id", None))
        r = self.client.chat.completions.create(
            model=spec.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **params)
        choice, u = r.choices[0], r.usage
        return LLMResult(text=choice.message.content or "",
                         input_tokens=getattr(u, "prompt_tokens", None),
                         output_tokens=getattr(u, "completion_tokens", None), model_reported=r.model,
                         stop_reason=choice.finish_reason, latency_s=time.time() - t0,
                         request_id=getattr(r, "_request_id", None))

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())


class MockProvider:
    """Offline stand-in used by --mock: persistence with widening log-normal intervals, R0
    multipliers around 1. Deterministic per prompt."""

    def complete(self, spec: ModelSpec, system: str, user: str, prompt=None) -> LLMResult:
        from scipy.stats import norm

        seed = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        H = prompt.mock["horizons"]
        if prompt.kind == "direct":
            z = norm.ppf(prompt.mock["levels"])
            fc = {}
            for key, last in zip(prompt.keys, prompt.mock["last"]):
                base = max(float(last), 1.0) * np.exp(rng.normal(0, 0.05))
                fc[key] = {str(h): (base * np.exp(0.3 * np.sqrt(h) * z)).round(2).tolist()
                           for h in range(1, H + 1)}
            out = {"forecasts": fc}
        else:
            out = {"multipliers": {k: {str(h): sorted(np.round(np.exp(rng.normal(0, 0.05))
                                                               * np.array([0.8, 1.0, 1.25]), 3).tolist())
                                       for h in range(1, H + 1)} for k in prompt.keys},
                   "rationale": {k: "mock" for k in prompt.keys}}
        text = json.dumps(out)
        return LLMResult(text=text, input_tokens=(len(system) + len(user)) // 4,
                         output_tokens=len(text) // 4, model_reported="mock",
                         stop_reason="end_turn", latency_s=0.0)

    def list_models(self) -> list[str]:
        return ["mock"]


_PROVIDERS: dict[str, object] = {}
_LOCK = threading.Lock()


def get_provider(name: str, key_env: str | None = None):
    """key_env: which environment variable holds the key (default: the provider's primary key)"""
    with _LOCK:
        cache = (name, key_env)
        if cache not in _PROVIDERS:
            if name == "anthropic":
                _PROVIDERS[cache] = AnthropicProvider()
            elif name in OPENAI_COMPAT_BASE:
                _PROVIDERS[cache] = OpenAICompatProvider(name, key_env)
            elif name == "mock":
                _PROVIDERS[cache] = MockProvider()
            else:
                raise ValueError(f"unknown provider {name}")
        return _PROVIDERS[cache]
