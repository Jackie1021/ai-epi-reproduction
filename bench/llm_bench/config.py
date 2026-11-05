"""配置：config.yaml + .env（密钥只进环境变量，从不打印或写盘）"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).resolve().parents[1]

KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "modelscope": "MODELSCOPE_API_KEY",
}
# 同一平台的备用 key（另一个账号）：前一个 key 当天额度用完后依次换用；在 bench/.env 中设置
KEY_FALLBACK = {"openrouter": ("OPENROUTER_API_KEY_2",), "gemini": ("GEMINI_API_KEY_2",)}


def key_chain(provider: str) -> list[str]:
    """environment variables holding keys for this provider, primary first, only those that are set"""
    names = [KEY_ENV[provider], *KEY_FALLBACK.get(provider, ())] if provider in KEY_ENV else []
    return [n for n in names if os.environ.get(n)]


@dataclass
class ModelSpec:
    id: str
    provider: str
    model: str
    params: dict = field(default_factory=dict)
    price_in: float | None = None     # USD per 1M input tokens
    price_out: float | None = None    # USD per 1M output tokens
    concurrency: int | None = None    # 这个模型同时发出的请求数上限（覆盖 concurrency_per_model）
    rpm: float | None = None          # 每分钟最多发几个请求（主动控速，避免触发免费档的分钟限额）
    enabled: bool = True
    label: str | None = None          # 图里的名字
    family: str | None = None         # 同一模型家族共用一个颜色（后加的成员用虚线 + 自己的标记区分）

    @property
    def configured(self) -> bool:
        return bool(self.model) and self.model != "FILL_ME"

    @property
    def has_key(self) -> bool:
        env = KEY_ENV.get(self.provider)
        return env is None or bool(os.environ.get(env))

    def cost(self, tokens_in: int | None, tokens_out: int | None) -> float | None:
        if None in (self.price_in, self.price_out, tokens_in, tokens_out):
            return None
        return (tokens_in * self.price_in + tokens_out * self.price_out) / 1e6


MOCK_MODEL = ModelSpec(id="mock", provider="mock", model="mock")


@dataclass
class Config:
    raw: dict
    path: Path

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def project_dir(self) -> Path:
        return (BENCH_DIR.parent / Path(self.raw["project_dir"])).resolve()

    def data_path(self, key: str) -> Path | None:
        rel = self.raw["data"].get(key)
        return self.project_dir / rel if rel else None

    def origins_dir(self, smoke: bool = False) -> Path:
        return self.project_dir / self.raw["origins_smoke_dir" if smoke else "origins_dir"]

    @property
    def strains(self) -> list[str]:
        return list(self.raw["strains"])

    @property
    def quantiles(self) -> list[float]:
        return [float(q) for q in self.raw["score_quantiles"]]

    @property
    def horizons(self) -> int:
        return int(self.raw["horizons"])

    def models(self) -> list[ModelSpec]:
        return [ModelSpec(**m) for m in self.raw["models"]]


def load_env(path: Path = BENCH_DIR / ".env") -> None:
    """Minimal KEY=VALUE loader; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> Config:
    """A config may start with `extends: config.yaml`: it then only lists what differs (dicts are
    merged key by key, everything else is replaced; `null` removes a key)."""
    path = Path(path) if path else BENCH_DIR / "config.yaml"
    if not path.is_absolute() and not path.exists():
        path = BENCH_DIR / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("extends"):
        base = load_config(path.parent / raw.pop("extends")).raw
        raw = {k: v for k, v in _merge(base, raw).items() if v is not None}
    if os.environ.get("AI_EPI_OFFLINE") != "1":
        load_env()
    return Config(raw=raw, path=path)
