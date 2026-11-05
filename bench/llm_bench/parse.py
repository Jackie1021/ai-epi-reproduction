"""把 LLM 回复解析成数组；格式不对抛 ParseError（上层会重试），小毛病修正后记 flag"""
from __future__ import annotations

import json
import re

import numpy as np


class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    if not text or not text.strip():
        raise ParseError("empty response")
    t = text.strip()
    # 有些模型（如 Gemma、Qwen 思考模式）会先输出 <thought>…</thought> 或 <think>…</think>
    t = re.sub(r"<(thought|think|thinking)>.*?</\1>", "", t, flags=re.S).strip()
    for tag in ("</thought>", "</think>", "</thinking>"):
        if tag in t:
            t = t.rsplit(tag, 1)[1].strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if fence:
        t = fence.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ParseError("no JSON object found")
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        obj = _last_answer_object(t)          # 修正案 13：答案前的推理文字里也有大括号时，取最后一个完整的答案对象
        if obj is None:
            obj = _close_brackets(t[start:end + 1])   # 漏写右方括号（如 "H3":[[..],[..],"H1":...）时补上
        if obj is None:
            raise ParseError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ParseError("top level is not an object")
    return obj


_ANSWER_START = re.compile(r'\{\s*"(multipliers|forecasts)"\s*:')


def _last_answer_object(t: str):
    """The last complete JSON object that starts with the requested key ("multipliers" or "forecasts").
    Used only when the text from the first to the last brace is not valid JSON."""
    dec = json.JSONDecoder()
    for m in reversed(list(_ANSWER_START.finditer(t))):
        try:
            obj, _ = dec.raw_decode(t, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _close_brackets(t: str):
    """Repair arrays that were left unclosed before the next key or the closing brace, e.g.
    '"H3":[[1,2,3],[1,2,3],"H1":...' or '...[1,2,3]}}'. A ']' is inserted wherever a key starts, or a '}'
    closes, while a '[' is still open. Used only when the text does not parse as JSON; the result must
    still pass the shape checks of parse_r0 / parse_direct."""
    out, stack, i, n = [], [], 0, len(t)
    while i < n:
        c = t[i]
        if c == '"':                                   # copy a string literal; a key closes open arrays first
            j = i + 1
            while j < n and t[j] != '"':
                j += 2 if t[j] == "\\" else 1
            k = j + 1
            while k < n and t[k] in " \t\r\n":
                k += 1
            if k < n and t[k] == ":" and stack and stack[-1] == "[":
                while stack and stack[-1] == "[":      # drop the comma we just wrote, close, then re-add it
                    if out and out[-1].strip() == ",":
                        comma = out.pop()
                        out.append("]")
                        out.append(comma)
                    else:
                        out.append("]")
                    stack.pop()
            out.append(t[i:j + 1])
            i = j + 1
            continue
        if c in "[{":
            stack.append(c)
        elif c == "}":
            while stack and stack[-1] == "[":
                out.append("]")
                stack.pop()
            if stack:
                stack.pop()
        elif c == "]":
            if stack and stack[-1] == "[":
                stack.pop()
        out.append(c)
        i += 1
    try:
        obj = json.loads("".join(out))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _horizon(d, h: int):
    if isinstance(d, list) and len(d) >= h and isinstance(d[h - 1], (list, dict)):
        return d[h - 1]
    if isinstance(d, dict):
        for k in (str(h), h, f"h{h}", f"H{h}", f"w{h}", f"week{h}", f"week_{h}"):
            if k in d:
                return d[k]
    raise ParseError(f"horizon {h} missing")


def _vector(v, n: int) -> np.ndarray:
    if isinstance(v, dict):                               # {"0.025": x, ...}
        try:
            v = [x for _, x in sorted(((float(k), x) for k, x in v.items()), key=lambda kv: kv[0])]
        except ValueError as e:
            raise ParseError("quantile keys must be numbers") from e
    try:
        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError) as e:
        raise ParseError(f"non-numeric values: {v!r:.80}") from e
    if arr.shape != (n,):
        raise ParseError(f"expected {n} numbers, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ParseError("non-finite values")
    return arr


def parse_direct(obj: dict, keys: list[str], horizons: int, n_q: int) -> tuple[np.ndarray, list[str]]:
    """-> (len(keys), horizons, n_q) quantiles in the prompt's units."""
    fc = obj.get("forecasts", obj)
    if not isinstance(fc, dict):
        raise ParseError("missing 'forecasts' object")
    out = np.empty((len(keys), horizons, n_q))
    flags = []
    for i, k in enumerate(keys):
        if k not in fc:
            raise ParseError(f"series {k} missing")
        for h in range(1, horizons + 1):
            arr = _vector(_horizon(fc[k], h), n_q)
            if np.any(np.diff(arr) < 0):
                flags.append(f"{k}/h{h}: quantiles sorted")
                arr = np.sort(arr)
            if np.any(arr < 0):
                flags.append(f"{k}/h{h}: negatives clipped")
                arr = np.maximum(arr, 0.0)
            out[i, h - 1] = arr
    return out, flags


def parse_r0(obj: dict, keys: list[str], horizons: int,
             lo: float = 0.3, hi: float = 3.0) -> tuple[np.ndarray, dict, list[str]]:
    """-> ((len(keys), horizons, 3) multipliers p10/p50/p90, rationale dict, flags)."""
    m = obj.get("multipliers")
    if not isinstance(m, dict):
        raise ParseError("missing 'multipliers' object")
    out = np.empty((len(keys), horizons, 3))
    flags = []
    for i, k in enumerate(keys):
        if k not in m:
            raise ParseError(f"pathogen {k} missing")
        for h in range(1, horizons + 1):
            arr = _vector(_horizon(m[k], h), 3)
            if np.any(np.diff(arr) < 0):
                flags.append(f"{k}/w{h}: sorted")
                arr = np.sort(arr)
            if np.any((arr < lo) | (arr > hi)):
                flags.append(f"{k}/w{h}: clipped to [{lo}, {hi}]")
                arr = np.clip(arr, lo, hi)
            out[i, h - 1] = arr
    rationale = obj.get("rationale") if isinstance(obj.get("rationale"), dict) else {}
    return out, rationale, flags
