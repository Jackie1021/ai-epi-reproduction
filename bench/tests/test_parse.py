import json

import numpy as np
import pytest

from llm_bench.parse import ParseError, extract_json, parse_direct, parse_r0


def test_extract_json_from_fenced_text():
    obj = extract_json('Here you go:\n```json\n{"forecasts": {"A": {}}}\n```\nthanks')
    assert obj == {"forecasts": {"A": {}}}


def test_extract_json_rejects_garbage():
    with pytest.raises(ParseError):
        extract_json("no json here")


def test_parse_direct_sorts_and_accepts_key_variants():
    fc = {"forecasts": {
        "A": {"h1": [3, 2, 1], "2": {"0.5": 5, "0.1": 4, "0.9": 6}},
        "B": [[0, 1, 2], [1, 2, 3]],
    }}
    arr, flags = parse_direct(fc, ["A", "B"], horizons=2, n_q=3)
    assert arr.shape == (2, 2, 3)
    assert np.all(np.diff(arr, axis=-1) >= 0)
    assert arr[0, 1].tolist() == [4, 5, 6]
    assert any("sorted" in f for f in flags)


def test_parse_direct_missing_series():
    with pytest.raises(ParseError):
        parse_direct({"forecasts": {"A": {"1": [1, 2, 3]}}}, ["A", "B"], 1, 3)


def test_parse_r0_clips_and_keeps_rationale():
    obj = json.loads('{"multipliers": {"H3": {"1": [0.1, 1.0, 5.0]}}, "rationale": {"H3": "school"}}')
    arr, rationale, flags = parse_r0(obj, ["H3"], 1)
    assert arr[0, 0].tolist() == [0.3, 1.0, 3.0]
    assert rationale["H3"] == "school"
    assert flags


def test_compact_lists_and_thought_blocks():
    text = '<thought>try {"x": 1} first</thought>\n{"forecasts": {"A": [[1,2,3],[2,3,4]]}}'
    arr, _ = parse_direct(extract_json(text), ["A"], horizons=2, n_q=3)
    assert arr.tolist() == [[[1, 2, 3], [2, 3, 4]]]


def test_error_classification():
    from llm_bench.runner import classify_error

    class RateLimitError(Exception):
        pass

    class InternalServerError(Exception):
        pass

    per_day = RateLimitError("429 quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' retry in 3.2s")
    per_min = RateLimitError("429 quotaId': 'GenerateRequestsPerMinutePerProjectPerModel' Please retry in 12.5s")
    assert classify_error(per_day)[0] == "exhaust"
    assert classify_error(per_min) == ("retry", 14.5)
    assert classify_error(InternalServerError("503 high demand"))[0] == "retry"
    assert classify_error(RateLimitError("You have no credits remaining"))[0] == "exhaust"
    assert classify_error(RateLimitError("429 Rate limit exceeded: free-models-per-day. Add 10 credits"))[0] == "exhaust"
    assert classify_error(RateLimitError("429 Rate limit exceeded: free-models-per-min."))[0] == "retry"
    assert classify_error(ValueError("bad"))[0] == "fail"
