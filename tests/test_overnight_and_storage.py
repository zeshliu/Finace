from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.overnight_strategy import calculate_gap_statistics
from src.storage import DailyCache, atomic_write_json, atomic_write_json_bundle, read_json, write_validated_payload


def gap_frame():
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=4),
            "open": [10.0, 11.0, 9.0, 12.0],
            "close": [10.0, 10.0, 10.0, 10.0],
        }
    )


def test_next_open_returns_and_high_open_rate():
    stats = calculate_gap_statistics(gap_frame(), days=60, recent_count=20)
    assert stats["sample_count"] == 3
    assert stats["high_open_rate"] == 2 / 3
    assert np.isclose(stats["average_open_return"], (0.1 - 0.1 + 0.2) / 3)
    assert stats["below_minus_1_probability"] == 1 / 3
    assert np.isclose(stats["max_low_open"], -0.1)


def test_next_open_formula_uses_next_day_open_over_current_close():
    frame = gap_frame()
    frame.loc[0, "close"] = 20
    stats = calculate_gap_statistics(frame)
    assert stats["recent_results"][0]["return_pct"] == -45.0


def test_json_output_is_valid_and_nan_becomes_null(tmp_path):
    path = tmp_path / "result.json"
    payload = {"candidates": [{"score": np.nan}], "ok": np.bool_(True)}
    write_validated_payload(path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {"candidates": [{"score": None}], "ok": True}


def test_invalid_json_payload_is_rejected(tmp_path):
    path = tmp_path / "result.json"
    try:
        write_validated_payload(path, {"items": []})
    except ValueError:
        pass
    else:
        raise AssertionError("无 candidates 数组时应拒绝写入")
    assert not path.exists()


def test_read_json_handles_missing_and_broken_files(tmp_path):
    path = tmp_path / "missing.json"
    assert read_json(path, {"fallback": True}) == {"fallback": True}
    path.write_text("not json", encoding="utf-8")
    assert read_json(path, []) == []


def test_atomic_write_json_replaces_existing_file(tmp_path):
    path = tmp_path / "atomic.json"
    atomic_write_json(path, {"version": 1})
    atomic_write_json(path, {"version": 2})
    assert read_json(path)["version"] == 2


def test_daily_cache_round_trip(tmp_path, history_factory):
    cache = DailyCache(tmp_path / "daily")
    original = history_factory(rows=5)
    cache.save("600000", original)
    loaded = cache.load("600000")
    assert len(loaded) == 5
    assert loaded["code"].iloc[0] == "600000"
    assert loaded["close"].iloc[-1] == original["close"].iloc[-1]


def test_json_bundle_publishes_all_files(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    atomic_write_json(first, {"old": True})
    atomic_write_json_bundle({first: {"version": 2}, second: {"version": 2}})
    assert read_json(first) == {"version": 2}
    assert read_json(second) == {"version": 2}


def test_candidate_pool_tracking_and_day_counts(tmp_path):
    from src.pipeline import build_history_index_and_stats, enrich_candidates_with_history_stats, update_candidate_pool
    data_dir = tmp_path / "data"
    
    update_candidate_pool("oversold", "2026-07-22", ["600000", "000001"], data_dir)
    update_candidate_pool("oversold", "2026-07-23", ["600000"], data_dir)
    update_candidate_pool("oversold", "2026-07-24", ["600000", "000001", "600002"], data_dir)
    
    _, stats = build_history_index_and_stats(data_dir)
    cands = [{"code": "600000"}, {"code": "000001"}, {"code": "600002"}]
    enriched = enrich_candidates_with_history_stats(cands, "oversold", stats, "2026-07-24")
    
    cand_map = {item["code"]: item for item in enriched}
    assert cand_map["600000"]["selected_days"] == 3
    assert cand_map["000001"]["selected_days"] == 2
    assert cand_map["600002"]["selected_days"] == 1

