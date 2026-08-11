"""存储、缓存与候选池测试。"""

from __future__ import annotations

import json

import numpy as np
from src.storage import DailyCache, atomic_write_json, atomic_write_json_bundle, read_json, write_validated_payload


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
