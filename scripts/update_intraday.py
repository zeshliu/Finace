"""交易日下午更新隔夜高开候选，不重新下载全部历史日线。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.overnight_strategy import calculate_gap_statistics, evaluate_overnight
from src.pipeline import (
    ROOT,
    build_history_index_and_stats,
    build_provider,
    enrich_candidates_with_history_stats,
    load_cached_histories,
    load_config,
    now_china,
    preliminary_spot_filter,
    update_metadata,
)
from src.storage import DailyCache, archive_payload, atomic_write_json_bundle
from src.strategy_utils import passes_basic_filters

LOGGER = logging.getLogger(__name__)


def _historical_priority(stats: dict) -> float:
    return (
        stats.get("high_open_rate", 0) * 30
        + max(-0.01, min(0.02, stats.get("average_open_return", 0))) * 500
        + stats.get("recent_high_open_rate", 0) * 15
        - stats.get("below_minus_1_probability", 1) * 20
    )


def run(config_path=None) -> dict:
    config = load_config(config_path)
    now = now_china(config)
    generated_at = now.isoformat(timespec="seconds")
    provider = build_provider(config)
    spot_all = provider.get_spot(int(config["data"].get("min_spot_rows", 1000)))
    spot = preliminary_spot_filter(spot_all, config)
    cache = DailyCache(ROOT / "data" / "cache" / "daily")
    histories = load_cached_histories(spot["code"].astype(str).tolist(), cache)
    if not histories:
        raise RuntimeError("没有可用日线缓存；请先运行 daily，上一版盘中结果未被覆盖")

    eligible = []
    min_samples = int(config["overnight"]["min_samples"])
    apply_return_limits = bool(config["overnight"].get("basic_return_filter_enabled", False))
    for _, row in spot.iterrows():
        code = str(row["code"]).zfill(6)
        history = histories.get(code)
        if history is None:
            continue
        passed, _ = passes_basic_filters(row, history, config, apply_return_limits=apply_return_limits)
        if not passed:
            continue
        stats = calculate_gap_statistics(history, int(config["data"]["overnight_stat_days"]), int(config["data"]["recent_gap_results"]))
        if stats.get("sample_count", 0) < min_samples:
            continue
        if stats.get("below_minus_1_probability", 1) > float(config["overnight"]["max_low_open_probability"]):
            continue
        eligible.append((_historical_priority(stats), code, row))

    eligible.sort(key=lambda item: item[0], reverse=True)
    shortlist = eligible[: int(config["data"].get("intraday_shortlist", 120))]
    minutes: dict[str, pd.DataFrame] = {}
    failures = []

    def fetch_minute(code: str):
        try:
            return code, provider.get_intraday(code, now.strftime("%Y-%m-%d"), int(config["data"]["intraday_period_minutes"])), None
        except Exception as exc:
            return code, pd.DataFrame(), str(exc)

    with ThreadPoolExecutor(max_workers=max(1, int(config["data"].get("max_workers", 4)))) as executor:
        futures = [executor.submit(fetch_minute, code) for _, code, _ in shortlist]
        for future in as_completed(futures):
            code, frame, error = future.result()
            if not frame.empty:
                minutes[code] = frame
            if error:
                failures.append(f"{code}: {error}")

    minimum_coverage = float(config["data"].get("min_intraday_coverage", 0.50))
    if shortlist and len(minutes) / len(shortlist) < minimum_coverage:
        raise RuntimeError(f"可用分钟线覆盖率不足 {minimum_coverage:.0%}，保留上一版数据")

    candidates = []
    for _, code, row in shortlist:
        if code not in minutes:
            continue
        try:
            item = evaluate_overnight(row, histories[code], minutes[code], config, generated_at)
            if item and item["score"] >= float(config["overnight"]["score_threshold"]):
                candidates.append(item)
        except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
            continue
    candidates.sort(key=lambda item: (-item["score"], item["code"]))

    trade_date = now.strftime("%Y-%m-%d")
    _, history_stats = build_history_index_and_stats(ROOT / "docs" / "data")
    candidates = enrich_candidates_with_history_stats(candidates, "overnight", history_stats, trade_date)

    payload = {
        "strategy": "overnight",
        "title": "隔夜高开候选",
        "trade_date": trade_date,
        "generated_at": generated_at,
        "score_threshold": config["overnight"]["score_threshold"],
        "scanned_stocks": len(eligible),
        "minute_checked_stocks": len(minutes),
        "failed_stocks": len(failures),
        "candidates": candidates,
        "disclaimer": "本板块只提供尾盘观察统计，不显示买入建议；隔夜波动风险较高。",
    }
    output_path = ROOT / "data" / "output" / "overnight_latest.json"
    docs_path = ROOT / "docs" / "data" / "overnight_latest.json"
    metadata = update_metadata(
        "overnight",
        {
            "latest_trade_date": trade_date,
            "last_updated": generated_at,
            "success": True,
            "scanned_stocks": len(eligible),
            "candidate_count": len(candidates),
            "minute_checked_stocks": len(minutes),
            "failed_stocks": len(failures),
        },
        config,
        write=False,
    )
    atomic_write_json_bundle(
        {
            output_path: payload,
            docs_path: payload,
            ROOT / "docs" / "data" / "metadata.json": metadata,
        }
    )
    try:
        archive_payload(docs_path, ROOT / "docs" / "data" / "history", "overnight", int(config["site"]["history_retention"]))
        build_history_index_and_stats(ROOT / "docs" / "data")
    except OSError as exc:
        LOGGER.warning("主结果已发布，但历史归档或索引失败: %s", exc)
    LOGGER.info("盘中更新完成：基础合格 %s，分钟线 %s，候选 %s", len(eligible), len(minutes), len(candidates))
    return {"payload": payload, "metadata": metadata}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
