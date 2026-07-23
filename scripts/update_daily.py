"""每日收盘后更新超跌候选与隔夜历史统计。"""

from __future__ import annotations

import logging

from src.industry import enrich_candidate_industries
from src.overnight_strategy import calculate_gap_statistics
from src.oversold_strategy import scan_oversold
from src.pipeline import ROOT, build_provider, load_config, now_china, preliminary_spot_filter, refresh_histories, update_metadata
from src.storage import DailyCache, archive_payload, atomic_write_json_bundle

LOGGER = logging.getLogger(__name__)


def run(config_path=None) -> dict:
    config = load_config(config_path)
    now = now_china(config)
    provider = build_provider(config)
    spot_all = provider.get_spot(int(config["data"].get("min_spot_rows", 1000)))
    spot = preliminary_spot_filter(spot_all, config)
    if spot.empty:
        raise RuntimeError("基础快照筛选后无股票，保留上一版数据")

    cache = DailyCache(ROOT / "data" / "cache" / "daily")
    histories, failures = refresh_histories(spot, provider, cache, config, now)
    minimum_coverage = float(config["data"].get("min_history_coverage", 0.70))
    if not histories or len(histories) / len(spot) < minimum_coverage:
        raise RuntimeError(f"可用日线覆盖率不足 {minimum_coverage:.0%}，保留上一版数据")

    candidates = scan_oversold(spot, histories, config)
    candidates = enrich_candidate_industries(
        candidates,
        provider,
        ROOT / "data" / "cache" / "industry.json",
        int(config["data"].get("max_workers", 4)),
    )
    provider.close()
    trade_dates = [frame["date"].max() for frame in histories.values() if not frame.empty]
    trade_date = max(trade_dates).strftime("%Y-%m-%d")
    generated_at = now.isoformat(timespec="seconds")
    payload = {
        "strategy": "oversold",
        "title": "超跌反弹初期",
        "trade_date": trade_date,
        "generated_at": generated_at,
        "score_threshold": config["oversold"]["score_threshold"],
        "scanned_stocks": len(histories),
        "failed_stocks": len(failures),
        "candidates": candidates,
        "disclaimer": "仅用于技术研究，不构成投资建议。技术信号可能失效。",
    }

    output_path = ROOT / "data" / "output" / "oversold_latest.json"
    docs_path = ROOT / "docs" / "data" / "oversold_latest.json"
    gap_stats = {}
    for code, history in histories.items():
        stats = calculate_gap_statistics(history, int(config["data"]["overnight_stat_days"]), int(config["data"]["recent_gap_results"]))
        if stats.get("sample_count", 0) >= int(config["overnight"]["min_samples"]):
            gap_stats[code] = stats
    metadata = update_metadata(
        "oversold",
        {
            "latest_trade_date": trade_date,
            "last_updated": generated_at,
            "success": True,
            "scanned_stocks": len(histories),
            "candidate_count": len(candidates),
            "failed_stocks": len(failures),
        },
        config,
        write=False,
    )
    atomic_write_json_bundle(
        {
            output_path: payload,
            docs_path: payload,
            ROOT / "data" / "output" / "overnight_stats.json": {"trade_date": trade_date, "generated_at": generated_at, "stocks": gap_stats},
            ROOT / "docs" / "data" / "metadata.json": metadata,
        }
    )
    try:
        archive_payload(docs_path, ROOT / "docs" / "data" / "history", "oversold", int(config["site"]["history_retention"]))
    except OSError as exc:
        LOGGER.warning("主结果已发布，但历史归档失败: %s", exc)
    LOGGER.info("每日更新完成：扫描 %s，候选 %s，失败/缓存回退 %s", len(histories), len(candidates), len(failures))
    return {"payload": payload, "metadata": metadata}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
