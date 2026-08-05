"""收盘后更新普通 A 股账户可交易的 T+0 ETF 候选。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.pipeline import (
    ROOT,
    build_history_index_and_stats,
    build_provider,
    enrich_candidates_with_history_stats,
    load_config,
    now_china,
    update_candidate_pool,
    update_metadata,
)
from src.providers import history_request_range, normalize_code
from src.storage import DailyCache, archive_payload, atomic_write_json_bundle
from src.t0_etf_strategy import classify_t0_etf, scan_t0_etfs

LOGGER = logging.getLogger(__name__)


def run(config_path=None) -> dict:
    config = load_config(config_path)
    now = now_china(config)
    provider = build_provider(config)
    spot_all = provider.get_etf_spot(int(config["data"].get("min_etf_spot_rows", 1000)))
    categories = spot_all.apply(
        lambda row: classify_t0_etf(row.get("name", ""), row.get("fund_type", ""), row.get("code", "")),
        axis=1,
    )
    spot = spot_all[categories.notna()].copy()
    spot["category"] = categories[categories.notna()].values
    spot = spot.drop_duplicates("code", keep="first").reset_index(drop=True)
    if spot.empty:
        raise RuntimeError("未识别到 T+0 ETF，保留上一版数据")

    cache = DailyCache(ROOT / "data" / "cache" / "etf_daily")
    histories: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    lookback = int(config["data"].get("etf_history_lookback_days", config["data"]["history_lookback_days"]))
    adjust = str(config["data"].get("adjust", "qfq"))

    def fetch(code: str):
        old = cache.load(code)
        start_date, end_date = history_request_range(old, now, lookback)
        try:
            fresh = provider.get_etf_history(code, start_date, end_date, adjust)
            combined = pd.concat([old, fresh], ignore_index=True) if not old.empty else fresh
            combined = combined.sort_values("date").drop_duplicates("date", keep="last")
            cache.save(code, combined)
            return code, combined.tail(280).reset_index(drop=True), None
        except Exception as exc:
            if not old.empty:
                return code, old.tail(280).reset_index(drop=True), str(exc)
            return code, pd.DataFrame(), str(exc)

    workers = max(1, int(config["data"].get("etf_max_workers", 4)))
    codes = spot["code"].astype(str).map(normalize_code).tolist()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, code) for code in codes]
        for completed, future in enumerate(as_completed(futures), start=1):
            code, frame, error = future.result()
            if not frame.empty:
                histories[code] = frame
            if error:
                failures.append(f"{code}: {error}")
            if completed % 25 == 0 or completed == len(futures):
                LOGGER.info("ETF 日线进度 %s/%s，可用 %s", completed, len(futures), len(histories))
    provider.close()

    minimum_coverage = float(config["data"].get("min_etf_history_coverage", 0.75))
    if not histories or len(histories) / len(spot) < minimum_coverage:
        raise RuntimeError(f"可用 ETF 日线覆盖率不足 {minimum_coverage:.0%}，保留上一版数据")

    candidates = scan_t0_etfs(spot, histories, config)
    trade_dates = [frame["date"].max() for frame in histories.values() if not frame.empty]
    trade_date = max(trade_dates).strftime("%Y-%m-%d")
    generated_at = now.isoformat(timespec="seconds")
    update_candidate_pool("t0_etf", trade_date, [item["code"] for item in candidates], ROOT / "docs" / "data")
    _, history_stats = build_history_index_and_stats(ROOT / "docs" / "data")
    candidates = enrich_candidates_with_history_stats(candidates, "t0_etf", history_stats, trade_date)

    payload = {
        "strategy": "t0_etf",
        "title": "T+0 ETF",
        "trade_date": trade_date,
        "generated_at": generated_at,
        "score_threshold": config["t0_etf"]["score_threshold"],
        "scanned_stocks": len(histories),
        "eligible_etfs": len(spot),
        "failed_stocks": len(failures),
        "candidates": candidates,
        "disclaimer": "仅筛选规则识别的 T+0 ETF；交易制度与产品属性可能调整，下单前请以券商标识和基金公告为准。",
    }
    output_path = ROOT / "data" / "output" / "t0_etf_latest.json"
    docs_path = ROOT / "docs" / "data" / "t0_etf_latest.json"
    metadata = update_metadata(
        "t0_etf",
        {
            "latest_trade_date": trade_date,
            "last_updated": generated_at,
            "success": True,
            "scanned_stocks": len(histories),
            "eligible_etfs": len(spot),
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
            ROOT / "docs" / "data" / "metadata.json": metadata,
        }
    )
    try:
        archive_payload(docs_path, ROOT / "docs" / "data" / "history", "t0_etf", int(config["site"]["history_retention"]))
        build_history_index_and_stats(ROOT / "docs" / "data")
    except OSError as exc:
        LOGGER.warning("ETF 主结果已发布，但历史归档或索引失败: %s", exc)
    LOGGER.info("T+0 ETF 更新完成：可交易范围 %s，日线可用 %s，候选 %s", len(spot), len(histories), len(candidates))
    return {"payload": payload, "metadata": metadata}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
