"""数据流水线共享能力。"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from .providers import MarketDataProvider, board_for_code, history_request_range, normalize_code
from .storage import DailyCache, atomic_write_json, read_json

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def now_china(config: dict) -> datetime:
    return datetime.now(ZoneInfo(config.get("site", {}).get("timezone", "Asia/Shanghai")))


def build_provider(config: dict) -> MarketDataProvider:
    data = config["data"]
    return MarketDataProvider(
        retries=data.get("request_retries", 3),
        retry_seconds=data.get("retry_seconds", 2),
        timeout=data.get("request_timeout", 15),
        spot_sources=data.get("spot_sources"),
        history_sources=data.get("history_sources"),
        intraday_sources=data.get("intraday_sources"),
        sina_history_interval=data.get("sina_history_interval_seconds", 0.25),
    )


def preliminary_spot_filter(spot: pd.DataFrame, config: dict) -> pd.DataFrame:
    screening = config["screening"]
    result = spot.copy()
    result["code"] = result["code"].map(normalize_code)
    allowed_boards = set(screening.get("allowed_boards") or [])
    if allowed_boards:
        result = result[result["code"].map(board_for_code).isin(allowed_boards)]
    result = result[result["price"].between(float(screening["price_min"]), float(screening["price_max"]), inclusive="both")]
    result = result[~result["name"].astype(str).str.upper().str.replace(" ", "", regex=False).str.match(r"^\*?ST")]
    result = result[result["volume"].fillna(0) > 0]
    max_stocks = int(os.environ.get("A_STOCK_MAX_STOCKS", "0") or 0)
    if max_stocks > 0:
        result = result.head(max_stocks)
    return result.reset_index(drop=True)


def refresh_histories(
    spot: pd.DataFrame,
    provider: MarketDataProvider,
    cache: DailyCache,
    config: dict,
    end: datetime,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """并发刷新日线；单只失败时使用已有缓存，全部失败时由调用方终止发布。"""
    histories: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    lookback = int(config["data"]["history_lookback_days"])
    adjust = str(config["data"].get("adjust", "qfq"))

    def fetch(code: str):
        old = cache.load(code)
        start_date, end_date = history_request_range(old, end, lookback)
        try:
            fresh = provider.get_history(code, start_date, end_date, adjust)
            combined = pd.concat([old, fresh], ignore_index=True) if not old.empty else fresh
            combined = combined.sort_values("date").drop_duplicates("date", keep="last")
            cache.save(code, combined)
            return code, combined.tail(280).reset_index(drop=True), None
        except Exception as exc:
            if not old.empty:
                return code, old.tail(280).reset_index(drop=True), str(exc)
            return code, pd.DataFrame(), str(exc)

    codes = spot["code"].astype(str).tolist()
    workers = max(1, int(config["data"].get("max_workers", 4)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(fetch, code): code for code in codes}
        for completed, future in enumerate(as_completed(future_map), start=1):
            code, frame, error = future.result()
            if not frame.empty:
                histories[code] = frame
            if error:
                failures.append(f"{code}: {error}")
            if completed % 100 == 0 or completed == len(codes):
                LOGGER.info("日线进度 %s/%s，可用 %s", completed, len(codes), len(histories))
    provider.close()
    return histories, failures


def load_cached_histories(codes: list[str], cache: DailyCache) -> dict[str, pd.DataFrame]:
    result = {}
    for code in codes:
        frame = cache.load(code)
        if not frame.empty:
            result[normalize_code(code)] = frame.tail(280).reset_index(drop=True)
    return result


def update_metadata(section: str, section_data: dict, config: dict, write: bool = True) -> dict:
    path = ROOT / "docs" / "data" / "metadata.json"
    current = read_json(path, {}) or {}
    sections = current.get("sections", {})
    sections[section] = section_data
    metadata = {
        "latest_trade_date": section_data["latest_trade_date"],
        "last_updated": section_data["last_updated"],
        "success": True,
        "scanned_stocks": section_data["scanned_stocks"],
        "candidate_count": section_data["candidate_count"],
        "last_job": section,
        "sections": sections,
        "data_sources": [
            "新浪财经（经 AKShare，主源）",
            "腾讯财经/东方财富（经 AKShare，备用）",
            "BaoStock（备用）",
        ],
        "notice": "免费数据可能延迟或短暂不可用；内容仅供技术研究，不构成投资建议。",
    }
    if write:
        atomic_write_json(path, metadata)
    return metadata


def build_history_index_and_stats(docs_data_dir: str | Path | None = None) -> tuple[dict, dict[str, dict[str, set[str]]]]:
    """扫描 docs/data/history/ 目录，生成 history_index.json 并计算各股票上榜天数。"""
    data_dir = Path(docs_data_dir) if docs_data_dir else ROOT / "docs" / "data"
    history_dir = data_dir / "history"
    
    index_data: dict[str, Any] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "oversold": [],
        "overnight": [],
    }
    
    stats: dict[str, dict[str, set[str]]] = {
        "oversold": {},
        "overnight": {},
    }

    if not history_dir.exists():
        history_dir.mkdir(parents=True, exist_ok=True)
        
    for file_path in sorted(history_dir.glob("*.json")):
        if file_path.name.startswith("."):
            continue
        prefix = "oversold" if file_path.name.startswith("oversold_") else ("overnight" if file_path.name.startswith("overnight_") else None)
        if not prefix:
            continue
        data = read_json(file_path)
        if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
            continue
        trade_date = data.get("trade_date")
        generated_at = data.get("generated_at")
        candidates = data.get("candidates", [])
        
        index_data[prefix].append({
            "filename": file_path.name,
            "trade_date": trade_date,
            "generated_at": generated_at,
            "candidate_count": len(candidates),
        })
        
        for cand in candidates:
            code = cand.get("code")
            if not code:
                continue
            if code not in stats[prefix]:
                stats[prefix][code] = set()
            if trade_date:
                stats[prefix][code].add(trade_date)
                
    for prefix in ("oversold", "overnight"):
        index_data[prefix].sort(key=lambda item: item.get("generated_at") or "", reverse=True)
        
    atomic_write_json(data_dir / "history_index.json", index_data)
    return index_data, stats


def enrich_candidates_with_history_stats(
    candidates: list[dict],
    strategy_name: str,
    stats: dict[str, dict[str, set[str]]],
    current_trade_date: str | None = None,
) -> list[dict]:
    """为候选股票注入 selected_days（已上榜天数）和 history_dates（历史日期列表）。"""
    strategy_stats = stats.get(strategy_name, {})
    for cand in candidates:
        code = cand.get("code")
        if not code:
            continue
        dates_set = set(strategy_stats.get(code, set()))
        trade_date = cand.get("trade_date") or current_trade_date
        if trade_date:
            dates_set.add(trade_date)
        cand["selected_days"] = len(dates_set)
        cand["history_dates"] = sorted(list(dates_set), reverse=True)
    return candidates
