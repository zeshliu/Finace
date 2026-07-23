"""候选股票行业分类解析与缓存。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .providers import MarketDataProvider, normalize_code, normalize_industry_name
from .storage import atomic_write_json, read_json

LOGGER = logging.getLogger(__name__)


def enrich_candidate_industries(
    candidates: list[dict],
    provider: MarketDataProvider,
    cache_path: str | Path,
    max_workers: int = 4,
) -> list[dict]:
    """为最终候选补充行业；细分行业失败时使用 BaoStock 全市场分类。"""
    if not candidates:
        return candidates

    path = Path(cache_path)
    cached_payload = read_json(path, {}) or {}
    cached = cached_payload.get("industries", cached_payload)
    industries = {
        normalize_code(code): normalize_industry_name(value)
        for code, value in cached.items()
        if normalize_industry_name(value) != "未分类"
    }
    codes = list(dict.fromkeys(normalize_code(item.get("code", "")) for item in candidates))
    pending = [code for code in codes if code not in industries]

    fallback: dict[str, str] = {}
    if pending:
        try:
            fallback = provider.get_industry_map_baostock()
        except Exception as exc:
            LOGGER.warning("BaoStock 行业分类获取失败，将继续尝试细分行业: %s", exc)

        resolved: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            futures = {executor.submit(provider.get_industry, code): code for code in pending}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    resolved[code] = normalize_industry_name(future.result())
                except Exception as exc:
                    fallback_name = normalize_industry_name(fallback.get(code, ""))
                    if fallback_name != "未分类":
                        resolved[code] = fallback_name
                    else:
                        LOGGER.warning("行业分类缺失 %s: %s", code, exc)
        industries.update(resolved)

    for item in candidates:
        item["industry"] = industries.get(normalize_code(item.get("code", "")), "未分类")

    if industries:
        atomic_write_json(path, {"industries": industries})
    return candidates
