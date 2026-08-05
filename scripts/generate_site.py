"""检查静态站点与数据文件，缺失时生成安全的空结果。"""

from __future__ import annotations

from datetime import datetime

from src.pipeline import ROOT, build_history_index_and_stats, load_config
from src.storage import atomic_write_json, read_json, write_validated_payload


def run(config_path=None) -> dict:
    config = load_config(config_path)
    data_dir = ROOT / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    build_history_index_and_stats(data_dir)
    defaults = {
        "oversold_latest.json": ("oversold", "超跌反弹初期", config["oversold"]["score_threshold"]),
        "overnight_latest.json": ("overnight", "隔夜高开候选", config["overnight"]["score_threshold"]),
        "t0_etf_latest.json": ("t0_etf", "T+0 ETF", config["t0_etf"]["score_threshold"]),
    }
    for filename, (strategy, title, threshold) in defaults.items():
        path = data_dir / filename
        existing = read_json(path)
        if not isinstance(existing, dict) or not isinstance(existing.get("candidates"), list):
            write_validated_payload(
                path,
                {
                    "strategy": strategy,
                    "title": title,
                    "trade_date": None,
                    "generated_at": None,
                    "score_threshold": threshold,
                    "scanned_stocks": 0,
                    "candidates": [],
                    "disclaimer": "尚未生成行情数据，请先运行对应更新任务。",
                },
            )
    metadata_path = data_dir / "metadata.json"
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        metadata = {
            "latest_trade_date": None,
            "last_updated": None,
            "success": False,
            "scanned_stocks": 0,
            "candidate_count": 0,
            "last_job": None,
            "sections": {},
            "data_sources": ["AKShare", "BaoStock"],
            "notice": "尚未运行更新；网站只用于技术研究，不构成投资建议。",
        }
        atomic_write_json(metadata_path, metadata)

    required = ["index.html", "oversold.html", "overnight.html", "t0-etf.html", "css/style.css", "js/app.js"]
    missing = [relative for relative in required if not (ROOT / "docs" / relative).exists()]
    if missing:
        raise FileNotFoundError(f"静态站点缺少文件: {', '.join(missing)}")
    return {"generated_at": datetime.now().isoformat(timespec="seconds"), "data_files": list(defaults), "metadata": metadata}


if __name__ == "__main__":
    run()

