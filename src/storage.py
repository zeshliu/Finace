"""缓存和原子 JSON 输出。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)


def sanitize_json(value: Any) -> Any:
    """递归清理 JSON 不支持的 NaN/Infinity 和 numpy/pandas 标量。"""
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """同目录临时文件 + os.replace，避免半写入文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(sanitize_json(payload), ensure_ascii=False, indent=2, cls=EnhancedJSONEncoder, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, target)


def atomic_write_json_bundle(items: dict[str | Path, Any]) -> None:
    """以可回滚方式发布一组 JSON；任一替换失败则恢复全部旧文件。"""
    temporary: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    replaced: set[Path] = set()
    try:
        for raw_target, payload in items.items():
            target = Path(raw_target)
            target.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(sanitize_json(payload), ensure_ascii=False, indent=2, cls=EnhancedJSONEncoder, allow_nan=False)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
                temporary[target] = Path(handle.name)

        for target in temporary:
            if target.exists():
                backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.bak")
                os.replace(target, backup)
                backups[target] = backup
            os.replace(temporary[target], target)
            replaced.add(target)
    except Exception:
        for target in reversed(list(temporary)):
            if target in replaced:
                target.unlink(missing_ok=True)
            backup = backups.get(target)
            if backup and backup.exists():
                os.replace(backup, target)
        raise
    finally:
        for temp in temporary.values():
            temp.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def write_validated_payload(path: str | Path, payload: dict) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError("候选结果必须是包含 candidates 数组的对象")
    atomic_write_json(path, payload)


def archive_payload(source: str | Path, history_dir: str | Path, prefix: str, retention: int = 60) -> None:
    source_path = Path(source)
    if not source_path.exists():
        return
    history = Path(history_dir)
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(source_path, history / f"{prefix}_{stamp}.json")
    old_files = sorted(history.glob(f"{prefix}_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for old in old_files[retention:]:
        old.unlink(missing_ok=True)


class DailyCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, code: str) -> Path:
        safe_code = "".join(ch for ch in str(code) if ch.isdigit()).zfill(6)
        return self.root / f"{safe_code}.csv"

    def load(self, code: str) -> pd.DataFrame:
        path = self.path_for(code)
        if not path.exists():
            return pd.DataFrame()
        try:
            frame = pd.read_csv(path, dtype={"code": str})
            if "date" in frame:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            return frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        except (OSError, ValueError, pd.errors.ParserError):
            return pd.DataFrame()

    def save(self, code: str, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        target = self.path_for(code)
        cleaned = frame.copy().sort_values("date").drop_duplicates("date", keep="last")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", dir=target.parent, delete=False, suffix=".tmp", newline="") as handle:
            cleaned.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
            temp_name = handle.name
        os.replace(temp_name, target)
