from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any


class AuditLog:
    """Append-only local operator audit log."""

    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = runtime_dir / "audit_log.jsonl"
        self._lock = RLock()

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.log_file.exists():
            return []
        with self._lock:
            lines = self.log_file.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(records))

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "")).strip()
        if action not in {"确认", "驳回", "转人工复核", "备注"}:
            raise ValueError("action 必须为确认、驳回、转人工复核或备注")
        record = {
            "id": datetime.now().strftime("AUD-%Y%m%d%H%M%S%f"),
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "operator": str(payload.get("operator", "值班员")).strip()[:30] or "值班员",
            "action": action,
            "target": str(payload.get("target", "未指定对象")).strip()[:100],
            "note": str(payload.get("note", "")).strip()[:500],
            "source": str(payload.get("source", "平台操作"))[:50],
        }
        with self._lock:
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

