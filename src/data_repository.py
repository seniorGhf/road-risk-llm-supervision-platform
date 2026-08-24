from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any


class DataRepository:
    """Load immutable research evidence and expose filtered platform views."""

    def __init__(self, data_file: Path):
        self.data_file = data_file
        self._lock = RLock()
        self._mtime = 0.0
        self._data: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            stat = self.data_file.stat()
            if stat.st_mtime <= self._mtime and self._data:
                return
            self._data = json.loads(self.data_file.read_text(encoding="utf-8"))
            self._mtime = stat.st_mtime

    def snapshot(self) -> dict[str, Any]:
        self.reload()
        with self._lock:
            return deepcopy(self._data)

    def overview(self) -> dict[str, Any]:
        data = self.snapshot()
        return {
            "system": data["system"],
            "snapshot": data["snapshot"],
            "metrics": data["metrics"],
            "pipeline": data["pipeline"],
            "route_series": data["route_series"],
            "watchlist": data["watchlist"],
            "selected_case": data["selected_case"],
            "default_features": data["default_features"],
            "boundaries": data["boundaries"],
        }

    def watchlist(
        self,
        road: str = "all",
        direction: str = "all",
        level: str = "all",
        query: str = "",
    ) -> list[dict[str, Any]]:
        rows = self.snapshot()["watchlist"]
        query = query.strip().lower()

        def matches(row: dict[str, Any]) -> bool:
            if road != "all" and row["road_code"] != road:
                return False
            if direction != "all" and str(row["direction_code"]) != direction:
                return False
            if level != "all" and row["risk_level"] != level:
                return False
            if query:
                haystack = f"{row['segment_id']} {row['road_code']} {row['stake_km']:.3f}".lower()
                if query not in haystack:
                    return False
            return True

        return [row for row in rows if matches(row)]

    def case(self, event_id: str | None = None) -> dict[str, Any]:
        data = self.snapshot()
        if not event_id:
            return data["selected_case"]
        for item in data["cases"]:
            if item["event_id"] == event_id:
                return item
        raise KeyError(f"未找到案例：{event_id}")

    def policies(self) -> dict[str, Any]:
        return self.snapshot()["policy_library"]

    def governance(self) -> dict[str, Any]:
        data = self.snapshot()
        return {
            "model": data["model"],
            "benchmarks": data["benchmarks"],
            "validation": data["validation"],
            "boundaries": data["boundaries"],
            "optimization": data["optimization"],
        }
