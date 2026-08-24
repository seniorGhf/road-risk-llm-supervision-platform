from __future__ import annotations

from typing import Any


class DispatchOptimizer:
    """Capacity-aware snapshot allocator for operator review.

    The frozen paper-level optimization result remains available in governance data.
    This class powers the interactive snapshot simulation and does not claim causal
    intervention effectiveness.
    """

    def allocate(self, rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        warning_capacity = self._capacity(config, "warning_capacity", 8)
        patrol_capacity = self._capacity(config, "patrol_capacity", 3)
        emergency_capacity = self._capacity(config, "emergency_capacity", 1)
        separation_m = self._capacity(config, "minimum_patrol_separation_m", 1000, maximum=10000)
        scenario = str(config.get("scenario", "loss_cost_ratio_5000"))

        ranked = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
        warnings = ranked[:warning_capacity]
        patrols: list[dict[str, Any]] = []
        for row in ranked:
            if len(patrols) >= patrol_capacity:
                break
            if float(row["score"]) < 0.6:
                continue
            if self._separated(row, patrols, separation_m):
                patrols.append(row)
        emergencies = [row for row in ranked if float(row["score"]) >= 0.8][:emergency_capacity]

        actions: list[dict[str, Any]] = []
        for row in warnings:
            actions.append(self._action(row, "warning", "发布上游预警"))
        for row in patrols:
            actions.append(self._action(row, "patrol", "派出巡查核验"))
        for row in emergencies:
            actions.append(self._action(row, "emergency", "预备应急资源"))

        return {
            "scenario": scenario,
            "capacities": {
                "warning": warning_capacity,
                "patrol": patrol_capacity,
                "emergency": emergency_capacity,
                "minimum_patrol_separation_m": separation_m,
            },
            "counts": {
                "candidate": len(rows),
                "warning": len(warnings),
                "patrol": len(patrols),
                "emergency": len(emergencies),
            },
            "actions": actions,
            "interpretation": "当前结果是单个回放时窗的容量约束建议组合，需由值班人员确认；成本参数是敏感性假设。",
        }

    @staticmethod
    def _capacity(config: dict[str, Any], key: str, default: int, maximum: int = 20) -> int:
        try:
            value = int(config.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须为整数") from exc
        if not 0 <= value <= maximum:
            raise ValueError(f"{key} 必须位于 0 至 {maximum} 之间")
        return value

    @staticmethod
    def _separated(candidate: dict[str, Any], selected: list[dict[str, Any]], distance: int) -> bool:
        for row in selected:
            if row["road_code"] != candidate["road_code"]:
                continue
            if int(row["direction_code"]) != int(candidate["direction_code"]):
                continue
            if abs(float(row["stake_m"]) - float(candidate["stake_m"])) < distance:
                return False
        return True

    @staticmethod
    def _action(row: dict[str, Any], kind: str, label: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "label": label,
            "segment_id": row["segment_id"],
            "road_code": row["road_code"],
            "direction_code": row["direction_code"],
            "stake_km": row["stake_km"],
            "score": row["score"],
            "status": "待确认",
        }

