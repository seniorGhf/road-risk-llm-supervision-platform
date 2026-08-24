from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class RiskEngine:
    """Serve the frozen distilled student with explicit input validation."""

    def __init__(self, model_dir: Path, default_features: dict[str, float]):
        self.model_dir = model_dir
        self.default_features = {k: float(v) for k, v in default_features.items()}
        manifest_path = model_dir / "confirmation_model_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.features: list[str] = self.manifest["student_features"]
        self.threshold = float(self.manifest["thresholds"]["distilled_student"])
        self._model = None
        self._calibrator = None
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._model is not None or self._load_error:
            return
        try:
            import joblib

            self._model = joblib.load(self.model_dir / "distilled_student.joblib")
            calibrator = self.model_dir / "site_level_probability_calibrator.joblib"
            if calibrator.exists():
                self._calibrator = joblib.load(calibrator)
        except Exception as exc:  # fail closed and surface a readable diagnostic
            self._load_error = f"模型加载失败：{exc}"

    @property
    def status(self) -> dict[str, Any]:
        self._load()
        return {
            "ready": self._model is not None,
            "model": "22 特征 LightGBM 蒸馏学生模型",
            "threshold": self.threshold,
            "error": self._load_error,
        }

    @staticmethod
    def _bounded_number(name: str, value: Any, lower: float, upper: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须为数值") from exc
        if not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"{name} 必须位于 {lower} 至 {upper} 之间")
        return number

    def normalize(self, payload: dict[str, Any]) -> dict[str, float]:
        values = dict(self.default_features)
        for name in self.features:
            if name in payload:
                values[name] = payload[name]

        probability_fields = [
            "v4_model_accident_probability",
            "current_risk_probability",
            "collision_risk_value",
            "risk_lag_5m",
            "risk_lag_30m",
            "risk_mean_past_30m",
            "upstream_mean_300m",
            "upstream_max_300m",
            "downstream_mean_300m",
            "downstream_max_300m",
            "neighborhood_mean_500m",
            "neighborhood_max_500m",
            "neighborhood_std_500m",
        ]
        for name in probability_fields:
            values[name] = self._bounded_number(name, values[name], 0.0, 1.0)
        for name in ["risk_change_5m", "risk_change_30m", "spatial_gradient_600m", "center_neighbor_contrast"]:
            values[name] = self._bounded_number(name, values[name], -1.0, 1.0)
        values["high_risk_cells_500m"] = self._bounded_number(
            "high_risk_cells_500m", values["high_risk_cells_500m"], 0.0, 11.0
        )
        values["direction_code"] = self._bounded_number("direction_code", values["direction_code"], 0.0, 1.0)
        values["road_is_s5"] = self._bounded_number("road_is_s5", values["road_is_s5"], 0.0, 1.0)
        values["hour_sin"] = self._bounded_number("hour_sin", values["hour_sin"], -1.0, 1.0)
        values["hour_cos"] = self._bounded_number("hour_cos", values["hour_cos"], -1.0, 1.0)
        return {name: float(values[name]) for name in self.features}

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._load()
        if self._model is None:
            raise RuntimeError(self._load_error or "模型尚未就绪")
        values = self.normalize(payload)
        row = [[values[name] for name in self.features]]
        score = float(self._model.predict(row)[0])
        score = min(1.0, max(0.0, score))
        probability = None
        if self._calibrator is not None:
            probability = float(self._calibrator.predict_proba([[score]])[0, 1])

        alert = score >= self.threshold
        if score >= 0.8:
            level = "高"
        elif alert:
            level = "中"
        else:
            level = "低"

        evidence = self._evidence(values)
        missing_context = []
        if values["risk_mean_past_30m"] == 0:
            missing_context.append("过去30分钟风险记忆为空")
        if values["neighborhood_max_500m"] == 0:
            missing_context.append("500米邻域风险为空")

        return {
            "score": score,
            "calibrated_probability": probability,
            "threshold": self.threshold,
            "alert": alert,
            "risk_level": level,
            "features": values,
            "evidence": evidence,
            "degradation_flags": missing_context,
            "semantics": "监督分数用于候选复核；校准概率仅作诊断，不直接替代人工处置判断。",
        }

    @staticmethod
    def _evidence(values: dict[str, float]) -> list[dict[str, Any]]:
        items = [
            {
                "key": "当前风险",
                "value": values["current_risk_probability"],
                "note": "决策时刻的第一阶段风险输入",
            },
            {
                "key": "30分钟均值",
                "value": values["risk_mean_past_30m"],
                "note": "过去30分钟风险记忆",
            },
            {
                "key": "风险变化",
                "value": values["risk_change_30m"],
                "note": "当前相对30分钟前的变化",
            },
            {
                "key": "上游300米",
                "value": values["upstream_mean_300m"],
                "note": "同路同方向上游均值",
            },
            {
                "key": "下游300米",
                "value": values["downstream_mean_300m"],
                "note": "同路同方向下游均值",
            },
            {
                "key": "邻域峰值",
                "value": values["neighborhood_max_500m"],
                "note": "500米邻域最大风险",
            },
        ]
        return sorted(items, key=lambda item: abs(float(item["value"])), reverse=True)

