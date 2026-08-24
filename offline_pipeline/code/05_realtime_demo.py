from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import (
    ACCIDENT_XLSX,
    MODEL_DIR,
    PROCESSED_DIR,
    PROJECT_DIR,
    TABLE_DIR,
    ensure_dirs,
    save_json,
)


def load_training_module():
    path = Path(__file__).resolve().parent / "02_train_confirmation_models.py"
    spec = spec_from_file_location("confirmation_training", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_reasoning_module():
    path = Path(__file__).resolve().parent / "03_build_reasoning_and_distillation.py"
    spec = spec_from_file_location("reasoning_training", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def nearest_historical_context(
    event_time: pd.Timestamp,
    direction_code: int,
) -> pd.Series:
    registry = pd.read_excel(ACCIDENT_XLSX, engine="openpyxl")
    registry["事故发生时间"] = pd.to_datetime(registry["事故发生时间"], errors="coerce")
    registry["direction_code"] = registry["路线方向"].map({"上行": 1, "下行": 0})
    same_direction = registry[registry["direction_code"].eq(direction_code)].copy()
    same_direction["time_distance_min"] = (
        same_direction["事故发生时间"] - event_time
    ).abs().dt.total_seconds() / 60.0
    return same_direction.sort_values("time_distance_min").iloc[0]


def severity_gate(type_name: str, context: pd.Series) -> dict:
    serious_flags = []
    if type_name in {"翻车", "失火"}:
        serious_flags.append("high-consequence accident type")
    if pd.to_numeric(pd.Series([context.get("重伤（人数）")]), errors="coerce").fillna(0).iloc[0] > 0:
        serious_flags.append("reported serious injury")
    if pd.to_numeric(pd.Series([context.get("死亡（人数）")]), errors="coerce").fillna(0).iloc[0] > 0:
        serious_flags.append("reported fatality")
    if str(context.get("两客一危")) == "属于":
        serious_flags.append("key passenger or hazardous vehicle")
    return {
        "predicted_severity_band": (
            "potentially_general_or_higher" if serious_flags else "minor_or_uncertain"
        ),
        "serious_flags": serious_flags,
        "human_review_required": True,
        "reason": (
            "Only 7 of 306 historical events are general/major; automated severity "
            "classification is not statistically reliable."
        ),
    }


def main() -> None:
    ensure_dirs()
    training = load_training_module()
    reasoning = load_reasoning_module()
    candidate_data = training.add_context_features(
        pd.read_csv(
            PROCESSED_DIR / "causal_candidate_dataset_2025.csv",
            low_memory=False,
        )
    )
    test = training.temporal_split(candidate_data, training.load_config())["test"]
    student = joblib.load(MODEL_DIR / "distilled_student.joblib")
    scores = np.clip(
        student.predict(test[training.STUDENT_FEATURES]),
        0.0,
        1.0,
    )
    test = test.copy()
    test["supervisory_score"] = scores
    eligible = test[
        test["target_event_30m"].eq(1)
        & test["supervisory_score"].ge(0.60)
    ].sort_values(
        ["supervisory_score", "lead_time_min"],
        ascending=[False, False],
    )
    if eligible.empty:
        raise ValueError("No detected test event is available for the replay demo.")
    row = eligible.iloc[0]

    mapped_events = pd.read_csv(
        PROCESSED_DIR / "mapped_accident_events_2025.csv",
        low_memory=False,
    )
    mapped_events["bin5"] = pd.to_datetime(mapped_events["bin5"], errors="coerce")
    event = mapped_events[
        mapped_events["accident_id"].astype(str).eq(str(row["target_event_id"]))
    ].iloc[0]
    event_time = pd.Timestamp(event["bin5"])
    historical_context = nearest_historical_context(
        event_time,
        int(event["direction_code"]),
    )

    type_model = joblib.load(MODEL_DIR / "accident_type_model.joblib")
    context_frame = pd.DataFrame([historical_context])
    minute = context_frame["事故发生时间"].dt.hour * 60 + context_frame["事故发生时间"].dt.minute
    context_frame["hour_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    context_frame["hour_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    type_probability = type_model.predict_proba(
        context_frame[
            reasoning.NUMERIC_FEATURES + reasoning.CATEGORICAL_FEATURES
        ]
    )[0]
    type_classes = type_model.named_steps["classifier"].classes_
    top_indices = np.argsort(type_probability)[::-1][:3]
    type_hypotheses = [
        {
            "type": str(type_classes[index]),
            "probability": float(type_probability[index]),
        }
        for index in top_indices
    ]
    predicted_type = type_hypotheses[0]["type"]

    compressed_memory = {
        "current_stage1_probability": float(row["v4_model_accident_probability"]),
        "past_30min": {
            "lag_5min": float(row["risk_lag_5m"])
            if pd.notna(row["risk_lag_5m"])
            else None,
            "lag_30min": float(row["risk_lag_30m"])
            if pd.notna(row["risk_lag_30m"])
            else None,
            "mean": float(row["risk_mean_past_30m"])
            if pd.notna(row["risk_mean_past_30m"])
            else None,
            "change": float(row["risk_change_30m"])
            if pd.notna(row["risk_change_30m"])
            else None,
        },
        "directed_spatial_context": {
            "upstream_mean_300m": float(row["upstream_mean_300m"]),
            "upstream_max_300m": float(row["upstream_max_300m"]),
            "downstream_mean_300m": float(row["downstream_mean_300m"]),
            "downstream_max_300m": float(row["downstream_max_300m"]),
            "neighborhood_max_500m": float(row["neighborhood_max_500m"]),
            "high_risk_cells_500m": int(row["high_risk_cells_500m"]),
            "spatial_gradient_600m": float(row["spatial_gradient_600m"]),
        },
    }
    memory_json = json.dumps(compressed_memory, ensure_ascii=False)
    severity = severity_gate(predicted_type, historical_context)
    measures = reasoning.policy_for_event(
        historical_context.assign(事故类型=predicted_type)
        if hasattr(historical_context, "assign")
        else historical_context
    )
    # Series.assign is unavailable; create an explicit copy for policy generation.
    policy_row = historical_context.copy()
    policy_row["事故类型"] = predicted_type
    measures = reasoning.policy_for_event(policy_row)

    output = {
        "mode": "historical_causal_replay",
        "causal_boundary": {
            "confirmation_module": (
                "uses only the decision-time row and past 30-minute/spatial context"
            ),
            "type_module": (
                "demonstration uses event-time traffic/weather context; its type metrics "
                "are reported separately from pre-event confirmation metrics"
            ),
        },
        "decision": {
            "decision_time": str(row["bin5"]),
            "alert": bool(row["supervisory_score"] >= 0.60),
            "supervisory_score": float(row["supervisory_score"]),
            "threshold": 0.60,
            "predicted_horizon": "within 30 minutes",
            "lead_time_to_recorded_event_min": float(row["lead_time_min"]),
        },
        "location": {
            "segment_id": str(row["segment_id"]),
            "road_code": str(row["road_code"]),
            "direction_code": int(row["direction_code"]),
            "stake_m": float(row["stake"]),
            "upstream_downstream_orientation_assumption": (
                "direction 1 follows increasing stake; direction 0 follows decreasing stake"
            ),
        },
        "compressed_long_term_memory": compressed_memory,
        "memory_packet_bytes_utf8": len(memory_json.encode("utf-8")),
        "type_hypotheses": type_hypotheses,
        "severity_gate": severity,
        "management_measures": measures,
        "audit": {
            "target_event_id": str(row["target_event_id"]),
            "recorded_event_time": str(event_time),
            "historical_context_time_distance_min": float(
                historical_context["time_distance_min"]
            ),
            "operator_confirmation_required": True,
        },
    }
    demo_dir = PROJECT_DIR / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    save_json(demo_dir / "realtime_inference_example.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
