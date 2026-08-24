from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (
    ACCIDENT_XLSX,
    MODEL_DIR,
    PROCESSED_DIR,
    TABLE_DIR,
    ensure_dirs,
    load_config,
    save_json,
)


NUMERIC_FEATURES = [
    "匹配桩号值(km)",
    "温度",
    "湿度",
    "能见度",
    "MQI评分",
    "PCI评分",
    "交通量",
    "median_speed",
    "traffic_density",
    "lane_density",
    "upstream_traffic",
    "downstream_traffic",
    "segment_count",
    "hour_sin",
    "hour_cos",
]

CATEGORICAL_FEATURES = [
    "路线方向",
    "事故区域",
    "事故车辆车型",
    "两客一危",
    "是否免费通行时间",
    "高明天气",
    "风向",
    "风力",
]


def load_registry() -> pd.DataFrame:
    frame = pd.read_excel(ACCIDENT_XLSX, engine="openpyxl")
    frame["事故发生时间"] = pd.to_datetime(frame["事故发生时间"], errors="coerce")
    frame = frame[frame["事故发生时间"].dt.year.eq(2025)].copy()
    minute = frame["事故发生时间"].dt.hour * 60 + frame["事故发生时间"].dt.minute
    frame["hour_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    frame["hour_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    frame["事故类型_建模"] = frame["事故类型"].replace({"翻车,碰撞": "翻车"})
    return frame.sort_values("事故发生时间").reset_index(drop=True)


def split_registry(frame: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    train_end = pd.Timestamp(config["train_end"])
    validation_end = pd.Timestamp(config["validation_end"])
    return {
        "train": frame[frame["事故发生时间"].le(train_end)].copy(),
        "validation": frame[
            frame["事故发生时间"].gt(train_end)
            & frame["事故发生时间"].le(validation_end)
        ].copy(),
        "test": frame[frame["事故发生时间"].gt(validation_end)].copy(),
    }


def fit_type_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=420,
        max_depth=8,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )
    combined = pd.concat([train, validation], ignore_index=True)
    pipeline.fit(
        combined[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        combined["事故类型_建模"],
    )
    return pipeline


def evaluate_type_model(
    model: Pipeline,
    test: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    truth = test["事故类型_建模"].astype(str)
    prediction = model.predict(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    probability = model.predict_proba(test[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    classes = model.named_steps["classifier"].classes_
    result = {
        "test_rows": int(len(test)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(truth, prediction, average="weighted", zero_division=0)
        ),
        "classes": classes.tolist(),
        "classification_report": classification_report(
            truth,
            prediction,
            output_dict=True,
            zero_division=0,
        ),
    }
    predictions = test[
        ["序号", "事故发生时间", "事故类型_建模", "事故等级", "路线方向", "匹配桩号值(km)"]
    ].copy()
    predictions["predicted_type"] = prediction
    predictions["prediction_confidence"] = probability.max(axis=1)
    matrix = pd.DataFrame(
        confusion_matrix(truth, prediction, labels=classes),
        index=classes,
        columns=classes,
    )
    return result, predictions, matrix


def policy_for_event(row: pd.Series) -> list[str]:
    accident_type = str(row.get("事故类型", "其他"))
    weather = str(row.get("高明天气", "未知"))
    traffic = pd.to_numeric(pd.Series([row.get("交通量")]), errors="coerce").iloc[0]
    median_speed = pd.to_numeric(
        pd.Series([row.get("median_speed")]), errors="coerce"
    ).iloc[0]
    upstream = pd.to_numeric(
        pd.Series([row.get("upstream_traffic")]), errors="coerce"
    ).iloc[0]
    downstream = pd.to_numeric(
        pd.Series([row.get("downstream_traffic")]), errors="coerce"
    ).iloc[0]
    measures = [
        "Activate upstream variable-message warning and publish the affected direction and kilometre range.",
        "Verify the event by CCTV or patrol before escalating the response level.",
    ]
    if any(token in weather for token in ["雨", "雾", "雪", "沙"]):
        measures.extend(
            [
                "Apply weather-adaptive speed control and increase the recommended following distance.",
                "Strengthen visibility and skid-risk warnings at upstream gantries.",
            ]
        )
    if accident_type in {"追尾", "碰撞", "刮擦"}:
        measures.extend(
            [
                "Protect the incident tail with an upstream buffer and prevent secondary collisions.",
                "Coordinate lane control and guide traffic to the available lanes before the queue reaches the upstream detector.",
            ]
        )
    if "翻车" in accident_type:
        measures.extend(
            [
                "Dispatch heavy rescue and lifting capability and inspect cargo leakage.",
                "Temporarily isolate adjacent lanes until vehicle stability and debris are assessed.",
            ]
        )
    if "失火" in accident_type:
        measures.extend(
            [
                "Dispatch fire response and establish an ignition and hazardous-material exclusion zone.",
                "Stop traffic upstream when smoke or thermal risk affects visibility or evacuation.",
            ]
        )
    if pd.notna(traffic) and traffic >= 1800:
        measures.append(
            "Use ramp or entrance-flow control to prevent additional demand from entering the congested section."
        )
    if (
        pd.notna(upstream)
        and pd.notna(downstream)
        and upstream > max(downstream * 1.25, downstream + 30)
    ):
        measures.append(
            "Treat the upstream-downstream imbalance as queue formation and extend the warning area upstream."
        )
    if pd.notna(median_speed) and median_speed < 40:
        measures.append(
            "Deploy low-speed queue warnings because the observed median speed indicates shockwave risk."
        )
    return list(dict.fromkeys(measures))


def build_teacher_seed(registry: pd.DataFrame) -> tuple[list[dict], dict]:
    records: list[dict] = []
    for _, row in registry.iterrows():
        input_packet = {
            "decision_time": str(row["事故发生时间"]),
            "location": {
                "stake_km": row.get("匹配桩号值(km)"),
                "direction": row.get("路线方向"),
                "area": row.get("事故区域"),
            },
            "traffic": {
                "volume_pcu_h": row.get("交通量"),
                "median_speed_km_h": row.get("median_speed"),
                "density": row.get("traffic_density"),
                "upstream_volume": row.get("upstream_traffic"),
                "downstream_volume": row.get("downstream_traffic"),
            },
            "weather": {
                "condition": row.get("高明天气"),
                "temperature_c": row.get("温度"),
                "humidity_pct": row.get("湿度"),
                "visibility": row.get("能见度"),
                "wind_direction": row.get("风向"),
                "wind_force": row.get("风力"),
            },
            "road": {
                "mqi": row.get("MQI评分"),
                "pci": row.get("PCI评分"),
            },
        }
        output_packet = {
            "event_confirmation": "confirmed_historical_event",
            "accident_type": str(row.get("事故类型")),
            "severity": str(row.get("事故等级")),
            "confidence_policy": (
                "Historical ground truth; online output must include calibrated confidence."
            ),
            "evidence": [
                "traffic-weather-road context",
                "historical event record",
                "upstream-downstream traffic context",
            ],
            "management_measures": policy_for_event(row),
            "human_review_required": bool(str(row.get("事故等级")) != "轻微事故"),
        }
        records.append(
            {
                "instruction": (
                    "Using only the supplied current-and-past evidence, confirm the suspected "
                    "road event, infer type and severity, and return targeted management actions."
                ),
                "input": input_packet,
                "output": output_packet,
                "provenance": {
                    "source": ACCIDENT_XLSX.name,
                    "row_id": int(row.get("序号")),
                    "generation": "rule-grounded teacher seed; not an LLM-generated claim",
                },
            }
        )
    policy_library = {
        "policy_version": "1.0",
        "principles": [
            "Verify by CCTV or patrol before irreversible control.",
            "Protect the upstream queue tail before optimizing throughput.",
            "Escalate weather, fire, rollover, injury, and hazardous-material scenarios.",
            "Every recommendation includes location, direction, trigger evidence, and review status.",
        ],
        "type_specific_examples": {
            accident_type: policy_for_event(group.iloc[0])
            for accident_type, group in registry.groupby("事故类型", observed=True)
        },
        "severity_gate": {
            "轻微事故": "operator confirmation",
            "一般事故": "mandatory supervisor review",
            "重大事故": "mandatory emergency-command review",
        },
    }
    return records, policy_library


def add_negative_teacher_seeds(records: list[dict], seed: int) -> list[dict]:
    candidate_path = PROCESSED_DIR / "causal_candidate_dataset_2025.csv"
    if not candidate_path.exists():
        return records
    usecols = [
        "segment_id",
        "bin5",
        "road_code",
        "direction_code",
        "stake",
        "candidate_source",
        "v4_model_accident_probability",
        "risk_change_30m",
        "upstream_mean_300m",
        "downstream_mean_300m",
        "neighborhood_std_500m",
        "spatial_coherence_500m",
        "target_event_30m",
    ]
    candidates = pd.read_csv(candidate_path, usecols=usecols, low_memory=False)
    negatives = candidates[
        candidates["target_event_30m"].eq(0)
        & candidates["candidate_source"].isin(["hard_negative", "uncertain_negative"])
    ]
    if len(negatives) > 1000:
        negatives = negatives.sample(1000, random_state=seed)
    for _, row in negatives.iterrows():
        records.append(
            {
                "instruction": (
                    "Use current-and-past risk evidence to decide whether this high-risk "
                    "candidate should be escalated or remain under monitoring."
                ),
                "input": {
                    "decision_time": str(row["bin5"]),
                    "segment_id": row["segment_id"],
                    "road_code": row["road_code"],
                    "direction_code": int(row["direction_code"]),
                    "stake_m": float(row["stake"]),
                    "stage1_probability": float(row["v4_model_accident_probability"]),
                    "risk_change_30m": row["risk_change_30m"],
                    "upstream_mean_300m": row["upstream_mean_300m"],
                    "downstream_mean_300m": row["downstream_mean_300m"],
                    "neighborhood_std_500m": row["neighborhood_std_500m"],
                    "spatial_coherence_500m": row["spatial_coherence_500m"],
                },
                "output": {
                    "event_confirmation": "not_confirmed_in_30min_ground_truth",
                    "recommended_state": "continue_monitoring",
                    "management_measures": [
                        "Retain the segment in the next 10-minute watchlist update.",
                        "Escalate only if temporal growth and spatial propagation become coherent.",
                    ],
                    "human_review_required": False,
                },
                "provenance": {
                    "source": candidate_path.name,
                    "generation": "hard-negative data distillation seed",
                },
            }
        )
    return records


def main() -> None:
    ensure_dirs()
    config = load_config()
    registry = load_registry()
    splits = split_registry(registry, config)
    model = fit_type_model(
        splits["train"],
        splits["validation"],
        int(config["random_seed"]),
    )
    model_path = MODEL_DIR / "accident_type_model.joblib"
    joblib.dump(model, model_path, compress=3)
    result, predictions, matrix = evaluate_type_model(model, splits["test"])
    predictions.to_csv(
        TABLE_DIR / "accident_type_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matrix.to_csv(
        TABLE_DIR / "accident_type_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    type_counts = registry["事故类型_建模"].value_counts().rename_axis("accident_type")
    type_counts.reset_index(name="count").to_csv(
        TABLE_DIR / "accident_type_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    severity_counts = registry["事故等级"].value_counts().rename_axis("severity")
    severity_counts.reset_index(name="count").to_csv(
        TABLE_DIR / "accident_severity_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    records, policy_library = build_teacher_seed(registry)
    records = add_negative_teacher_seeds(records, int(config["random_seed"]))
    jsonl_path = PROCESSED_DIR / "teacher_data_distillation_seed.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    save_json(PROCESSED_DIR / "management_policy_library.json", policy_library)

    test_sample = splits["test"][NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    tracemalloc.start()
    start = time.perf_counter()
    for _ in range(10):
        model.predict_proba(test_sample)
    elapsed = (time.perf_counter() - start) / 10.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.update(
        {
            "model_file_mb": model_path.stat().st_size / (1024**2),
            "mean_test_batch_latency_ms": elapsed * 1000.0,
            "python_peak_memory_mb": peak / (1024**2),
            "train_rows": len(splits["train"]),
            "validation_rows": len(splits["validation"]),
            "test_date_min": str(splits["test"]["事故发生时间"].min()),
            "test_date_max": str(splits["test"]["事故发生时间"].max()),
            "teacher_seed_rows": len(records),
            "severity_model_status": (
                "not fitted: 299/306 events are minor; general/major severity "
                "must use a conservative escalation gate and human review."
            ),
        }
    )
    save_json(TABLE_DIR / "reasoning_and_type_model_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
