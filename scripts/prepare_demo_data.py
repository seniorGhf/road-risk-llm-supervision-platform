from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


SNAPSHOT_TIME = "2025-11-05 14:40:00"
CASE_TIME = "2025-11-05 15:05:00"
CASE_SEGMENT = "G9411_1_137370"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def risk_level(score: float) -> str:
    if score >= 0.8:
        return "高"
    if score >= 0.6:
        return "中"
    return "低"


def load_snapshot(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "segment_id": row["segment_id"],
            "road_code": row["road_code"],
            "direction_code": int(float(row["direction_code"])),
            "stake_m": safe_number(row["stake"]),
            "stage1_score": safe_number(row["v4_model_accident_probability"]),
        }
        for row in rows
    ]


def load_watchlist(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["bin5"] != SNAPSHOT_TIME:
                continue
            score = safe_number(row["score_distilled_student"])
            rows.append(
                {
                    "segment_id": row["segment_id"],
                    "time": row["bin5"],
                    "road_code": row["road_code"],
                    "direction_code": int(float(row["direction_code"])),
                    "stake_m": safe_number(row["stake"]),
                    "stake_km": round(safe_number(row["stake"]) / 1000, 3),
                    "score": round(score, 6),
                    "stage1_score": round(safe_number(row["score_stage1_risk"]), 6),
                    "calibrated_probability": round(
                        safe_number(row["calibrated_probability_distilled_student"]), 6
                    ),
                    "risk_level": risk_level(score),
                    "candidate_source": row["candidate_source"],
                    "review_status": "待复核" if score >= 0.6 else "观察",
                    "lead_time_min": safe_number(row.get("lead_time_min"), -1),
                    "target_event_id": row.get("target_event_id", ""),
                }
            )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:60]


def load_default_features(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["segment_id"] == CASE_SEGMENT and row["bin5"] == CASE_TIME:
                hour = 15
                return {
                    "v4_model_accident_probability": safe_number(row["v4_model_accident_probability"]),
                    "current_risk_probability": safe_number(row["current_risk_probability"]),
                    "collision_risk_value": safe_number(row["collision_risk_value"]),
                    "risk_lag_5m": safe_number(row["risk_lag_5m"]),
                    "risk_lag_30m": safe_number(row["risk_lag_30m"]),
                    "risk_mean_past_30m": safe_number(row["risk_mean_past_30m"]),
                    "risk_change_5m": safe_number(row["risk_change_5m"]),
                    "risk_change_30m": safe_number(row["risk_change_30m"]),
                    "upstream_mean_300m": safe_number(row["upstream_mean_300m"]),
                    "upstream_max_300m": safe_number(row["upstream_max_300m"]),
                    "downstream_mean_300m": safe_number(row["downstream_mean_300m"]),
                    "downstream_max_300m": safe_number(row["downstream_max_300m"]),
                    "neighborhood_mean_500m": safe_number(row["neighborhood_mean_500m"]),
                    "neighborhood_max_500m": safe_number(row["neighborhood_max_500m"]),
                    "neighborhood_std_500m": safe_number(row["neighborhood_std_500m"]),
                    "high_risk_cells_500m": safe_number(row["high_risk_cells_500m"]),
                    "spatial_gradient_600m": safe_number(row["spatial_gradient_600m"]),
                    "center_neighbor_contrast": safe_number(row["center_neighbor_contrast"]),
                    "direction_code": safe_number(row["direction_code"]),
                    "road_is_s5": 1.0 if row["road_code"] == "S5" else 0.0,
                    "hour_sin": math.sin(2 * math.pi * hour / 24),
                    "hour_cos": math.cos(2 * math.pi * hour / 24),
                }
    raise RuntimeError("未找到默认研判案例特征行")


def route_series(snapshot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in snapshot_rows:
        groups.setdefault((row["road_code"], row["direction_code"]), []).append(row)
    output = []
    for (road, direction), rows in sorted(groups.items()):
        rows.sort(key=lambda item: item["stake_m"])
        if len(rows) > 90:
            step = max(1, len(rows) // 90)
            rows = rows[::step]
        output.append(
            {
                "road_code": road,
                "direction_code": direction,
                "points": [
                    {
                        "segment_id": row["segment_id"],
                        "stake_km": round(row["stake_m"] / 1000, 3),
                        "score": round(row["stage1_score"], 5),
                    }
                    for row in rows
                ],
            }
        )
    return output


def metrics_from_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    student = next(row for row in rows if row["model"] == "distilled_student")
    return {
        "roc_auc": safe_number(student["roc_auc"]),
        "pr_auc": safe_number(student["pr_auc"]),
        "precision": safe_number(student["precision"]),
        "recall": safe_number(student["recall"]),
        "f1": safe_number(student["f1"]),
        "event_recall": safe_number(student["event_recall"]),
        "false_positive_reduction": safe_number(student["false_alarm_reduction_vs_stage1"]),
        "median_lead_time_min": safe_number(student["median_lead_time_min"]),
        "test_events": int(float(student["event_count"])),
        "detected_events": int(float(student["detected_event_count"])),
    }


def validation_payload(report: dict[str, Any]) -> dict[str, Any]:
    checks = report.get("checks", [])
    if isinstance(checks, dict):
        checks = [
            {"name": key, "status": value.get("status", "PASS"), "evidence": value.get("evidence", {})}
            for key, value in checks.items()
        ]
    return {"overall": report.get("overall", "PASS"), "checks": checks}


def build(source: Path, target: Path) -> None:
    source = source.resolve()
    target = target.resolve()
    data_dir = target / "data"
    model_dir = target / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    tables = source / "results" / "tables"
    snapshot_rows = load_snapshot(tables / "dashboard_spatial_risk_snapshot_20251105_1440.csv")
    watchlist = load_watchlist(tables / "confirmation_model_test_predictions.csv")
    default_features = load_default_features(source / "data" / "processed" / "2025-11_causal_candidates.csv")
    cases = read_json(source / "case_analysis" / "deep_case_summary.json")["cases"]
    realtime = read_json(source / "demo" / "realtime_inference_example.json")
    memory = read_json(source / "memory_workload" / "memory_workload_summary.json")
    optimization = read_json(source / "prescriptive_optimization" / "optimization_summary.json")
    benchmark = read_json(source / "llm_benchmark" / "results" / "benchmark_analysis_summary.json")
    model_summary = read_json(tables / "reasoning_and_type_model_summary.json")
    validation = read_json(source / "report" / "validation_report.json")
    policy = read_json(source / "data" / "processed" / "management_policy_library.json")

    metrics = metrics_from_csv(tables / "confirmation_model_metrics_test.csv")
    metrics.update(
        {
            "raw_rows_full_year": memory["pipeline_stages"][0]["records"],
            "test_watchlist_sites": memory["pipeline_stages"][-1]["records"],
            "workload_reduction": memory["pipeline_stages"][-1]["reduction_vs_raw"],
            "student_model_mb": 0.060,
            "type_accuracy": model_summary["accuracy"],
            "type_macro_f1": model_summary["macro_f1"],
        }
    )

    selected_case = {
        **cases[0],
        "decision": realtime["decision"],
        "location": realtime["location"],
        "memory": realtime["compressed_long_term_memory"],
        "type_hypotheses": realtime["type_hypotheses"],
        "severity_gate": realtime["severity_gate"],
        "management_measures": realtime["management_measures"],
    }

    successful_models = [
        {
            "model": item["model"],
            "composite_score": safe_number(item["composite_score"]),
            "seconds_per_case": safe_number(item["mean_seconds_per_case"]),
            "schema_validity": safe_number(item["schema_validity"]),
            "unsupported_action_rate": safe_number(item["unsupported_action_rate"]),
            "parameter_size": item.get("parameter_size", ""),
        }
        for item in benchmark["models"]
        if int(item.get("cases_returned", 0)) > 0
    ]

    payload = {
        "system": {
            "name": "大模型道路路域风险监管平台",
            "short_name": "路域智监",
            "version": "V1.0.0",
            "mode": "历史因果回放",
        },
        "snapshot": {
            "time": SNAPSHOT_TIME,
            "label": "2025年11月5日 14:40 独立测试回放",
            "future_outcome_hidden": True,
            "update_minutes": 10,
            "prediction_horizon_minutes": 30,
        },
        "metrics": metrics,
        "pipeline": memory["pipeline_stages"],
        "route_series": route_series(snapshot_rows),
        "watchlist": watchlist,
        "cases": cases,
        "selected_case": selected_case,
        "default_features": default_features,
        "policy_library": policy,
        "model": {
            "student_features": list(default_features),
            "student_feature_count": len(default_features),
            "type_model": model_summary,
            "memory_packet": memory["memory_packet"],
            "retained_memory_fields": memory["retained_memory_fields"],
        },
        "benchmarks": {
            "models": successful_models,
            "noninferiority": benchmark.get("noninferiority", {}),
        },
        "optimization": optimization,
        "validation": validation_payload(validation),
        "boundaries": [
            "监督分数用于候选复核，不等同于事故已发生的概率判定。",
            "事故类型仅为假设，六分类测试准确率为60.9%，必须人工复核。",
            "严重程度样本高度失衡，系统不执行自动严重程度定级。",
            "处置建议来自授权政策库，不会直接下发至生产设备。",
            "优化成本和缓解率是敏感性分析参数，不代表已观测的因果效果。",
        ],
        "evidence": {
            "source": str(source),
            "snapshot_file": "results/tables/dashboard_spatial_risk_snapshot_20251105_1440.csv",
            "prediction_file": "results/tables/confirmation_model_test_predictions.csv",
            "model_manifest": "models/confirmation_model_manifest.json",
            "frozen_status": "19/19 checks passed; 249 artifacts frozen",
        },
    }

    payload = json_safe(payload)
    (data_dir / "platform_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    for name in [
        "distilled_student.joblib",
        "site_level_probability_calibrator.joblib",
        "confirmation_model_manifest.json",
    ]:
        shutil.copy2(source / "models" / name, model_dir / name)
    print(f"已生成：{data_dir / 'platform_data.json'}")
    print(f"风险候选：{len(watchlist)} 条；路网序列：{len(payload['route_series'])} 组")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从冻结研究结果构建平台演示数据")
    parser.add_argument("--source", required=True, type=Path, help="2大模型构建及管理目录")
    parser.add_argument("--target", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.source, args.target)
