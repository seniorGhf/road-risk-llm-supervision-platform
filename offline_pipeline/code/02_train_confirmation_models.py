from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import (
    MODEL_DIR,
    PROCESSED_DIR,
    RISK_COL,
    TABLE_DIR,
    binary_metrics,
    choose_threshold,
    ensure_dirs,
    load_config,
    save_json,
)


COMPONENT_FEATURES = [
    RISK_COL,
    "current_risk_probability",
    "collision_risk_value",
    "v4_main_dynamic_probability",
    "v4_static_collision_probability",
    "v4_main_adaptive_dynamic_weight",
]

TEMPORAL_FEATURES = [
    "risk_lag_5m",
    "risk_lag_10m",
    "risk_lag_20m",
    "risk_lag_30m",
    "risk_mean_past_30m",
    "risk_max_past_30m",
    "risk_std_past_30m",
    "risk_change_5m",
    "risk_change_30m",
    "risk_acceleration",
]

SPATIAL_FEATURES = [
    "upstream_risk_100m",
    "upstream_risk_200m",
    "upstream_risk_300m",
    "upstream_risk_400m",
    "upstream_risk_500m",
    "downstream_risk_100m",
    "downstream_risk_200m",
    "downstream_risk_300m",
    "downstream_risk_400m",
    "downstream_risk_500m",
    "upstream_mean_300m",
    "upstream_max_300m",
    "downstream_mean_300m",
    "downstream_max_300m",
    "neighborhood_mean_500m",
    "neighborhood_max_500m",
    "neighborhood_std_500m",
    "high_risk_cells_500m",
    "spatial_gradient_600m",
    "center_neighbor_contrast",
    "spatial_coherence_500m",
]

CONTEXT_FEATURES = [
    "direction_code",
    "stake_scaled",
    "road_is_s5",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

STUDENT_FEATURES = [
    RISK_COL,
    "current_risk_probability",
    "collision_risk_value",
    "risk_lag_5m",
    "risk_lag_30m",
    "risk_mean_past_30m",
    "risk_change_5m",
    "risk_change_30m",
    "upstream_mean_300m",
    "upstream_max_300m",
    "downstream_mean_300m",
    "downstream_max_300m",
    "neighborhood_mean_500m",
    "neighborhood_max_500m",
    "neighborhood_std_500m",
    "high_risk_cells_500m",
    "spatial_gradient_600m",
    "center_neighbor_contrast",
    "direction_code",
    "road_is_s5",
    "hour_sin",
    "hour_cos",
]


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["bin5"] = pd.to_datetime(
        frame["bin5"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    minute_of_day = frame["bin5"].dt.hour * 60 + frame["bin5"].dt.minute
    frame["hour_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    weekday = frame["bin5"].dt.dayofweek
    frame["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
    frame["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
    frame["road_is_s5"] = frame["road_code"].astype(str).str.upper().eq("S5").astype(int)
    frame["stake_scaled"] = pd.to_numeric(frame["stake"], errors="coerce") / 100_000.0
    return frame


def temporal_split(frame: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    train_end = pd.Timestamp(config["train_end"])
    validation_end = pd.Timestamp(config["validation_end"])
    test_end = pd.Timestamp(config["test_end"])
    return {
        "train": frame[frame["bin5"].le(train_end)].copy(),
        "validation": frame[
            frame["bin5"].gt(train_end) & frame["bin5"].le(validation_end)
        ].copy(),
        "test": frame[
            frame["bin5"].gt(validation_end) & frame["bin5"].le(test_end)
        ].copy(),
    }


def fit_classifier(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    config: dict,
) -> tuple[lgb.LGBMClassifier, dict]:
    positive = max(int(train["target_event_30m"].sum()), 1)
    negative = max(len(train) - positive, 1)
    scale_positive = min(25.0, negative / positive)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=360,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=8,
        min_child_samples=45,
        subsample=0.85,
        colsample_bytree=0.82,
        reg_alpha=0.12,
        reg_lambda=0.35,
        scale_pos_weight=scale_positive,
        random_state=int(config["random_seed"]),
        n_jobs=-1,
        verbosity=-1,
    )
    history: dict = {}
    model.fit(
        train[features],
        train["target_event_30m"].astype(int),
        eval_set=[
            (train[features], train["target_event_30m"].astype(int)),
            (validation[features], validation["target_event_30m"].astype(int)),
        ],
        eval_names=["train", "validation"],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[lgb.record_evaluation(history)],
    )
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path, compress=3)
    return model, history


def platt_calibrate(
    raw_validation: np.ndarray,
    y_validation: np.ndarray,
    raw_other: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression]:
    eps = 1e-6
    validation_logit = np.log(
        np.clip(raw_validation, eps, 1 - eps)
        / np.clip(1 - raw_validation, eps, 1 - eps)
    ).reshape(-1, 1)
    other_logit = np.log(
        np.clip(raw_other, eps, 1 - eps)
        / np.clip(1 - raw_other, eps, 1 - eps)
    ).reshape(-1, 1)
    calibrator = LogisticRegression(C=100.0, max_iter=1000)
    calibrator.fit(validation_logit, y_validation)
    return calibrator.predict_proba(other_logit)[:, 1], calibrator


def bootstrap_intervals(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    working = frame[["bin5", "target_event_30m", score_column]].copy()
    working["date"] = working["bin5"].dt.date
    dates = np.array(sorted(working["date"].unique()))
    values: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "f1": []}
    for _ in range(iterations):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        sample = pd.concat(
            [working[working["date"].eq(date)] for date in sampled_dates],
            ignore_index=True,
        )
        y = sample["target_event_30m"].to_numpy(dtype=int)
        score = sample[score_column].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue
        values["pr_auc"].append(float(average_precision_score(y, score)))
        values["roc_auc"].append(float(roc_auc_score(y, score)))
        values["f1"].append(
            float(f1_score(y, score >= threshold, zero_division=0))
        )
    return {
        metric: [
            float(np.quantile(observations, 0.025)),
            float(np.quantile(observations, 0.975)),
        ]
        for metric, observations in values.items()
        if observations
    }


def event_level_metrics(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> dict[str, float]:
    positives = frame[frame["target_event_30m"].eq(1)].copy()
    if positives.empty:
        return {
            "event_count": 0,
            "detected_event_count": 0,
            "event_recall": float("nan"),
            "median_lead_time_min": float("nan"),
        }
    grouped = positives.groupby("target_event_id", observed=True)
    event_peak = grouped[score_column].max()
    detected_ids = event_peak[event_peak.ge(threshold)].index
    detected = positives[
        positives["target_event_id"].isin(detected_ids)
        & positives[score_column].ge(threshold)
    ]
    lead = detected.groupby("target_event_id", observed=True)["lead_time_min"].max()
    return {
        "event_count": int(event_peak.size),
        "detected_event_count": int(len(detected_ids)),
        "event_recall": float(len(detected_ids) / max(event_peak.size, 1)),
        "median_lead_time_min": float(lead.median()) if len(lead) else float("nan"),
        "mean_lead_time_min": float(lead.mean()) if len(lead) else float("nan"),
    }


def model_resource_profile(
    model_path: Path,
    model,
    sample: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    sample = sample[features].head(10_000)
    tracemalloc.start()
    start = time.perf_counter()
    for _ in range(5):
        if hasattr(model, "predict_proba"):
            model.predict_proba(sample)
        else:
            model.predict(sample)
    elapsed = (time.perf_counter() - start) / 5.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "model_file_mb": float(model_path.stat().st_size / (1024**2)),
        "batch_rows": int(len(sample)),
        "mean_batch_latency_ms": float(elapsed * 1000.0),
        "mean_latency_ms_per_row": float(elapsed * 1000.0 / max(len(sample), 1)),
        "python_peak_memory_mb": float(peak / (1024**2)),
        "feature_count": int(len(features)),
    }


def fit_student_variant(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    soft_target: np.ndarray,
    sample_weight: np.ndarray,
    config: dict,
) -> tuple[lgb.LGBMRegressor, dict]:
    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=96,
        learning_rate=0.045,
        num_leaves=15,
        max_depth=5,
        min_child_samples=55,
        subsample=0.85,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=0.40,
        random_state=int(config["random_seed"]),
        n_jobs=-1,
        verbosity=-1,
    )
    history: dict = {}
    model.fit(
        train[features],
        soft_target,
        sample_weight=sample_weight,
        eval_set=[
            (
                validation[features],
                validation["target_event_30m"].to_numpy(dtype=float),
            )
        ],
        eval_names=["validation"],
        eval_metric="l2",
        callbacks=[lgb.record_evaluation(history)],
    )
    joblib.dump(model, MODEL_DIR / f"{name}.joblib", compress=3)
    return model, history


def main() -> None:
    ensure_dirs()
    config = load_config()
    data_path = PROCESSED_DIR / "causal_candidate_dataset_2025.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} is missing. Run 01_build_causal_spatiotemporal_dataset.py first."
        )
    data = pd.read_csv(data_path, low_memory=False)
    data = add_context_features(data)
    data = data.dropna(subset=["bin5", "target_event_30m"])
    data["target_event_30m"] = data["target_event_30m"].astype(int)
    splits = temporal_split(data, config)
    if any(part.empty for part in splits.values()):
        raise ValueError({name: len(part) for name, part in splits.items()})

    feature_sets = {
        "temporal_only": list(dict.fromkeys(COMPONENT_FEATURES + TEMPORAL_FEATURES + CONTEXT_FEATURES)),
        "spatial_only": list(dict.fromkeys(COMPONENT_FEATURES + SPATIAL_FEATURES + CONTEXT_FEATURES)),
        "no_spatial": list(dict.fromkeys(COMPONENT_FEATURES + TEMPORAL_FEATURES + CONTEXT_FEATURES)),
        "no_temporal": list(dict.fromkeys(COMPONENT_FEATURES + SPATIAL_FEATURES + CONTEXT_FEATURES)),
        "full_teacher": list(
            dict.fromkeys(
                COMPONENT_FEATURES + TEMPORAL_FEATURES + SPATIAL_FEATURES + CONTEXT_FEATURES
            )
        ),
    }
    available = set(data.columns)
    feature_sets = {
        name: [feature for feature in features if feature in available]
        for name, features in feature_sets.items()
    }

    trained: dict[str, lgb.LGBMClassifier] = {}
    histories: dict[str, dict] = {}
    for name in ["temporal_only", "spatial_only", "full_teacher"]:
        model, history = fit_classifier(
            name,
            splits["train"],
            splits["validation"],
            feature_sets[name],
            config,
        )
        trained[name] = model
        histories[name] = history

    distilled_train = splits["train"][
        splits["train"]["candidate_source"].isin(["future_event", "random_negative"])
    ].copy()
    no_distill, no_distill_history = fit_classifier(
        "no_data_distillation",
        distilled_train,
        splits["validation"],
        feature_sets["full_teacher"],
        config,
    )
    trained["no_data_distillation"] = no_distill
    histories["no_data_distillation"] = no_distill_history

    teacher = trained["full_teacher"]
    teacher_train_score = teacher.predict_proba(
        splits["train"][feature_sets["full_teacher"]]
    )[:, 1]
    soft_target = (
        0.72 * splits["train"]["target_event_30m"].to_numpy(dtype=float)
        + 0.28 * teacher_train_score
    )
    positive = max(int(splits["train"]["target_event_30m"].sum()), 1)
    negative = max(len(splits["train"]) - positive, 1)
    student_weights = np.where(
        splits["train"]["target_event_30m"].to_numpy(dtype=int) == 1,
        min(20.0, negative / positive),
        1.0,
    )
    student = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=96,
        learning_rate=0.045,
        num_leaves=15,
        max_depth=5,
        min_child_samples=55,
        subsample=0.85,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=0.40,
        random_state=int(config["random_seed"]),
        n_jobs=-1,
        verbosity=-1,
    )
    student_history: dict = {}
    student.fit(
        splits["train"][STUDENT_FEATURES],
        soft_target,
        sample_weight=student_weights,
        eval_set=[
            (
                splits["validation"][STUDENT_FEATURES],
                splits["validation"]["target_event_30m"].to_numpy(dtype=float),
            )
        ],
        eval_names=["validation"],
        eval_metric="l2",
        callbacks=[lgb.record_evaluation(student_history)],
    )
    raw_student_validation = np.clip(
        student.predict(splits["validation"][STUDENT_FEATURES]), 0.0, 1.0
    )
    raw_student_test = np.clip(
        student.predict(splits["test"][STUDENT_FEATURES]), 0.0, 1.0
    )
    calibrated_student_test, calibrator = platt_calibrate(
        raw_student_validation,
        splits["validation"]["target_event_30m"].to_numpy(dtype=int),
        raw_student_test,
    )
    calibrated_student_validation, _ = platt_calibrate(
        raw_student_validation,
        splits["validation"]["target_event_30m"].to_numpy(dtype=int),
        raw_student_validation,
    )
    student_path = MODEL_DIR / "distilled_student.joblib"
    calibrator_path = MODEL_DIR / "student_platt_calibrator.joblib"
    joblib.dump(student, student_path, compress=3)
    joblib.dump(calibrator, calibrator_path, compress=3)
    histories["distilled_student"] = student_history

    compact_no_spatial = [
        feature
        for feature in list(dict.fromkeys(COMPONENT_FEATURES + TEMPORAL_FEATURES + CONTEXT_FEATURES))
        if feature in data.columns
    ]
    compact_no_temporal = [
        feature
        for feature in list(
            dict.fromkeys(
                COMPONENT_FEATURES
                + [
                    "upstream_mean_300m",
                    "upstream_max_300m",
                    "downstream_mean_300m",
                    "downstream_max_300m",
                    "neighborhood_mean_500m",
                    "neighborhood_max_500m",
                    "neighborhood_std_500m",
                    "high_risk_cells_500m",
                    "spatial_gradient_600m",
                    "center_neighbor_contrast",
                    "spatial_coherence_500m",
                ]
                + CONTEXT_FEATURES
            )
        )
        if feature in data.columns
    ]
    student_variants: dict[str, tuple[lgb.LGBMRegressor, list[str]]] = {}
    for variant_name, variant_features, variant_target, variant_train, variant_weight in [
        (
            "student_no_spatial",
            compact_no_spatial,
            soft_target,
            splits["train"],
            student_weights,
        ),
        (
            "student_no_temporal",
            compact_no_temporal,
            soft_target,
            splits["train"],
            student_weights,
        ),
        (
            "student_no_teacher_kd",
            STUDENT_FEATURES,
            splits["train"]["target_event_30m"].to_numpy(dtype=float),
            splits["train"],
            student_weights,
        ),
    ]:
        variant_model, variant_history = fit_student_variant(
            variant_name,
            variant_train,
            splits["validation"],
            variant_features,
            variant_target,
            variant_weight,
            config,
        )
        student_variants[variant_name] = (variant_model, variant_features)
        histories[variant_name] = variant_history

    no_hard_mask = ~splits["train"]["candidate_source"].eq("hard_negative")
    no_hard_train = splits["train"].loc[no_hard_mask].copy()
    no_hard_target = soft_target[no_hard_mask.to_numpy()]
    no_hard_weight = student_weights[no_hard_mask.to_numpy()]
    no_hard_model, no_hard_history = fit_student_variant(
        "student_no_hard_negative_distillation",
        no_hard_train,
        splits["validation"],
        STUDENT_FEATURES,
        no_hard_target,
        no_hard_weight,
        config,
    )
    student_variants["student_no_hard_negative_distillation"] = (
        no_hard_model,
        STUDENT_FEATURES,
    )
    histories["student_no_hard_negative_distillation"] = no_hard_history

    validation_scores: dict[str, np.ndarray] = {
        "stage1_risk": splits["validation"][RISK_COL].to_numpy(dtype=float),
        "distilled_student": raw_student_validation,
    }
    test_scores: dict[str, np.ndarray] = {
        "stage1_risk": splits["test"][RISK_COL].to_numpy(dtype=float),
        "distilled_student": raw_student_test,
    }
    for name, model in trained.items():
        features = (
            feature_sets["full_teacher"]
            if name == "no_data_distillation"
            else feature_sets[name]
        )
        validation_scores[name] = model.predict_proba(
            splits["validation"][features]
        )[:, 1]
        test_scores[name] = model.predict_proba(splits["test"][features])[:, 1]
    for name, (variant_model, variant_features) in student_variants.items():
        validation_scores[name] = np.clip(
            variant_model.predict(splits["validation"][variant_features]),
            0.0,
            1.0,
        )
        test_scores[name] = np.clip(
            variant_model.predict(splits["test"][variant_features]),
            0.0,
            1.0,
        )

    thresholds: dict[str, float] = {}
    metrics_rows: list[dict] = []
    test_output = splits["test"][
        [
            "segment_id",
            "bin5",
            "road_code",
            "direction_code",
            "stake",
            "candidate_source",
            "target_event_30m",
            "target_event_id",
            "lead_time_min",
        ]
    ].copy()
    for name, score in test_scores.items():
        if name == "stage1_risk" or name.startswith("distilled_student") or name.startswith("student_"):
            threshold = float(config["hard_negative_threshold"])
        else:
            threshold = choose_threshold(
                splits["validation"]["target_event_30m"].to_numpy(dtype=int),
                validation_scores[name],
                minimum=float(config["alert_threshold_minimum"]),
            )
        thresholds[name] = threshold
        test_output[f"score_{name}"] = score
        if name == "distilled_student":
            test_output["calibrated_probability_distilled_student"] = (
                calibrated_student_test
            )
        result = binary_metrics(
            splits["test"]["target_event_30m"].to_numpy(dtype=int),
            score,
            threshold,
        )
        result.update(
            {
                "model": name,
                "split": "test_2025_11_12",
                "candidate_rows": int(len(splits["test"])),
                "positive_rows": int(splits["test"]["target_event_30m"].sum()),
            }
        )
        event_result = event_level_metrics(
            test_output,
            f"score_{name}",
            threshold,
        )
        result.update(event_result)
        metrics_rows.append(result)

    metrics = pd.DataFrame(metrics_rows)
    baseline_fp = float(
        metrics.loc[metrics["model"].eq("stage1_risk"), "false_positive_count"].iloc[0]
    )
    metrics["false_alarm_reduction_vs_stage1"] = (
        baseline_fp - metrics["false_positive_count"]
    ) / max(baseline_fp, 1.0)
    metrics.to_csv(
        TABLE_DIR / "confirmation_model_metrics_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_output.to_csv(
        TABLE_DIR / "confirmation_model_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importance = pd.DataFrame(
        {
            "feature": feature_sets["full_teacher"],
            "gain_importance": teacher.booster_.feature_importance(
                importance_type="gain"
            ),
            "split_importance": teacher.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("gain_importance", ascending=False)
    importance["gain_share"] = importance["gain_importance"] / max(
        importance["gain_importance"].sum(), 1e-12
    )
    importance.to_csv(
        TABLE_DIR / "teacher_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    history_rows = []
    for model_name, history in histories.items():
        for dataset_name, metrics_history in history.items():
            for metric_name, values in metrics_history.items():
                for iteration, value in enumerate(values, start=1):
                    history_rows.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "metric": metric_name,
                            "iteration": iteration,
                            "value": float(value),
                        }
                    )
    pd.DataFrame(history_rows).to_csv(
        TABLE_DIR / "training_iteration_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_for_bootstrap = test_output.copy()
    bootstrap = bootstrap_intervals(
        test_for_bootstrap,
        "score_distilled_student",
        thresholds["distilled_student"],
        iterations=200,
        seed=int(config["random_seed"]),
    )
    resource_rows = [
        {
            "model": "full_teacher",
            **model_resource_profile(
                MODEL_DIR / "full_teacher.joblib",
                teacher,
                splits["test"],
                feature_sets["full_teacher"],
            ),
        },
        {
            "model": "distilled_student",
            **model_resource_profile(
                student_path,
                student,
                splits["test"],
                STUDENT_FEATURES,
            ),
        },
    ]
    pd.DataFrame(resource_rows).to_csv(
        TABLE_DIR / "model_resource_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_summary = {
        name: {
            "rows": int(len(part)),
            "positive_rows": int(part["target_event_30m"].sum()),
            "positive_rate": float(part["target_event_30m"].mean()),
            "event_count": int(
                part.loc[part["target_event_30m"].eq(1), "target_event_id"].nunique()
            ),
            "date_min": str(part["bin5"].min()),
            "date_max": str(part["bin5"].max()),
        }
        for name, part in splits.items()
    }
    save_json(
        MODEL_DIR / "confirmation_model_manifest.json",
        {
            "thresholds": thresholds,
            "feature_sets": feature_sets,
            "student_features": STUDENT_FEATURES,
            "student_ablation_features": {
                "student_no_spatial": compact_no_spatial,
                "student_no_temporal": compact_no_temporal,
                "student_no_teacher_kd": STUDENT_FEATURES,
                "student_no_hard_negative_distillation": STUDENT_FEATURES,
            },
            "split_summary": split_summary,
            "bootstrap_95ci_distilled_student": bootstrap,
            "label_definition": (
                "Future accident within 30 minutes, same road/direction, "
                "within 300 m; no future input features."
            ),
            "score_semantics": (
                "The distilled-student output is a supervisory decision score trained "
                "on soft teacher targets. The fixed 0.60 alert threshold is applied to "
                "this score. Platt-scaled values are retained separately as prevalence-"
                "dependent probability diagnostics and are not used for alerting."
            ),
            "calibration": (
                "Platt scaling fitted on Sep-Oct validation data and saved only for "
                "probability diagnostics."
            ),
            "candidate_sampling": (
                "All positives plus stratified hard, uncertain, and random negatives."
            ),
        },
    )
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
