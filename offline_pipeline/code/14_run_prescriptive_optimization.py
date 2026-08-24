from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from common import MODEL_DIR, PROCESSED_DIR, PROJECT_DIR, TABLE_DIR, load_config, save_json


CONFIG = PROJECT_DIR / "config" / "prescriptive_optimization.json"
OUTPUT_DIR = PROJECT_DIR / "prescriptive_optimization"
TRAINING_SCRIPT = Path(__file__).with_name("02_train_confirmation_models.py")

ACTIONS = ["warning", "patrol", "emergency"]


def aggregate_sites(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = frame[
        frame[["score_stage1_risk", "score_distilled_student"]]
        .max(axis=1)
        .ge(config["candidate_score_minimum"])
    ].copy()
    frame["decision_window"] = frame["bin5"].dt.floor(
        f"{config['update_window_minutes']}min"
    )
    frame["spatial_zone"] = (
        np.floor(frame["stake"] / config["spatial_zone_m"])
        * config["spatial_zone_m"]
    ).astype(int)
    group_keys = [
        "decision_window",
        "road_code",
        "direction_code",
        "spatial_zone",
    ]

    def event_ids(series: pd.Series) -> str:
        valid = sorted(
            {
                str(value)
                for value in series
                if pd.notna(value) and str(value).strip() not in {"", "0"}
            }
        )
        return "|".join(valid)

    sites = (
        frame.groupby(group_keys, observed=True)
        .agg(
            segment_count=("segment_id", "nunique"),
            representative_segment=("segment_id", "first"),
            stake_m=("stake", "mean"),
            stage1_score=("score_stage1_risk", "max"),
            student_score=("score_distilled_student", "max"),
            actual_positive=("target_event_30m", "max"),
            event_ids=("target_event_id", event_ids),
            maximum_lead_time_min=("lead_time_min", "max"),
        )
        .reset_index()
    )
    sites["site_id"] = [f"SITE_{index + 1:06d}" for index in range(len(sites))]
    return sites


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "confirmation_training_for_site_calibration", TRAINING_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_validation_rows() -> pd.DataFrame:
    module = load_training_module()
    data = pd.read_csv(
        PROCESSED_DIR / "causal_candidate_dataset_2025.csv", low_memory=False
    )
    data = module.add_context_features(data)
    validation = module.temporal_split(data, load_config())["validation"].copy()
    student = joblib.load(MODEL_DIR / "distilled_student.joblib")
    validation["score_stage1_risk"] = validation[
        "v4_model_accident_probability"
    ].to_numpy(dtype=float)
    validation["score_distilled_student"] = np.clip(
        student.predict(validation[module.STUDENT_FEATURES]), 0.0, 1.0
    )
    return validation


def calibrate_site_probability(
    test_sites: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, dict]:
    validation_sites = aggregate_sites(score_validation_rows(), config)
    eps = 1e-6

    def logit(values: pd.Series) -> np.ndarray:
        clipped = np.clip(values.to_numpy(dtype=float), eps, 1.0 - eps)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    calibrator = LogisticRegression(C=10.0, max_iter=1000)
    calibrator.fit(
        logit(validation_sites["student_score"]),
        validation_sites["actual_positive"].to_numpy(dtype=int),
    )
    validation_probability = calibrator.predict_proba(
        logit(validation_sites["student_score"])
    )[:, 1]
    test_sites = test_sites.copy()
    test_sites["calibrated_probability"] = calibrator.predict_proba(
        logit(test_sites["student_score"])
    )[:, 1]
    joblib.dump(
        calibrator,
        MODEL_DIR / "site_level_probability_calibrator.joblib",
        compress=3,
    )
    summary = {
        "validation_site_count": int(len(validation_sites)),
        "validation_positive_sites": int(
            validation_sites["actual_positive"].sum()
        ),
        "validation_positive_rate": float(
            validation_sites["actual_positive"].mean()
        ),
        "validation_raw_score_mean": float(
            validation_sites["student_score"].mean()
        ),
        "validation_calibrated_probability_mean": float(
            validation_probability.mean()
        ),
        "validation_raw_brier": float(
            brier_score_loss(
                validation_sites["actual_positive"],
                validation_sites["student_score"],
            )
        ),
        "validation_calibrated_brier": float(
            brier_score_loss(
                validation_sites["actual_positive"],
                validation_probability,
            )
        ),
        "test_site_count": int(len(test_sites)),
        "test_raw_score_mean": float(test_sites["student_score"].mean()),
        "test_calibrated_probability_mean": float(
            test_sites["calibrated_probability"].mean()
        ),
        "fit_boundary": (
            "The site-level Platt calibrator is fitted only on Sep-Oct "
            "validation sites after applying the same candidate, temporal, and "
            "spatial aggregation rules. Nov-Dec labels are not used for fitting."
        ),
    }
    return test_sites, summary


def build_sites(config: dict) -> tuple[pd.DataFrame, dict]:
    prediction_path = TABLE_DIR / "confirmation_model_test_predictions.csv"
    frame = pd.read_csv(prediction_path, parse_dates=["bin5"], low_memory=False)
    sites = aggregate_sites(frame, config)
    return calibrate_site_probability(sites, config)


def incremental_coefficients(
    probability: np.ndarray, scenario: dict, config: dict
) -> np.ndarray:
    coefficients = []
    loss = float(scenario["accident_loss"])
    for action in ACTIONS:
        mitigation = float(config["incremental_mitigation"][action])
        action_cost = float(scenario["action_cost"][action])
        false_penalty = float(scenario["false_alarm_penalty"][action])
        coefficient = (
            action_cost
            + (1.0 - probability) * false_penalty
            - probability * loss * mitigation
        )
        coefficients.append(coefficient)
    return np.concatenate(coefficients)


def solve_binary_program(
    window: pd.DataFrame,
    scenario: dict,
    config: dict,
    probability_column: str,
) -> np.ndarray:
    n = len(window)
    p = window[probability_column].to_numpy(dtype=float)
    coefficients = incremental_coefficients(p, scenario, config).reshape(3, n)
    warning_cost, patrol_cost, emergency_cost = coefficients
    warning_capacity = int(config["capacities_per_window"]["warning"])
    patrol_capacity = int(config["capacities_per_window"]["patrol"])
    emergency_capacity = int(config["capacities_per_window"]["emergency"])
    max_per_road = int(config["maximum_patrols_per_road"])
    minimum_separation = float(config["minimum_patrol_separation_m"])

    roads = window["road_code"].astype(str).to_numpy()
    directions = window["direction_code"].to_numpy()
    zones = window["spatial_zone"].to_numpy(dtype=float)

    def valid_patrol_subset(subset: tuple[int, ...]) -> bool:
        road_counts: dict[str, int] = defaultdict(int)
        for position in subset:
            road_counts[roads[position]] += 1
            if road_counts[roads[position]] > max_per_road:
                return False
        for left_index, left in enumerate(subset):
            for right in subset[left_index + 1 :]:
                if (
                    roads[left] == roads[right]
                    and directions[left] == directions[right]
                    and abs(zones[left] - zones[right]) < minimum_separation
                ):
                    return False
        return True

    best_cost = 0.0
    best_decision = np.zeros((n, 3), dtype=int)
    positions = tuple(range(n))
    # Exact dominance pruning: if neither patrol nor patrol+emergency has a
    # negative incremental cost, upgrading that site beyond warning can never
    # improve a feasible solution.
    patrol_positions = tuple(
        position
        for position in positions
        if min(
            patrol_cost[position],
            patrol_cost[position] + emergency_cost[position],
        )
        < 0
    )
    for patrol_count in range(min(patrol_capacity, len(patrol_positions)) + 1):
        for patrol_subset in combinations(patrol_positions, patrol_count):
            if not valid_patrol_subset(patrol_subset):
                continue
            emergency_choices: tuple[int | None, ...]
            if emergency_capacity > 0:
                emergency_choices = (None, *patrol_subset)
            else:
                emergency_choices = (None,)
            for emergency_position in emergency_choices:
                remaining_warning_capacity = warning_capacity - patrol_count
                patrol_set = set(patrol_subset)
                warning_candidates = [
                    position
                    for position in positions
                    if position not in patrol_set and warning_cost[position] < 0
                ]
                warning_candidates.sort(key=lambda position: warning_cost[position])
                warning_only = warning_candidates[:remaining_warning_capacity]
                cost = float(
                    sum(
                        warning_cost[position] + patrol_cost[position]
                        for position in patrol_subset
                    )
                    + sum(warning_cost[position] for position in warning_only)
                )
                if emergency_position is not None:
                    cost += float(emergency_cost[emergency_position])
                if cost >= best_cost:
                    continue
                decision = np.zeros((n, 3), dtype=int)
                decision[list(patrol_subset), 0] = 1
                decision[list(patrol_subset), 1] = 1
                decision[warning_only, 0] = 1
                if emergency_position is not None:
                    decision[emergency_position, 2] = 1
                best_cost = cost
                best_decision = decision
    return best_decision


def ranking_policy(
    window: pd.DataFrame, score_column: str, config: dict
) -> np.ndarray:
    n = len(window)
    decision = np.zeros((n, 3), dtype=int)
    score = window[score_column].to_numpy(dtype=float)
    eligible = np.flatnonzero(score >= float(config["ranking_alert_threshold"]))
    order = eligible[np.argsort(-score[eligible])]
    warning = list(order[: config["capacities_per_window"]["warning"]])
    patrol = []
    road_counts: dict[str, int] = defaultdict(int)
    for position in order:
        road = str(window.iloc[position]["road_code"])
        if position not in warning:
            continue
        if road_counts[road] >= config["maximum_patrols_per_road"]:
            continue
        patrol.append(position)
        road_counts[road] += 1
        if len(patrol) >= config["capacities_per_window"]["patrol"]:
            break
    emergency = patrol[: config["capacities_per_window"]["emergency"]]
    decision[warning, 0] = 1
    decision[patrol, 1] = 1
    decision[emergency, 2] = 1
    return decision


def greedy_policy(window: pd.DataFrame, scenario: dict, config: dict) -> np.ndarray:
    n = len(window)
    p = window["calibrated_probability"].to_numpy(dtype=float)
    coefficients = incremental_coefficients(p, scenario, config).reshape(3, n)
    decision = np.zeros((n, 3), dtype=int)
    road_counts: dict[str, int] = defaultdict(int)

    warning_order = np.argsort(coefficients[0])
    for position in warning_order:
        if coefficients[0, position] >= 0:
            continue
        if decision[:, 0].sum() >= config["capacities_per_window"]["warning"]:
            break
        decision[position, 0] = 1

    patrol_order = np.argsort(coefficients[1])
    for position in patrol_order:
        road = str(window.iloc[position]["road_code"])
        if coefficients[1, position] >= 0 or decision[position, 0] == 0:
            continue
        if decision[:, 1].sum() >= config["capacities_per_window"]["patrol"]:
            break
        if road_counts[road] >= config["maximum_patrols_per_road"]:
            continue
        decision[position, 1] = 1
        road_counts[road] += 1

    emergency_order = np.argsort(coefficients[2])
    for position in emergency_order:
        if coefficients[2, position] >= 0 or decision[position, 1] == 0:
            continue
        if decision[:, 2].sum() >= config["capacities_per_window"]["emergency"]:
            break
        decision[position, 2] = 1
    return decision


def no_action(window: pd.DataFrame, *_args) -> np.ndarray:
    return np.zeros((len(window), 3), dtype=int)


def realized_event_cost(
    decisions: pd.DataFrame,
    sites: pd.DataFrame,
    scenario: dict,
    config: dict,
) -> float:
    merged = sites.merge(decisions, on="site_id", validate="one_to_one")
    y = merged["actual_positive"].to_numpy(dtype=float)
    decision = merged[ACTIONS].to_numpy(dtype=int)
    mitigation = (
        decision
        * np.asarray(
            [config["incremental_mitigation"][action] for action in ACTIONS]
        )
    ).sum(axis=1)
    action_cost = np.zeros(len(merged))
    false_alarm_cost = np.zeros(len(merged))
    for action_index, action in enumerate(ACTIONS):
        action_cost += (
            decision[:, action_index] * float(scenario["action_cost"][action])
        )
        false_alarm_cost += (
            decision[:, action_index]
            * (1.0 - y)
            * float(scenario["false_alarm_penalty"][action])
        )
    all_events = sorted(
        {
            event
            for value in merged.loc[merged["actual_positive"].eq(1), "event_ids"]
            for event in str(value).split("|")
            if event
        }
    )
    residual_event_loss = 0.0
    for event in all_events:
        event_mask = merged["event_ids"].str.split("|").apply(
            lambda values: event in values
        )
        maximum_mitigation = float(mitigation[event_mask].max())
        residual_event_loss += float(scenario["accident_loss"]) * (
            1.0 - maximum_mitigation
        )
    return float(residual_event_loss + action_cost.sum() + false_alarm_cost.sum())


def evaluate(
    decisions: pd.DataFrame, sites: pd.DataFrame, scenario_name: str, strategy: str
) -> dict:
    merged = sites.merge(decisions, on="site_id", validate="one_to_one")
    positive = merged[merged["actual_positive"].eq(1)].copy()
    all_events = {
        event
        for value in positive["event_ids"]
        for event in str(value).split("|")
        if event
    }
    escalated = positive[positive["patrol"].eq(1)]
    detected_events = {
        event
        for value in escalated["event_ids"]
        for event in str(value).split("|")
        if event
    }
    false_patrols = int(
        ((merged["patrol"].eq(1)) & (merged["actual_positive"].eq(0))).sum()
    )
    true_patrols = int(
        ((merged["patrol"].eq(1)) & (merged["actual_positive"].eq(1))).sum()
    )
    precision = true_patrols / max(true_patrols + false_patrols, 1)
    return {
        "scenario": scenario_name,
        "strategy": strategy,
        "site_count": int(len(merged)),
        "event_count": int(len(all_events)),
        "detected_event_count": int(len(detected_events)),
        "event_recall": len(detected_events) / max(len(all_events), 1),
        "patrol_precision": precision,
        "warning_actions": int(merged["warning"].sum()),
        "patrol_actions": int(merged["patrol"].sum()),
        "emergency_actions": int(merged["emergency"].sum()),
        "false_patrol_actions": false_patrols,
        "true_patrol_actions": true_patrols,
        "patrol_actions_per_day": float(
            merged["patrol"].sum()
            / max(merged["decision_window"].dt.date.nunique(), 1)
        ),
        "mean_maximum_lead_time_min": float(
            escalated.groupby("event_ids", observed=True)[
                "maximum_lead_time_min"
            ].max().mean()
        )
        if len(escalated)
        else np.nan,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    sites, site_calibration = build_sites(config)
    sites["robust_probability"] = np.clip(
        sites["calibrated_probability"]
        - float(config["robust_probability_radius"]),
        0.0,
        1.0,
    )
    sites.to_csv(
        OUTPUT_DIR / "watchlist_sites_test.csv", index=False, encoding="utf-8-sig"
    )

    all_decisions = []
    metric_rows = []
    cost_lookup = {}
    for scenario_name, scenario in config["scenarios"].items():
        strategy_frames = {
            "no_action": [],
            "stage1_priority": [],
            "student_priority": [],
            "greedy_expected_utility": [],
            "binary_program_expected_utility": [],
            "robust_binary_program": [],
        }
        for _, window in sites.groupby("decision_window", observed=True, sort=True):
            window = window.reset_index(drop=True)
            decisions = {
                "no_action": no_action(window),
                "stage1_priority": ranking_policy(window, "stage1_score", config),
                "student_priority": ranking_policy(window, "student_score", config),
                "greedy_expected_utility": greedy_policy(window, scenario, config),
                "binary_program_expected_utility": solve_binary_program(
                    window, scenario, config, "calibrated_probability"
                ),
                "robust_binary_program": solve_binary_program(
                    window, scenario, config, "robust_probability"
                ),
            }
            for strategy, matrix in decisions.items():
                decision_frame = pd.DataFrame(
                    {
                        "site_id": window["site_id"],
                        "scenario": scenario_name,
                        "strategy": strategy,
                        "warning": matrix[:, 0],
                        "patrol": matrix[:, 1],
                        "emergency": matrix[:, 2],
                    }
                )
                strategy_frames[strategy].append(decision_frame)
        for strategy, frames in strategy_frames.items():
            decision_frame = pd.concat(frames, ignore_index=True)
            all_decisions.append(decision_frame)
            metrics = evaluate(decision_frame, sites, scenario_name, strategy)
            realized = realized_event_cost(
                decision_frame, sites, scenario, config
            )
            metrics["realized_normalized_event_cost"] = float(realized)
            metric_rows.append(metrics)
            cost_lookup[(scenario_name, strategy)] = realized

    decisions = pd.concat(all_decisions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    for scenario_name in config["scenarios"]:
        baseline = cost_lookup[(scenario_name, "no_action")]
        mask = metrics["scenario"].eq(scenario_name)
        metrics.loc[mask, "event_cost_reduction_vs_no_action"] = (
            baseline
            - metrics.loc[mask, "realized_normalized_event_cost"]
        ) / max(baseline, 1.0)
    decisions.to_csv(
        OUTPUT_DIR / "optimization_decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        OUTPUT_DIR / "optimization_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "config": config,
        "watchlist_site_count": int(len(sites)),
        "decision_window_count": int(sites["decision_window"].nunique()),
        "formal_event_count": int(
            len(
                {
                    event
                    for value in sites.loc[
                        sites["actual_positive"].eq(1), "event_ids"
                    ]
                    for event in str(value).split("|")
                    if event
                }
            )
        ),
        "best_by_scenario": {
            scenario: metrics[metrics["scenario"].eq(scenario)]
            .sort_values("realized_normalized_event_cost")
            .iloc[0]
            .to_dict()
            for scenario in config["scenarios"]
        },
        "interpretation_boundary": config["interpretation_boundary"],
        "site_probability_calibration": site_calibration,
        "solver": (
            "Exact enumeration of the nested binary warning-patrol-emergency "
            "program under small operational capacities; equivalent feasible "
            "set to the stated binary formulation."
        ),
    }
    save_json(OUTPUT_DIR / "optimization_summary.json", summary)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
