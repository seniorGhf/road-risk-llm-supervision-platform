from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ACCIDENT_XLSX,
    PROCESSED_DIR,
    RISK_COL,
    ensure_dirs,
    load_config,
    month_prediction_files,
    safe_numeric,
    save_json,
    stable_sample,
)


USECOLS = [
    "segment_id",
    "bin5",
    "source_month",
    "direction_code",
    "road_code",
    "road_direction",
    "stake",
    "accident_label",
    "accident_level",
    "accident_id",
    "current_risk_probability",
    "collision_risk_value",
    "v4_main_dynamic_probability",
    "v4_static_collision_probability",
    "v4_main_adaptive_dynamic_weight",
    RISK_COL,
]

MODEL_COMPONENT_COLUMNS = [
    "current_risk_probability",
    "collision_risk_value",
    "v4_main_dynamic_probability",
    "v4_static_collision_probability",
    "v4_main_adaptive_dynamic_weight",
    RISK_COL,
]


def load_accident_registry() -> pd.DataFrame:
    registry = pd.read_excel(ACCIDENT_XLSX, engine="openpyxl")
    registry["事故发生时间"] = pd.to_datetime(registry["事故发生时间"], errors="coerce")
    registry = registry[registry["事故发生时间"].dt.year.eq(2025)].copy()
    registry["direction_code"] = registry["路线方向"].map({"上行": 1, "下行": 0})
    registry["accident_stake_m"] = pd.to_numeric(
        registry["匹配桩号值(km)"], errors="coerce"
    ) * 1000.0
    registry["accident_registry_row"] = np.arange(len(registry), dtype=int)
    return registry


def add_spatial_features(day: pd.DataFrame) -> pd.DataFrame:
    day = day.copy()
    day["travel_stake"] = np.where(
        day["direction_code"].eq(1),
        day["stake"],
        -day["stake"],
    )
    day = day.sort_values(
        ["bin5", "road_code", "direction_code", "travel_stake", "segment_id"]
    ).reset_index(drop=True)
    spatial_group = day.groupby(
        ["bin5", "road_code", "direction_code"], sort=False, observed=True
    )[RISK_COL]

    upstream_columns: list[str] = []
    downstream_columns: list[str] = []
    for step in range(1, 6):
        upstream = f"upstream_risk_{step * 100}m"
        downstream = f"downstream_risk_{step * 100}m"
        day[upstream] = spatial_group.shift(step)
        day[downstream] = spatial_group.shift(-step)
        upstream_columns.append(upstream)
        downstream_columns.append(downstream)

    day["upstream_mean_300m"] = day[upstream_columns[:3]].mean(axis=1)
    day["upstream_max_300m"] = day[upstream_columns[:3]].max(axis=1)
    day["downstream_mean_300m"] = day[downstream_columns[:3]].mean(axis=1)
    day["downstream_max_300m"] = day[downstream_columns[:3]].max(axis=1)
    neighbors = upstream_columns + [RISK_COL] + downstream_columns
    day["neighborhood_mean_500m"] = day[neighbors].mean(axis=1)
    day["neighborhood_max_500m"] = day[neighbors].max(axis=1)
    day["neighborhood_std_500m"] = day[neighbors].std(axis=1).fillna(0.0)
    day["high_risk_cells_500m"] = day[neighbors].ge(0.6).sum(axis=1)
    day["spatial_gradient_600m"] = (
        day["downstream_mean_300m"] - day["upstream_mean_300m"]
    )
    day["center_neighbor_contrast"] = (
        day[RISK_COL]
        - pd.concat(
            [day["upstream_mean_300m"], day["downstream_mean_300m"]], axis=1
        ).mean(axis=1)
    )
    day["spatial_coherence_500m"] = (
        1.0
        - day["neighborhood_std_500m"]
        / (day["neighborhood_mean_500m"].abs() + 1e-6)
    ).clip(-2.0, 1.0)
    return day


def add_temporal_features(day: pd.DataFrame) -> pd.DataFrame:
    day = day.sort_values(["segment_id", "bin5"]).reset_index(drop=True)
    temporal_group = day.groupby("segment_id", sort=False, observed=True)[RISK_COL]
    for steps, minutes in [(1, 5), (2, 10), (4, 20), (6, 30)]:
        day[f"risk_lag_{minutes}m"] = temporal_group.shift(steps)

    rolling = temporal_group.rolling(window=6, min_periods=2)
    day["risk_mean_past_30m"] = (
        rolling.mean().reset_index(level=0, drop=True).reindex(day.index)
    )
    day["risk_max_past_30m"] = (
        rolling.max().reset_index(level=0, drop=True).reindex(day.index)
    )
    day["risk_std_past_30m"] = (
        rolling.std().reset_index(level=0, drop=True).reindex(day.index).fillna(0.0)
    )
    day["risk_change_5m"] = day[RISK_COL] - day["risk_lag_5m"]
    day["risk_change_30m"] = day[RISK_COL] - day["risk_lag_30m"]
    day["risk_acceleration"] = (
        day["risk_change_5m"]
        - (day["risk_lag_5m"] - day["risk_lag_10m"])
    )
    return day.sort_values(
        ["bin5", "road_code", "direction_code", "travel_stake"]
    ).reset_index(drop=True)


def label_future_events(
    day: pd.DataFrame,
    horizon_minutes: int,
    radius_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = day.copy()
    day["target_event_30m"] = 0
    day["target_event_id"] = ""
    day["target_event_level"] = 0
    day["lead_time_min"] = np.nan
    day["event_guard"] = 0

    events = day[pd.to_numeric(day["accident_label"], errors="coerce").fillna(0).gt(0)].copy()
    events = events.sort_values("bin5").drop_duplicates("accident_id", keep="first")
    if events.empty:
        return day, events

    for event in events.itertuples(index=False):
        same_corridor = (
            day["road_code"].eq(event.road_code)
            & day["direction_code"].eq(event.direction_code)
            & day["stake"].sub(float(event.stake)).abs().le(radius_m)
        )
        delta_minutes = (event.bin5 - day["bin5"]).dt.total_seconds() / 60.0
        positive = same_corridor & delta_minutes.gt(0) & delta_minutes.le(horizon_minutes)
        guard = (
            same_corridor
            & delta_minutes.ge(-horizon_minutes)
            & delta_minutes.le(horizon_minutes)
        )
        day.loc[guard, "event_guard"] = 1
        if not positive.any():
            continue
        current_lead = day.loc[positive, "lead_time_min"]
        replace = current_lead.isna() | delta_minutes.loc[positive].lt(current_lead)
        indices = current_lead.index[replace]
        day.loc[indices, "target_event_30m"] = 1
        day.loc[indices, "target_event_id"] = str(event.accident_id)
        day.loc[indices, "target_event_level"] = int(event.accident_level)
        day.loc[indices, "lead_time_min"] = delta_minutes.loc[indices]
    return day, events


def select_candidate_rows(
    day: pd.DataFrame,
    config: dict,
    day_seed: int,
) -> pd.DataFrame:
    positives = day[day["target_event_30m"].eq(1)]
    eligible_negatives = day[
        day["target_event_30m"].eq(0)
        & day["event_guard"].eq(0)
        & pd.to_numeric(day["accident_label"], errors="coerce").fillna(0).eq(0)
    ]
    hard = eligible_negatives[eligible_negatives[RISK_COL].ge(config["hard_negative_threshold"])]
    uncertain = eligible_negatives[
        eligible_negatives[RISK_COL].ge(0.4)
        & eligible_negatives[RISK_COL].lt(config["hard_negative_threshold"])
    ]
    random_pool = eligible_negatives[eligible_negatives[RISK_COL].lt(0.4)]
    hard = stable_sample(
        hard,
        int(config["hard_negatives_per_day"]),
        day_seed + 1,
    )
    uncertain = stable_sample(
        uncertain,
        int(config["uncertain_negatives_per_day"]),
        day_seed + 2,
    )
    random_negative = stable_sample(
        random_pool,
        int(config["random_negatives_per_day"]),
        day_seed + 3,
    )
    selected = pd.concat(
        [positives, hard, uncertain, random_negative],
        ignore_index=True,
    )
    selected["candidate_source"] = np.select(
        [
            selected["target_event_30m"].eq(1),
            selected[RISK_COL].ge(config["hard_negative_threshold"]),
            selected[RISK_COL].ge(0.4),
        ],
        ["future_event", "hard_negative", "uncertain_negative"],
        default="random_negative",
    )
    return selected.sort_values(["bin5", "road_code", "direction_code", "stake"])


def process_day(
    day: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = day.copy()
    day = safe_numeric(
        day,
        [
            "direction_code",
            "stake",
            "accident_label",
            "accident_level",
            *MODEL_COMPONENT_COLUMNS,
        ],
    )
    day = day.dropna(
        subset=["segment_id", "bin5", "road_code", "direction_code", "stake", RISK_COL]
    )
    if day.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    day["direction_code"] = day["direction_code"].astype(int)
    day[RISK_COL] = day[RISK_COL].clip(0.0, 1.0)
    day["accident_label"] = day["accident_label"].fillna(0).astype(int)
    day["accident_level"] = day["accident_level"].fillna(0).astype(int)
    day["accident_id"] = day["accident_id"].fillna("").astype(str)
    invalid_event_id = (
        day["accident_label"].gt(0)
        & day["accident_id"].str.strip().str.lower().isin(
            {"", "0", "0.0", "nan", "none", "null"}
        )
    )
    if invalid_event_id.any():
        day.loc[invalid_event_id, "accident_label"] = 0
        day.loc[invalid_event_id, "accident_level"] = 0

    grain_duplicates = day.duplicated(["segment_id", "bin5"]).sum()
    if grain_duplicates:
        day = day.sort_values(RISK_COL).drop_duplicates(
            ["segment_id", "bin5"], keep="last"
        )

    day = add_spatial_features(day)
    day = add_temporal_features(day)
    day, events = label_future_events(
        day,
        int(config["prediction_horizon_minutes"]),
        float(config["positive_spatial_radius_m"]),
    )
    date_value = pd.Timestamp(day["bin5"].min()).date()
    day_seed = int(pd.Timestamp(date_value).strftime("%Y%m%d")) + int(
        config["random_seed"]
    )
    candidates = select_candidate_rows(day, config, day_seed)

    topology_columns = [
        "segment_id",
        "road_code",
        "direction_code",
        "road_direction",
        "stake",
        "travel_stake",
    ]
    topology = (
        day[topology_columns]
        .drop_duplicates("segment_id")
        .sort_values(["road_code", "direction_code", "travel_stake"])
    )
    return candidates, events, topology


def write_month_outputs(
    path: Path,
    config: dict,
    chunksize: int,
) -> dict:
    start = time.perf_counter()
    month_name = path.name.split("_", 1)[0]
    candidate_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    topology_parts: list[pd.DataFrame] = []
    carry = pd.DataFrame()
    last_timestamp: pd.Timestamp | None = None
    raw_rows = 0
    processed_days = 0
    monotonic_violations = 0
    off_month_rows = 0
    invalid_event_labels_excluded = 0
    expected_period = pd.Period(month_name, freq="M")

    reader = pd.read_csv(
        path,
        usecols=lambda column: column in USECOLS,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8-sig",
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        raw_rows += len(chunk)
        chunk["bin5"] = pd.to_datetime(
            chunk["bin5"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        chunk = chunk.dropna(subset=["bin5"])
        in_expected_month = chunk["bin5"].dt.to_period("M").eq(expected_period)
        off_month_rows += int((~in_expected_month).sum())
        chunk = chunk[in_expected_month].copy()
        label_numeric = pd.to_numeric(chunk["accident_label"], errors="coerce").fillna(0)
        id_normalized = chunk["accident_id"].fillna("").astype(str).str.strip().str.lower()
        invalid_event_labels_excluded += int(
            (label_numeric.gt(0) & id_normalized.isin({"", "0", "0.0", "nan", "none", "null"})).sum()
        )
        if chunk.empty:
            continue
        if last_timestamp is not None and chunk["bin5"].min() < last_timestamp:
            monotonic_violations += 1
        last_timestamp = chunk["bin5"].max()
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame()
        final_date = chunk["bin5"].dt.date.max()
        complete = chunk[chunk["bin5"].dt.date.ne(final_date)]
        carry = chunk[chunk["bin5"].dt.date.eq(final_date)].copy()
        for _, day in complete.groupby(complete["bin5"].dt.date, sort=True):
            candidates, events, topology = process_day(day, config)
            if not candidates.empty:
                candidate_parts.append(candidates)
            if not events.empty:
                event_parts.append(events)
            if not topology.empty:
                topology_parts.append(topology)
            processed_days += 1
        if chunk_number % 5 == 0:
            print(
                f"{month_name}: chunks={chunk_number}, raw_rows={raw_rows:,}, "
                f"days={processed_days}, candidates={sum(map(len, candidate_parts)):,}",
                flush=True,
            )

    if not carry.empty:
        candidates, events, topology = process_day(carry, config)
        if not candidates.empty:
            candidate_parts.append(candidates)
        if not events.empty:
            event_parts.append(events)
        if not topology.empty:
            topology_parts.append(topology)
        processed_days += 1

    if not candidate_parts or not topology_parts:
        raise ValueError(f"{month_name} produced no valid candidate/topology data.")
    candidates_month = pd.concat(candidate_parts, ignore_index=True)
    if event_parts:
        events_month = (
            pd.concat(event_parts, ignore_index=True)
            .sort_values("bin5")
            .drop_duplicates("accident_id", keep="first")
        )
    else:
        events_month = pd.DataFrame(columns=USECOLS)
    topology_month = (
        pd.concat(topology_parts, ignore_index=True)
        .drop_duplicates("segment_id")
        .sort_values(["road_code", "direction_code", "travel_stake"])
    )
    candidate_path = PROCESSED_DIR / f"{month_name}_causal_candidates.csv"
    event_path = PROCESSED_DIR / f"{month_name}_mapped_events.csv"
    topology_path = PROCESSED_DIR / f"{month_name}_segment_topology.csv"
    candidates_month.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    events_month.to_csv(event_path, index=False, encoding="utf-8-sig")
    topology_month.to_csv(topology_path, index=False, encoding="utf-8-sig")

    return {
        "month": month_name,
        "source": str(path),
        "raw_rows": int(raw_rows),
        "processed_days": int(processed_days),
        "candidate_rows": int(len(candidates_month)),
        "positive_candidate_rows": int(candidates_month["target_event_30m"].sum()),
        "event_count": int(events_month["accident_id"].nunique()),
        "segment_count": int(topology_month["segment_id"].nunique()),
        "grain_duplicate_rows_in_candidates": int(
            candidates_month.duplicated(["segment_id", "bin5"]).sum()
        ),
        "monotonic_chunk_violations": int(monotonic_violations),
        "off_month_rows_excluded": int(off_month_rows),
        "invalid_event_labels_excluded": int(invalid_event_labels_excluded),
        "seconds": round(time.perf_counter() - start, 3),
        "candidate_file": str(candidate_path),
        "event_file": str(event_path),
        "topology_file": str(topology_path),
    }


def combine_outputs(summaries: list[dict]) -> None:
    candidate_files = [Path(item["candidate_file"]) for item in summaries]
    event_files = [Path(item["event_file"]) for item in summaries]
    topology_files = [Path(item["topology_file"]) for item in summaries]

    candidates = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in candidate_files],
        ignore_index=True,
    )
    candidates["bin5"] = pd.to_datetime(candidates["bin5"], errors="coerce")
    candidates = candidates.sort_values(["bin5", "road_code", "direction_code", "stake"])
    candidates.to_csv(
        PROCESSED_DIR / "causal_candidate_dataset_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    events = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in event_files],
        ignore_index=True,
    )
    events["bin5"] = pd.to_datetime(events["bin5"], errors="coerce")
    events = events.sort_values("bin5").drop_duplicates("accident_id", keep="first")
    events.to_csv(
        PROCESSED_DIR / "mapped_accident_events_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topology = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in topology_files],
        ignore_index=True,
    )
    topology = topology.drop_duplicates("segment_id").sort_values(
        ["road_code", "direction_code", "travel_stake"]
    )
    group = topology.groupby(["road_code", "direction_code"], sort=False, observed=True)
    topology["upstream_segment_id"] = group["segment_id"].shift(1)
    topology["downstream_segment_id"] = group["segment_id"].shift(-1)
    topology["upstream_stake"] = group["stake"].shift(1)
    topology["downstream_stake"] = group["stake"].shift(-1)
    topology.to_csv(
        PROCESSED_DIR / "directed_segment_adjacency_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    registry = load_accident_registry()
    registry.to_csv(
        PROCESSED_DIR / "accident_registry_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_rows = sum(int(item["raw_rows"]) for item in summaries)
    profile = {
        "source_prediction_rows": source_rows,
        "candidate_rows": int(len(candidates)),
        "candidate_positive_rows": int(candidates["target_event_30m"].sum()),
        "candidate_positive_rate": float(candidates["target_event_30m"].mean()),
        "mapped_event_count": int(events["accident_id"].nunique()),
        "registry_event_count": int(len(registry)),
        "topology_segment_count": int(topology["segment_id"].nunique()),
        "candidate_grain_duplicates": int(
            candidates.duplicated(["segment_id", "bin5"]).sum()
        ),
        "candidate_null_rates": candidates[
            [
                RISK_COL,
                "upstream_mean_300m",
                "downstream_mean_300m",
                "risk_lag_30m",
                "target_event_30m",
            ]
        ]
        .isna()
        .mean()
        .to_dict(),
        "monthly_summaries": summaries,
        "label_definition": (
            "Accident occurs in (t, t+30 min] on the same road and direction "
            "within 300 m; features use data at or before t only."
        ),
        "sampling_caveat": (
            "All positives are retained; negatives are stratified samples. "
            "Precision therefore describes the candidate-review population, "
            "not the unsampled full-network prevalence."
        ),
    }
    save_json(PROCESSED_DIR / "data_quality_and_build_summary.json", profile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--months",
        default="all",
        help="all or comma-separated month numbers, e.g. 1,2,3",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    config = load_config()
    args = parse_args()
    files = month_prediction_files()
    if args.months != "all":
        selected = {int(item.strip()) for item in args.months.split(",")}
        files = [
            path
            for path in files
            if int(path.name.split("_", 1)[0].split("-")[1]) in selected
        ]
    if not files:
        raise FileNotFoundError("No monthly prediction files selected.")
    summaries = []
    for path in files:
        print(f"Processing {path.name}", flush=True)
        summaries.append(write_month_outputs(path, config, args.chunksize))
    if args.months == "all":
        combine_outputs(summaries)
    save_json(PROCESSED_DIR / "last_build_run.json", summaries)
    print("Causal candidate data build completed.", flush=True)


if __name__ == "__main__":
    main()
