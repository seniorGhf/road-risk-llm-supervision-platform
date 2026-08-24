from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parents[1]
V4_DIR = WORKSPACE_DIR / "第六章" / "v4"
PREDICTION_DIR = (
    V4_DIR
    / "第二路径_风险预测"
    / "预测结果_重建_20260529"
)
FEATURE_DIR = (
    V4_DIR
    / "第二路径_风险预测"
    / "全年风险预测"
    / "输入数据"
    / "特征数据_重建_20260529"
)
ACCIDENT_XLSX = WORKSPACE_DIR / "交通事故最终表.xlsx"
CONFIG_PATH = PROJECT_DIR / "config" / "experiment_config.json"

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"
TABLE_DIR = RESULT_DIR / "tables"
FIGURE_DIR = RESULT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "report"
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"
LOG_DIR = PROJECT_DIR / "logs"

RISK_COL = "v4_model_accident_probability"


def ensure_dirs() -> None:
    for path in [
        PROCESSED_DIR,
        MODEL_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        REPORT_DIR,
        NOTEBOOK_DIR,
        LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def month_prediction_files() -> list[Path]:
    files = list(PREDICTION_DIR.glob("2025-*_第六章v4_主模型事故风险概率预测.csv"))

    def month_number(path: Path) -> int:
        return int(path.name.split("_", 1)[0].split("-")[1])

    return sorted(files, key=month_number)


def setup_plot_style() -> None:
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f343b",
            "axes.labelcolor": "#20242a",
            "xtick.color": "#30343b",
            "ytick.color": "#30343b",
            "grid.color": "#d9dde3",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.grid": True,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def stable_sample(
    frame: pd.DataFrame,
    n: int,
    random_state: int,
) -> pd.DataFrame:
    if len(frame) <= n:
        return frame
    return frame.sample(n=n, random_state=random_state, replace=False)


def safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    score = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    pred = (score >= threshold).astype(int)
    result: dict[str, float] = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, score)),
        "alert_rate": float(pred.mean()),
        "false_positive_count": int(((pred == 1) & (y_true == 0)).sum()),
        "true_positive_count": int(((pred == 1) & (y_true == 1)).sum()),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, score))
        result["pr_auc"] = float(average_precision_score(y_true, score))
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    return result


def choose_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    minimum: float = 0.6,
) -> float:
    from sklearn.metrics import f1_score

    thresholds = np.round(np.linspace(minimum, 0.95, 71), 3)
    values = [
        f1_score(y_true, np.asarray(score) >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(values))])

