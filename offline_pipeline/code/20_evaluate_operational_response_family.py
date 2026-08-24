from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from common import PROJECT_DIR, TABLE_DIR, save_json, setup_plot_style


OUTPUT_DIR = PROJECT_DIR / "operational_response_family"
FAMILY_MAP = {
    "碰撞": "Lane-conflict",
    "追尾": "Lane-conflict",
    "刮擦": "Lane-conflict",
    "翻车": "Rollover",
    "失火": "Fire",
    "其他": "Other",
}
ACTION_MAP = {
    "Lane-conflict": {"VMS_WARNING", "VERIFY", "QUEUE_TAIL", "LANE_CONTROL"},
    "Rollover": {"VMS_WARNING", "VERIFY", "HEAVY_RESCUE", "LANE_ISOLATION"},
    "Fire": {"VMS_WARNING", "VERIFY", "FIRE_RESPONSE", "TRAFFIC_STOP"},
    "Other": {"VMS_WARNING", "VERIFY"},
}


def set_f1(left: set[str], right: set[str]) -> float:
    overlap = len(left & right)
    precision = overlap / max(len(left), 1)
    recall = overlap / max(len(right), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def main() -> None:
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(
        TABLE_DIR / "accident_type_test_predictions.csv", low_memory=False
    )
    predictions["true_response_family"] = predictions["事故类型_建模"].map(
        FAMILY_MAP
    )
    predictions["predicted_response_family"] = predictions["predicted_type"].map(
        FAMILY_MAP
    )
    truth = predictions["true_response_family"]
    predicted = predictions["predicted_response_family"]
    labels = ["Lane-conflict", "Rollover", "Fire", "Other"]
    matrix = confusion_matrix(truth, predicted, labels=labels)
    predictions["policy_action_f1"] = [
        set_f1(ACTION_MAP[prediction], ACTION_MAP[actual])
        for actual, prediction in zip(truth, predicted)
    ]
    metrics = {
        "test_events": int(len(predictions)),
        "six_class_type_accuracy": float(
            (predictions["事故类型_建模"] == predictions["predicted_type"]).mean()
        ),
        "operational_family_accuracy": float(accuracy_score(truth, predicted)),
        "operational_family_balanced_accuracy": float(
            balanced_accuracy_score(truth, predicted)
        ),
        "operational_family_macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "mean_policy_action_f1": float(predictions["policy_action_f1"].mean()),
        "claim_boundary": (
            "Operational response families are a policy grouping of type "
            "hypotheses, not improved six-class accident-type identification."
        ),
    }
    predictions.to_csv(
        OUTPUT_DIR / "operational_response_family_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        OUTPUT_DIR / "operational_response_family_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    save_json(OUTPUT_DIR / "operational_response_family_summary.json", metrics)

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Predicted response family")
    ax.set_ylabel("Observed response family")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white"
                if matrix[row, column] > matrix.max() / 2
                else "#111827",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Events")
    fig.savefig(
        OUTPUT_DIR / "Operational_Response_Family_Confusion_Matrix.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "Operational_Response_Family_Confusion_Matrix.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
