from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = ("precision", "recall", "avg_action_quality", "avg_weighted_error_cost")

METRIC_TITLES = {
    "precision": "Precision",
    "recall": "Recall",
    "avg_action_quality": "Action Quality",
    "avg_weighted_error_cost": "Weighted Recommendation-Error Cost",
}

CONFIGURATION_LABELS = {
    "full_dynamic": "Full dynamic",
    "all_tools": "All tools",
    "without_safety_validation": "Without safety\nvalidation",
    "without_follow_up_tracking": "Without follow-up\ntracking",
    "non_agentic": "Non-agentic",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 4 from checked-in experiment summary data.")
    parser.add_argument("summary_csv")
    parser.add_argument("--output", default="figure4.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to generate Figure 4.") from exc

    with Path(args.summary_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    configurations = list(dict.fromkeys(row["configuration"] for row in rows))
    lookup = {(row["configuration"], row["metric"]): row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, metric in zip(axes.flat, METRICS):
        values = [float(lookup[(name, metric)]["value"]) for name in configurations]
        lower = [values[i] - float(lookup[(name, metric)]["lower"]) for i, name in enumerate(configurations)]
        upper = [float(lookup[(name, metric)]["upper"]) - values[i] for i, name in enumerate(configurations)]
        labels = [CONFIGURATION_LABELS.get(name, name.replace("_", " ").title()) for name in configurations]
        colors = ["#1553A3", "#3989D0", "#77B8E8", "#9BCDF0", "#B7B7B7"]
        axis.bar(labels, values, yerr=[lower, upper], capsize=4, color=colors[: len(labels)])
        axis.set_title(METRIC_TITLES[metric])
        axis.grid(axis="y", linestyle="--", alpha=0.30)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", rotation=15, labelsize=8)
    figure.tight_layout()
    figure.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
