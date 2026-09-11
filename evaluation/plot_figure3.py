from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.agentic_system import _run_all_tools, build_evaluation_scenarios


ACTION_ORDER = (
    ("escalate_patient", "Escalate patient"),
    ("admit_support", "Admission support"),
    ("reprioritize_queue", "Reprioritize queue"),
    ("staffing_alert", "Staffing alert"),
    ("reassign_bed", "Bed reassignment"),
    ("discharge_support", "Discharge support"),
    ("monitor", "Monitor"),
)


def boarding_level(boarding_patients: int) -> str:
    if boarding_patients <= 1:
        return "low"
    if boarding_patients <= 3:
        return "moderate"
    if boarding_patients <= 6:
        return "high"
    return "critical"


def collect_counts() -> tuple[int, dict[str, Counter[str]]]:
    scenarios = build_evaluation_scenarios(count=180, seed=2026)
    counts: dict[str, Counter[str]] = {
        "actions": Counter(),
        "flow": Counter(),
        "staffing": Counter(),
        "bed_capacity": Counter(),
        "boarding": Counter(),
    }
    for scenario in scenarios:
        counts["actions"].update(scenario.expected_actions)
        flow, _, staffing, bed = _run_all_tools(scenario.payload)
        counts["flow"].update([flow.bottleneck_level])
        counts["staffing"].update([staffing.staffing_level])
        counts["bed_capacity"].update([bed.action_window])
        counts["boarding"].update([boarding_level(scenario.payload.boarding_patients)])
    return len(scenarios), counts


def plot_expected_actions(output_path: Path, scenario_count: int, counts: Counter[str]) -> None:
    labels = [label for _, label in ACTION_ORDER]
    values = [counts[key] for key, _ in ACTION_ORDER]
    colors = ["#1553A3", "#2A75C4", "#3989D0", "#54A2DD", "#77B8E8", "#9BCDF0", "#B7B7B7"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_title("Expected action categories", weight="bold")
    ax.set_xlabel("Number of scenarios")
    ax.set_xlim(0, max(values) * 1.24)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values[::-1]):
        ax.text(
            value + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{value} ({100 * value / scenario_count:.1f}%)",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_operational_states(output_path: Path, scenario_count: int, counts: dict[str, Counter[str]]) -> None:
    dimensions = (
        (f"Flow assessment\n(n = {scenario_count})", "flow", ("low", "moderate", "high", "critical")),
        (f"Staffing\n(n = {scenario_count})", "staffing", ("adequate", None, "strained", "critical")),
        (f"Bed capacity\n(n = {scenario_count})", "bed_capacity", ("open", None, "tight", "critical")),
        (f"Boarding\n(n = {scenario_count})", "boarding", ("low", "moderate", "high", "critical")),
    )

    stack_labels = (
        "Low / adequate / open",
        "Moderate",
        "High / strained / tight",
        "Critical",
    )
    colors = ("#1553A3", "#2A75C4", "#77B8E8", "#B7B7B7")
    x_positions = range(len(dimensions))
    bottoms = [0] * len(dimensions)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for stack_index, (stack_label, color) in enumerate(zip(stack_labels, colors)):
        values = []
        for _, key, levels in dimensions:
            category = levels[stack_index]
            values.append(counts[key][category] if category is not None else 0)
        bars = ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            color=color,
            edgecolor="white",
            label=stack_label,
        )
        for bar, value, bottom in zip(bars, values, bottoms):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{value}\n({100 * value / scenario_count:.1f}%)",
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                    fontsize=8,
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title("Tool-derived operational state distributions", weight="bold")
    ax.set_ylabel("Number of scenarios")
    ax.set_xticks(list(x_positions), [label for label, _, _ in dimensions])
    ax.set_ylim(0, scenario_count * 1.10)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_counts(output_path: Path, scenario_count: int, counts: dict[str, Counter[str]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dimension", "category", "count", "percentage", "scenario_count"])
        for action_key, action_label in ACTION_ORDER:
            value = counts["actions"][action_key]
            writer.writerow(["expected_action", action_label, value, round(100 * value / scenario_count, 3), scenario_count])
        for dimension in ("flow", "staffing", "bed_capacity", "boarding"):
            for category, value in sorted(counts[dimension].items()):
                writer.writerow([dimension, category, value, round(100 * value / scenario_count, 3), scenario_count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate manuscript Figure 3 from the final benchmark generator.")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_count, counts = collect_counts()
    plot_expected_actions(output_dir / "fig03a_benchmark_expected_actions.png", scenario_count, counts["actions"])
    plot_operational_states(output_dir / "fig03b_benchmark_operational_pressure.png", scenario_count, counts)
    write_counts(output_dir / "benchmark_figure_counts.csv", scenario_count, counts)


if __name__ == "__main__":
    main()
