from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from app.evaluation import compact_summary, write_evaluation_outputs
from app.mimic_iv_ed import evaluate_mimic_patient_risk
from app.nhamcs_ed import evaluate_nhamcs_ed, summarize_nhamcs_dataset
from app.orchestrator import evaluate_all_systems


def run_demo() -> None:
    sample_path = Path("demo_data/sample_case.json")
    payload = sample_path.read_text(encoding="utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8000/evaluate",
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))
        print(json.dumps(body, indent=2))


def run_evaluation() -> None:
    comparison = evaluate_all_systems()
    print(json.dumps(comparison, indent=2, default=list))


def run_compact_evaluation() -> None:
    comparison = evaluate_all_systems()
    print(json.dumps(compact_summary(comparison), indent=2))


def save_evaluation_outputs(output_dir: str) -> None:
    comparison = evaluate_all_systems()
    outputs = write_evaluation_outputs(comparison, output_dir)
    print(json.dumps(outputs, indent=2))


def run_mimic_evaluation(data_dir: str, label_name: str, limit: int | None) -> None:
    metrics = evaluate_mimic_patient_risk(data_dir=data_dir, label_name=label_name, limit=limit)
    print(json.dumps(metrics, indent=2))


def run_nhamcs_summary(data_path: str) -> None:
    summary = summarize_nhamcs_dataset(data_path)
    print(json.dumps(summary, indent=2))


def run_nhamcs_evaluation(data_path: str, task: str, limit: int | None, threshold: float) -> None:
    metrics = evaluate_nhamcs_ed(data_path=data_path, task=task, limit=limit, threshold=threshold)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="Run the built-in agentic vs rule-based comparison.")
    parser.add_argument("--compact", action="store_true", help="Print only summary metrics for the synthetic comparison.")
    parser.add_argument("--save-results", action="store_true", help="Write reproducible evaluation output files.")
    parser.add_argument("--output-dir", default="evaluation_outputs", help="Directory for saved evaluation files.")
    parser.add_argument("--mimic-data", help="Path to a MIMIC-IV-ED folder containing edstays and triage CSV files.")
    parser.add_argument("--mimic-label", default="admitted", choices=["admitted", "high_acuity", "prolonged_los", "critical_proxy"])
    parser.add_argument("--mimic-limit", type=int, default=5000, help="Maximum MIMIC-IV-ED rows to evaluate.")
    parser.add_argument("--nhamcs-data", help="Path to NHAMCS ED CSV or ZIP file.")
    parser.add_argument("--nhamcs-summary", action="store_true", help="Print NHAMCS dataset summary instead of metrics.")
    parser.add_argument(
        "--nhamcs-task",
        default="high_acuity",
        choices=["high_acuity", "critical_vitals", "prolonged_wait", "very_prolonged_wait", "revisit_72h"],
    )
    parser.add_argument("--nhamcs-limit", type=int, default=10000, help="Maximum NHAMCS rows to evaluate.")
    parser.add_argument("--nhamcs-threshold", type=float, default=0.55, help="Decision threshold for NHAMCS metrics.")
    args = parser.parse_args()

    if args.mimic_data:
        run_mimic_evaluation(args.mimic_data, args.mimic_label, args.mimic_limit)
        return

    if args.nhamcs_data:
        if args.nhamcs_summary:
            run_nhamcs_summary(args.nhamcs_data)
            return
        run_nhamcs_evaluation(args.nhamcs_data, args.nhamcs_task, args.nhamcs_limit, args.nhamcs_threshold)
        return

    if args.save_results:
        save_evaluation_outputs(args.output_dir)
        return

    if args.evaluate:
        if args.compact:
            run_compact_evaluation()
            return
        run_evaluation()
        return

    run_demo()


if __name__ == "__main__":
    main()
