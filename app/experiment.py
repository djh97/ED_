from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable

from app.agentic_system import DecisionRunner, EvaluationScenario, benchmark_manifest, evaluate_runner
from app.config import settings


@dataclass(frozen=True)
class ExperimentConfig:
    runs: int = 5
    scenario_seed: int = 2026
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    failure_policy: str = "score_as_failure"
    concurrency: int = 1


SUMMARY_METRICS = (
    "precision",
    "recall",
    "avg_action_quality",
    "avg_weighted_error_cost",
    "avg_structural_explanation_completeness",
    "avg_response_time_ms",
    "escalation_recall",
    "target_precision",
    "target_recall",
    "target_exact_set_accuracy",
    "failure_rate",
    "avg_tools_invoked",
    "tool_selection_fallback_rate",
)


class ExperimentCheckpointStore:
    """Persists each scenario atomically and validates resumptions."""

    INDEX_NAME = "checkpoint_index.json"

    def __init__(
        self,
        root: str | Path,
        metadata: dict[str, Any],
        resume: bool,
        allow_writer_upgrade: bool = False,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / self.INDEX_NAME
        self.metadata = json.loads(json.dumps(metadata, sort_keys=True))
        self._lock = threading.Lock()
        self._status = "running"
        self._created_at = _utc_now()
        self._completed: set[str] = set()
        self._migrations: list[dict[str, Any]] = []

        if resume:
            if not self.index_path.exists():
                raise FileNotFoundError(f"Cannot resume: checkpoint index not found at {self.index_path}")
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
            if index.get("metadata") != self.metadata:
                previous_metadata = index.get("metadata", {})
                if not allow_writer_upgrade or not _only_checkpoint_writer_changed(previous_metadata, self.metadata):
                    raise ValueError("Checkpoint metadata does not match this experiment configuration.")
                self._migrations = list(index.get("migrations", []))
                self._migrations.append(
                    {
                        "migrated_at": _utc_now(),
                        "reason": "Checkpoint-writer reliability fix; decision and scenario configuration unchanged.",
                        "previous_code_commit": previous_metadata.get("code_commit"),
                        "new_code_commit": self.metadata.get("code_commit"),
                        "previous_source_sha256": previous_metadata.get("source_sha256"),
                        "new_source_sha256": self.metadata.get("source_sha256"),
                    }
                )
            else:
                self._migrations = list(index.get("migrations", []))
            self._created_at = index["created_at"]
            self._status = index.get("status", "running")
            self._completed = set(index.get("completed", []))
            self._reconcile_scenario_files()
        else:
            existing_rows = list(self.root.rglob("scenario_*.json"))
            if self.index_path.exists() or existing_rows:
                raise FileExistsError(
                    f"Checkpoint data already exists at {self.root}. Use --resume or select a new output directory."
                )
            self._write_index_locked()

    @property
    def completed_count(self) -> int:
        with self._lock:
            return len(self._completed)

    def load_run_rows(
        self,
        configuration: str,
        run_index: int,
        scenarios: list[EvaluationScenario],
    ) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for scenario_index, scenario in enumerate(scenarios):
            path = self._scenario_path(configuration, run_index, scenario_index)
            if not path.exists():
                continue
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            expected = (configuration, run_index, scenario_index, scenario.name)
            actual = (
                wrapper.get("configuration"),
                wrapper.get("run_index"),
                wrapper.get("scenario_index"),
                wrapper.get("scenario_name"),
            )
            if actual != expected:
                raise ValueError(f"Checkpoint identity mismatch in {path}: expected {expected}, found {actual}")
            rows[scenario.name] = wrapper["row"]
        return rows

    def save_row(
        self,
        configuration: str,
        run_index: int,
        scenario_index: int,
        scenario_name: str,
        row: dict[str, Any],
    ) -> int:
        wrapper = {
            "configuration": configuration,
            "run_index": run_index,
            "scenario_index": scenario_index,
            "scenario_name": scenario_name,
            "saved_at": _utc_now(),
            "row": row,
        }
        path = self._scenario_path(configuration, run_index, scenario_index)
        _atomic_write_json(path, wrapper)
        with self._lock:
            self._completed.add(self._key(configuration, run_index, scenario_index))
            self._status = "running"
            if len(self._completed) % 25 == 0:
                self._write_index_best_effort_locked()
            return len(self._completed)

    def mark_scenario_evaluation_complete(self) -> None:
        with self._lock:
            self._status = "scenario_evaluation_complete"
            self._write_index_best_effort_locked()

    def mark_interrupted(self) -> None:
        with self._lock:
            self._status = "interrupted"
            self._write_index_best_effort_locked()

    def _reconcile_scenario_files(self) -> None:
        reconciled: set[str] = set()
        for path in self.root.rglob("scenario_*.json"):
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            reconciled.add(
                self._key(
                    str(wrapper["configuration"]),
                    int(wrapper["run_index"]),
                    int(wrapper["scenario_index"]),
                )
            )
        with self._lock:
            self._completed = reconciled
            self._write_index_best_effort_locked()

    def _write_index_best_effort_locked(self) -> None:
        try:
            self._write_index_locked()
        except OSError as exc:
            print(
                f"[experiment] checkpoint index update deferred; scenario files remain durable: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _write_index_locked(self) -> None:
        _atomic_write_json(
            self.index_path,
            {
                "version": 1,
                "status": self._status,
                "created_at": self._created_at,
                "updated_at": _utc_now(),
                "metadata": self.metadata,
                "migrations": self._migrations,
                "completed_count": len(self._completed),
                "completed": sorted(self._completed),
            },
        )

    def _scenario_path(self, configuration: str, run_index: int, scenario_index: int) -> Path:
        return self.root / configuration / f"run_{run_index:03d}" / f"scenario_{scenario_index:04d}.json"

    @staticmethod
    def _key(configuration: str, run_index: int, scenario_index: int) -> str:
        return f"{configuration}|{run_index}|{scenario_index}"


def run_matched_experiment(
    scenarios: list[EvaluationScenario],
    runner_factories: dict[str, Callable[[int], DecisionRunner]],
    config: ExperimentConfig,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    allow_checkpoint_writer_upgrade: bool = False,
) -> dict[str, Any]:
    """Runs matched configurations with optional atomic scenario checkpoints."""

    if config.runs < 1:
        raise ValueError("runs must be at least 1")
    if config.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")

    started_at = _utc_now()
    if config.concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    checkpoint_store = None
    if checkpoint_dir is not None:
        checkpoint_store = ExperimentCheckpointStore(
            checkpoint_dir,
            _checkpoint_metadata(scenarios, runner_factories, config),
            resume=resume,
            allow_writer_upgrade=allow_checkpoint_writer_upgrade,
        )
        if resume:
            print(
                f"[experiment] resume loaded {checkpoint_store.completed_count}/"
                f"{config.runs * len(runner_factories) * len(scenarios)} checkpoint rows",
                file=sys.stderr,
                flush=True,
            )

    results_by_index: dict[str, list[dict[str, Any] | None]] = {
        name: [None] * config.runs for name in runner_factories
    }
    stop_event = threading.Event()

    def evaluate_job(name: str, factory: Callable[[int], DecisionRunner], run_index: int) -> tuple[str, int, dict[str, Any]]:
        cached_rows = (
            checkpoint_store.load_run_rows(name, run_index, scenarios)
            if checkpoint_store is not None
            else {}
        )

        def checkpoint(scenario_index: int, row: dict[str, Any]) -> None:
            if checkpoint_store is None:
                return
            completed = checkpoint_store.save_row(
                name,
                run_index,
                scenario_index,
                scenarios[scenario_index].name,
                row,
            )
            total = config.runs * len(runner_factories) * len(scenarios)
            if completed % 25 == 0 or completed == total:
                print(f"[experiment] checkpointed {completed}/{total} scenario rows", file=sys.stderr, flush=True)

        result = evaluate_runner(
            name,
            factory(run_index),
            scenarios=scenarios,
            precomputed_rows=cached_rows,
            on_scenario_complete=checkpoint,
            should_stop=stop_event.is_set,
        )
        for row in result["scenario_results"]:
            row["run_index"] = run_index
            row["configuration"] = name
        return name, run_index, result

    jobs = [
        (name, factory, run_index)
        for run_index in range(config.runs)
        for name, factory in runner_factories.items()
    ]
    executor = ThreadPoolExecutor(max_workers=config.concurrency)
    futures = [executor.submit(evaluate_job, *job) for job in jobs]
    try:
        for future in as_completed(futures):
            name, run_index, result = future.result()
            results_by_index[name][run_index] = result
            print(
                f"[experiment] completed configuration={name} run={run_index + 1}/{config.runs}",
                file=sys.stderr,
                flush=True,
            )
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        if checkpoint_store is not None:
            checkpoint_store.mark_interrupted()
        raise
    else:
        executor.shutdown(wait=True)

    results: dict[str, list[dict[str, Any]]] = {
        name: [result for result in indexed_results if result is not None]
        for name, indexed_results in results_by_index.items()
    }
    if checkpoint_store is not None:
        checkpoint_store.mark_scenario_evaluation_complete()

    bootstrap = {
        name: bootstrap_scenario_clusters(runs, config.bootstrap_samples, config.bootstrap_seed)
        for name, runs in results.items()
    }
    paired = {}
    if "full_dynamic" in results and "all_tools" in results:
        paired["all_tools_minus_full_dynamic"] = paired_bootstrap_scenario_clusters(
            results["full_dynamic"],
            results["all_tools"],
            config.bootstrap_samples,
            config.bootstrap_seed,
        )

    return {
        "manifest": build_manifest(config, started_at=started_at, completed_at=_utc_now()),
        "checkpoint": {
            "enabled": checkpoint_store is not None,
            "directory": str(checkpoint_store.root) if checkpoint_store is not None else None,
            "resumed": resume,
            "checkpoint_writer_upgrade": allow_checkpoint_writer_upgrade,
            "checkpointed_scenario_rows": checkpoint_store.completed_count if checkpoint_store is not None else 0,
        },
        "configurations": list(runner_factories),
        "scenario_count": len(scenarios),
        "benchmark": benchmark_manifest(len(scenarios), config.scenario_seed),
        "runs": results,
        "bootstrap_confidence_intervals": bootstrap,
        "paired_bootstrap_differences": paired,
        "weighted_error_sensitivity": weighted_error_sensitivity(results),
        "empirical_safety_performance": {
            name: summarize_empirical_safety(runs)
            for name, runs in results.items()
        },
    }


def summarize_empirical_safety(runs: list[dict[str, Any]]) -> dict[str, int | float]:
    rows = [row for run in runs for row in run["scenario_results"]]
    evaluated_plans = 0
    initial_validation_failures = 0
    corrected_successes = 0
    unrecovered_validation_failures = 0
    execution_failures = 0
    for row in rows:
        log = row.get("safety_validation_log") or []
        if log:
            evaluated_plans += 1
            if not log[0]["passed"]:
                initial_validation_failures += 1
                if log[-1]["passed"]:
                    corrected_successes += 1
                else:
                    unrecovered_validation_failures += 1
        if row["status"] == "failed":
            execution_failures += 1
    return {
        "scenario_runs": len(rows),
        "plans_with_validator_logs": evaluated_plans,
        "initial_validation_failures": initial_validation_failures,
        "corrected_successes": corrected_successes,
        "unrecovered_validation_failures": unrecovered_validation_failures,
        "execution_failures": execution_failures,
        "initial_omission_rate": round(initial_validation_failures / max(evaluated_plans, 1), 6),
        "correction_success_rate": round(corrected_successes / max(initial_validation_failures, 1), 6),
    }


def weighted_error_sensitivity(
    results: dict[str, list[dict[str, Any]]],
    false_negative_weights: tuple[float, ...] = (4.0, 8.0, 12.0),
    false_positive_weights: tuple[float, ...] = (1.0, 2.0, 4.0),
    missed_target_weights: tuple[float, ...] = (8.0, 12.0, 16.0),
) -> list[dict[str, float | str]]:
    """Evaluates conclusions over a prespecified grid of error-cost weights."""

    output: list[dict[str, float | str]] = []
    for configuration, runs in results.items():
        rows = [row for run in runs for row in run["scenario_results"]]
        for fn_weight in false_negative_weights:
            for fp_weight in false_positive_weights:
                for target_weight in missed_target_weights:
                    costs = []
                    for row in rows:
                        metrics = row["metrics"]
                        costs.append(
                            metrics["false_negatives"] * fn_weight
                            + metrics["false_positives"] * fp_weight
                            + metrics["target_false_negatives"] * target_weight
                            + metrics["target_false_positives"] * fp_weight
                        )
                    output.append({
                        "configuration": configuration,
                        "false_negative_weight": fn_weight,
                        "false_positive_weight": fp_weight,
                        "missed_target_weight": target_weight,
                        "mean_weighted_error_cost": round(_mean(costs), 6),
                    })
    return output


def bootstrap_scenario_clusters(
    runs: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Bootstraps scenarios while retaining every repeated run for each sampled scenario."""

    clusters = _scenario_clusters(runs)
    scenario_names = sorted(clusters)
    if not scenario_names:
        return {}
    rng = random.Random(seed)
    estimates = {metric: [] for metric in SUMMARY_METRICS}
    point = _aggregate_rows([row for rows in clusters.values() for row in rows])
    for _ in range(samples):
        sampled_rows: list[dict[str, Any]] = []
        for _ in scenario_names:
            sampled_rows.extend(clusters[rng.choice(scenario_names)])
        aggregate = _aggregate_rows(sampled_rows)
        for metric in SUMMARY_METRICS:
            estimates[metric].append(aggregate[metric])
    return {
        metric: {
            "value": round(point[metric], 6),
            "lower": round(_percentile(values, 0.025), 6),
            "upper": round(_percentile(values, 0.975), 6),
        }
        for metric, values in estimates.items()
    }


def paired_bootstrap_scenario_clusters(
    reference_runs: list[dict[str, Any]],
    comparison_runs: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Returns comparison minus reference using matched run-scenario pairs."""

    reference = _paired_rows(reference_runs)
    comparison = _paired_rows(comparison_runs)
    keys = sorted(set(reference) & set(comparison))
    scenario_names = sorted({scenario for _, scenario in keys})
    if not scenario_names:
        return {}
    by_scenario = {
        scenario: [key for key in keys if key[1] == scenario]
        for scenario in scenario_names
    }
    rng = random.Random(seed)
    estimates = {metric: [] for metric in SUMMARY_METRICS}

    def difference(selected_keys: list[tuple[int, str]]) -> dict[str, float]:
        ref = _aggregate_rows([reference[key] for key in selected_keys])
        comp = _aggregate_rows([comparison[key] for key in selected_keys])
        return {metric: comp[metric] - ref[metric] for metric in SUMMARY_METRICS}

    point = difference(keys)
    for _ in range(samples):
        sampled_keys: list[tuple[int, str]] = []
        for _ in scenario_names:
            sampled_keys.extend(by_scenario[rng.choice(scenario_names)])
        delta = difference(sampled_keys)
        for metric in SUMMARY_METRICS:
            estimates[metric].append(delta[metric])
    return {
        metric: {
            "value": round(point[metric], 6),
            "lower": round(_percentile(values, 0.025), 6),
            "upper": round(_percentile(values, 0.975), 6),
        }
        for metric, values in estimates.items()
    }


def write_experiment_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "full_results": output / "experiment_results.json",
        "manifest": output / "experiment_manifest.json",
        "scenario_results": output / "scenario_results.jsonl",
        "summary": output / "summary_metrics.csv",
        "failures": output / "failure_report.csv",
        "tool_selection": output / "tool_selection_report.csv",
        "bootstrap": output / "bootstrap_intervals.csv",
        "weight_sensitivity": output / "weighted_error_sensitivity.csv",
        "safety_performance": output / "empirical_safety_performance.csv",
    }
    paths["full_results"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    paths["manifest"].write_text(json.dumps(result["manifest"], indent=2), encoding="utf-8")
    rows = _all_rows(result["runs"])
    paths["scenario_results"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    paths["summary"].write_text(_summary_csv(result), encoding="utf-8")
    paths["failures"].write_text(_failure_csv(rows), encoding="utf-8")
    paths["tool_selection"].write_text(_tool_selection_csv(rows), encoding="utf-8")
    paths["bootstrap"].write_text(_bootstrap_csv(result), encoding="utf-8")
    sensitivity_rows = result["weighted_error_sensitivity"]
    sensitivity_fields = [
        "configuration",
        "false_negative_weight",
        "false_positive_weight",
        "missed_target_weight",
        "mean_weighted_error_cost",
    ]
    paths["weight_sensitivity"].write_text(
        _rows_to_csv(sensitivity_fields, sensitivity_rows), encoding="utf-8"
    )
    safety_rows = [
        {"configuration": configuration, **values}
        for configuration, values in result["empirical_safety_performance"].items()
    ]
    safety_fields = ["configuration", *next(iter(result["empirical_safety_performance"].values())).keys()]
    paths["safety_performance"].write_text(
        _rows_to_csv(list(safety_fields), safety_rows), encoding="utf-8"
    )
    return {name: str(path) for name, path in paths.items()}


def build_manifest(config: ExperimentConfig, started_at: str, completed_at: str) -> dict[str, Any]:
    dependencies = {}
    for package in ("fastapi", "uvicorn", "pydantic", "httpx", "openai"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "run_id": f"ed-{started_at.replace(':', '').replace('-', '')}",
        "started_at": started_at,
        "completed_at": completed_at,
        "git_commit": os.getenv("EVALUATED_COMMIT") or _git_commit(),
        "model": {
            "input": settings.openai_input_model,
            "orchestration": settings.openai_orchestration_model,
            "summary": settings.openai_summary_model,
        },
        "prompt_schema_version": "dynamic-tools-v1",
        "experiment": asdict(config),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependencies,
        "command": sys.argv,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = [row["metrics"] for row in rows]
    tp = sum(item["true_positives"] for item in metrics)
    fp = sum(item["false_positives"] for item in metrics)
    fn = sum(item["false_negatives"] for item in metrics)
    target_tp = sum(item["target_true_positives"] for item in metrics)
    target_fp = sum(item["target_false_positives"] for item in metrics)
    target_fn = sum(item["target_false_negatives"] for item in metrics)
    escalation = [item for item in metrics if "escalate_patient" in item["expected_actions"]]
    tool_counts = [
        len(row["tool_selection"]["selected_tools"])
        if row.get("tool_selection")
        else 4
        for row in rows
    ]
    fallback_flags = [
        float(bool((row.get("tool_selection") or {}).get("fallback_used", False)))
        for row in rows
    ]
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "avg_action_quality": _mean(item["action_quality"] for item in metrics),
        "avg_weighted_error_cost": _mean(item["weighted_error_cost"] for item in metrics),
        "avg_structural_explanation_completeness": _mean(
            item["structural_explanation_completeness"] for item in metrics
        ),
        "avg_response_time_ms": _mean(item["response_time_ms"] for item in metrics),
        "escalation_recall": _mean(float(item["escalation_hit"]) for item in escalation),
        "target_precision": target_tp / max(target_tp + target_fp, 1),
        "target_recall": target_tp / max(target_tp + target_fn, 1),
        "target_exact_set_accuracy": _mean(float(item["escalation_target_exact_match"]) for item in escalation),
        "failure_rate": _mean(float(row["status"] == "failed") for row in rows),
        "avg_tools_invoked": _mean(tool_counts),
        "tool_selection_fallback_rate": _mean(fallback_flags),
    }


def _scenario_clusters(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for row in run["scenario_results"]:
            clusters.setdefault(row["scenario"], []).append(row)
    return clusters


def _paired_rows(runs: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (row["run_index"], row["scenario"]): row
        for run in runs
        for row in run["scenario_results"]
    }


def _all_rows(results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for runs in results.values() for run in runs for row in run["scenario_results"]]


def _summary_csv(result: dict[str, Any]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["configuration", "metric", "value", "lower", "upper"])
    writer.writeheader()
    for configuration, metrics in result["bootstrap_confidence_intervals"].items():
        for metric, values in metrics.items():
            writer.writerow({"configuration": configuration, "metric": metric, **values})
    return buffer.getvalue()


def _failure_csv(rows: list[dict[str, Any]]) -> str:
    fields = ["configuration", "run_index", "scenario", "status", "error_type", "error_message"]
    return _rows_to_csv(fields, [row for row in rows if row["status"] == "failed"])


def _tool_selection_csv(rows: list[dict[str, Any]]) -> str:
    fields = ["configuration", "run_index", "scenario", "selected_tools", "skipped_tools", "fallback_used", "error"]
    selected_rows = []
    for row in rows:
        selection = row.get("tool_selection") or {}
        selected_rows.append({
            "configuration": row["configuration"],
            "run_index": row["run_index"],
            "scenario": row["scenario"],
            "selected_tools": ";".join(selection.get("selected_tools", [])),
            "skipped_tools": ";".join(selection.get("skipped_tools", [])),
            "fallback_used": selection.get("fallback_used"),
            "error": selection.get("error"),
        })
    return _rows_to_csv(fields, selected_rows)


def _bootstrap_csv(result: dict[str, Any]) -> str:
    fields = ["comparison", "metric", "value", "lower", "upper"]
    rows = []
    for comparison, metrics in result["paired_bootstrap_differences"].items():
        for metric, values in metrics.items():
            rows.append({"comparison": comparison, "metric": metric, **values})
    return _rows_to_csv(fields, rows)


def _rows_to_csv(fields: list[str], rows: list[dict[str, Any]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _checkpoint_metadata(
    scenarios: list[EvaluationScenario],
    runner_factories: dict[str, Callable[[int], DecisionRunner]],
    config: ExperimentConfig,
) -> dict[str, Any]:
    scenario_records = [
        {
            "name": scenario.name,
            "description": scenario.description,
            "payload": scenario.payload.model_dump(mode="json"),
            "expected_actions": sorted(scenario.expected_actions),
            "urgent_targets": sorted(scenario.urgent_targets),
        }
        for scenario in scenarios
    ]
    scenario_bytes = json.dumps(scenario_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    experiment_signature = asdict(config)
    experiment_signature.pop("concurrency", None)
    return {
        "schema_version": 1,
        "code_commit": os.getenv("EVALUATED_COMMIT") or _git_commit(),
        "source_sha256": _source_sha256(),
        "prompt_schema_version": "dynamic-tools-v1",
        "models": {
            "input": settings.openai_input_model,
            "orchestration": settings.openai_orchestration_model,
            "summary": settings.openai_summary_model,
        },
        "experiment": experiment_signature,
        "configurations": list(runner_factories),
        "scenario_count": len(scenarios),
        "scenario_names": [scenario.name for scenario in scenarios],
        "scenario_sha256": hashlib.sha256(scenario_bytes).hexdigest(),
    }


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    source_files = sorted((root / "app").rglob("*.py")) + sorted((root / "evaluation").rglob("*.py"))
    source_files.extend(path for path in (root / "requirements.txt", root / "requirements.lock") if path.exists())
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _only_checkpoint_writer_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    previous_stable = dict(previous)
    current_stable = dict(current)
    for key in ("code_commit", "source_sha256"):
        previous_stable.pop(key, None)
        current_stable.pop(key, None)
    return previous_stable == current_stable


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
