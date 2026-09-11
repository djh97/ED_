from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents import InputUnderstandingAgent
from app.models import EDRequest


def load_extraction_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "case_id" not in item or "text" not in item or "expected" not in item:
                raise ValueError(f"Line {line_number} must contain case_id, text, and expected.")
            item["expected"] = EDRequest(**item["expected"])
            cases.append(item)
    return cases


def evaluate_free_text_extraction(
    cases: list[dict[str, Any]],
    agent: InputUnderstandingAgent | None = None,
    numeric_tolerance: float = 0.0,
) -> dict[str, Any]:
    agent = agent or InputUnderstandingAgent(use_llm=True)
    rows = []
    total_fields = 0
    correct_fields = 0
    failures = 0
    per_field: dict[str, dict[str, int]] = {}
    for case in cases:
        expected: EDRequest = case["expected"]
        try:
            result = agent.normalize(case["text"])
            comparison = compare_requests(expected, result.ed_request, numeric_tolerance)
            status = "completed"
            error_type = None
            error_message = None
        except Exception as exc:
            comparison = []
            status = "failed"
            error_type = type(exc).__name__
            error_message = str(exc)
            failures += 1
        for item in comparison:
            total_fields += 1
            correct_fields += int(item["correct"])
            field = per_field.setdefault(item["field"], {"correct": 0, "total": 0})
            field["correct"] += int(item["correct"])
            field["total"] += 1
        rows.append({
            "case_id": case["case_id"],
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
            "field_results": comparison,
        })
    return {
        "cases": len(cases),
        "completed_cases": len(cases) - failures,
        "failed_cases": failures,
        "failure_rate": round(failures / max(len(cases), 1), 6),
        "field_accuracy": round(correct_fields / max(total_fields, 1), 6),
        "per_field": {
            name: {
                **counts,
                "accuracy": round(counts["correct"] / max(counts["total"], 1), 6),
            }
            for name, counts in sorted(per_field.items())
        },
        "case_results": rows,
    }


def compare_requests(expected: EDRequest, predicted: EDRequest, numeric_tolerance: float = 0.0) -> list[dict[str, Any]]:
    expected_flat = _flatten(expected.model_dump())
    predicted_flat = _flatten(predicted.model_dump())
    results = []
    for field, expected_value in sorted(expected_flat.items()):
        predicted_value = predicted_flat.get(field)
        if isinstance(expected_value, (int, float)) and isinstance(predicted_value, (int, float)):
            correct = abs(float(expected_value) - float(predicted_value)) <= numeric_tolerance
        else:
            correct = predicted_value == expected_value
        results.append({
            "field": field,
            "expected": expected_value,
            "predicted": predicted_value,
            "correct": correct,
        })
    return results


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            output.update(_flatten(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(_flatten(child, f"{prefix}[{index}]"))
    else:
        output[prefix] = value
    return output
