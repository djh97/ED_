from __future__ import annotations

import csv
import io
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.models import EDRequest
from app.tools.flow_prediction import run_flow_prediction
from app.tools.patient_risk import run_patient_risk


@dataclass(frozen=True)
class NhamcsEDCase:
    visit_id: str
    request: EDRequest
    labels: dict[str, bool | float | str]


def load_nhamcs_ed_cases(data_path: str | Path, limit: int | None = None) -> list[NhamcsEDCase]:
    """Loads the public NHAMCS ED 2018-2022 Kaggle/CDC CSV into EDRequest cases.

    Supported input:
    - a CSV file with columns from the Kaggle NHAMCS 2018-2022 package
    - a ZIP file containing nhamcs_data_2018_22.csv

    NHAMCS is visit-level survey data, not a live operational event stream.
    Staffing and bed inputs are therefore conservative proxy context.
    """

    rows = list(_read_nhamcs_rows(data_path))
    if limit is not None:
        rows = rows[:limit]

    hour_counts = Counter(_arrival_bucket(row) for row in rows)
    cases: list[NhamcsEDCase] = []
    for index, row in enumerate(rows, start=1):
        visit_id = f"NHAMCS-{row.get('year', 'unknown')}-{index:06d}"
        request = _build_request(row, visit_id, hour_counts)
        labels = _labels(row)
        cases.append(NhamcsEDCase(visit_id=visit_id, request=request, labels=labels))
    return cases


def evaluate_nhamcs_ed(
    data_path: str | Path,
    task: str = "high_acuity",
    limit: int | None = 10000,
    threshold: float = 0.55,
) -> dict[str, float | int | str]:
    """Evaluates system tools against public NHAMCS ED labels."""

    cases = load_nhamcs_ed_cases(data_path, limit=limit)
    if task not in {"high_acuity", "critical_vitals", "prolonged_wait", "very_prolonged_wait", "revisit_72h"}:
        raise ValueError(f"Unsupported NHAMCS task: {task}")

    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []

    for case in cases:
        label = bool(case.labels[task])
        score = _score_case(case.request, task)
        y_true.append(1 if label else 0)
        y_score.append(score)
        y_pred.append(1 if score >= threshold else 0)

    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    best = _best_f1_threshold(y_true, y_score)

    return {
        "dataset": "NHAMCS ED 2018-2022",
        "source": "CDC/NCHS NHAMCS public-use data packaged on Kaggle",
        "task": task,
        "cases": len(cases),
        "threshold": threshold,
        "positive_rate": round(sum(y_true) / max(len(y_true), 1), 3),
        "accuracy": round((tp + tn) / max(len(cases), 1), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3),
        "auroc": round(_auroc(y_true, y_score), 3),
        "best_f1_threshold": best["threshold"],
        "best_f1": best["f1"],
        "best_f1_precision": best["precision"],
        "best_f1_recall": best["recall"],
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def summarize_nhamcs_dataset(data_path: str | Path) -> dict[str, object]:
    rows = list(_read_nhamcs_rows(data_path))
    year_counts = Counter(row.get("year", "") for row in rows)
    acuity_counts = Counter(row.get("target_triage_acuity", "") for row in rows)
    seen_72h_counts = Counter(row.get("seen_last_72h", "") for row in rows)
    ems_counts = Counter(row.get("ems_arrival", "") for row in rows)
    key_fields = [
        "target_triage_acuity",
        "wait_time_minutes",
        "age",
        "heart_rate",
        "resp_rate",
        "sys_bp",
        "spo2",
        "pain_score",
        "chief_complaint_text",
    ]
    missing = {
        field: sum(1 for row in rows if not str(row.get(field, "")).strip())
        for field in key_fields
    }
    return {
        "dataset": "NHAMCS ED 2018-2022",
        "rows": len(rows),
        "columns": len(rows[0]) if rows else 0,
        "years": dict(sorted(year_counts.items())),
        "target_triage_acuity": dict(acuity_counts.most_common()),
        "seen_last_72h": dict(seen_72h_counts.most_common()),
        "ems_arrival": dict(ems_counts.most_common()),
        "missing": missing,
        "supported_tasks": [
            "high_acuity",
            "critical_vitals",
            "prolonged_wait",
            "very_prolonged_wait",
            "revisit_72h",
        ],
    }


def _read_nhamcs_rows(data_path: str | Path) -> Iterable[dict[str, str]]:
    path = Path(data_path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            entry_name = _find_csv_entry(archive)
            with archive.open(entry_name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                yield from csv.DictReader(text)
        return

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _find_csv_entry(archive: zipfile.ZipFile) -> str:
    csv_entries = [entry.filename for entry in archive.infolist() if entry.filename.lower().endswith(".csv")]
    if not csv_entries:
        raise FileNotFoundError("NHAMCS ZIP does not contain a CSV file.")
    preferred = [entry for entry in csv_entries if "nhamcs" in entry.lower()]
    return preferred[0] if preferred else csv_entries[0]


def _build_request(row: dict[str, str], visit_id: str, hour_counts: Counter[str]) -> EDRequest:
    acuity = _int_value(row.get("target_triage_acuity"), default=3, lower=1, upper=5)
    wait = _int_value(row.get("wait_time_minutes"), default=35, lower=0, upper=720)
    arrivals = min(35, max(1, hour_counts[_arrival_bucket(row)]))
    queue_length = min(40, max(3, arrivals + wait // 12))
    occupied_beds = min(24, max(10, 12 + arrivals // 2 + wait // 30))

    return EDRequest(
        timestamp=_timestamp(row),
        current_queue_length=queue_length,
        arrivals_last_hour=arrivals,
        average_wait_minutes=min(180, wait),
        boarding_patients=1 if wait >= 180 else 0,
        patients=[
            {
                "patient_id": visit_id,
                "age": _int_value(row.get("age"), default=55, lower=0, upper=120),
                "sex": _sex(row.get("sex", "")),
                "triage_level": acuity,
                "chief_complaint": row.get("chief_complaint_text", "")[:240],
                "triage_notes": f"EMS={row.get('ems_arrival', '')}; revisit72h={row.get('seen_last_72h', '')}; year={row.get('year', '')}",
                "heart_rate": _int_value(row.get("heart_rate"), default=88, lower=20, upper=250),
                "systolic_bp": _int_value(row.get("sys_bp"), default=120, lower=40, upper=300),
                "respiratory_rate": _int_value(row.get("resp_rate"), default=18, lower=5, upper=80),
                "oxygen_saturation": _int_value(row.get("spo2"), default=97, lower=50, upper=100),
                "temperature_c": _temperature_c(row.get("temp")),
                "has_abnormal_labs": False,
                "suspected_sepsis": _suspected_sepsis(row.get("chief_complaint_text", "")),
                "pain_score": _int_value(row.get("pain_score"), default=0, lower=0, upper=10),
                "waiting_minutes": wait,
            }
        ],
        staffing={
            "available_nurses": 6,
            "available_physicians": 3,
            "nurse_capacity_per_hour": 4,
            "physician_capacity_per_hour": 6,
            "staff_absence_flag": False,
        },
        beds={
            "total_beds": 24,
            "occupied_beds": occupied_beds,
            "discharge_ready_beds": 2,
            "high_acuity_beds_available": 1 if acuity <= 2 else 2,
        },
    )


def _labels(row: dict[str, str]) -> dict[str, bool | float | str]:
    acuity = _int_value(row.get("target_triage_acuity"), default=3, lower=1, upper=5)
    wait = _int_value(row.get("wait_time_minutes"), default=0, lower=0, upper=720)
    sbp = _int_value(row.get("sys_bp"), default=120, lower=40, upper=300)
    spo2 = _int_value(row.get("spo2"), default=97, lower=50, upper=100)
    rr = _int_value(row.get("resp_rate"), default=18, lower=5, upper=80)
    hr = _int_value(row.get("heart_rate"), default=88, lower=20, upper=250)
    return {
        "high_acuity": acuity <= 2,
        "critical_vitals": sbp < 90 or spo2 < 92 or rr > 24 or hr > 120,
        "prolonged_wait": wait >= 60,
        "very_prolonged_wait": wait >= 120,
        "revisit_72h": str(row.get("seen_last_72h", "")).strip() == "1",
        "wait_time_minutes": float(wait),
        "triage_acuity": str(acuity),
    }


def _score_case(request: EDRequest, task: str) -> float:
    if task in {"high_acuity", "critical_vitals", "revisit_72h"}:
        risk = run_patient_risk(request)
        return risk.flagged_patients[0].risk_score if risk.flagged_patients else 0.0

    flow = run_flow_prediction(request)
    return flow.congestion_score


def _arrival_bucket(row: dict[str, str]) -> str:
    arrival = str(row.get("arrival_time", "")).zfill(4)
    hour = arrival[:2] if arrival[:2].isdigit() else "unknown"
    return f"{row.get('year', 'unknown')}-{row.get('visit_month', 'unknown')}-{row.get('day_of_week', 'unknown')}-{hour}"


def _timestamp(row: dict[str, str]) -> str:
    year = _int_value(row.get("year"), default=2022, lower=2018, upper=2022)
    month = _int_value(row.get("visit_month"), default=1, lower=1, upper=12)
    arrival = str(row.get("arrival_time", "")).zfill(4)
    hour = _int_value(arrival[:2], default=12, lower=0, upper=23)
    minute = _int_value(arrival[2:], default=0, lower=0, upper=59)
    return f"{year:04d}-{month:02d}-01T{hour:02d}:{minute:02d}:00Z"


def _sex(value: str) -> str:
    value = value.strip().lower()
    if value == "female":
        return "female"
    if value == "male":
        return "male"
    return "unknown"


def _temperature_c(value: str | None) -> float:
    parsed = _float_value(value, default=98.6)
    if parsed > 45.0:
        parsed = (parsed - 32.0) * 5.0 / 9.0
    return max(30.0, min(round(parsed, 1), 45.0))


def _suspected_sepsis(text: str) -> bool:
    lowered = text.lower()
    return "sepsis" in lowered or ("fever" in lowered and ("shortness of breath" in lowered or "weakness" in lowered))


def _int_value(value: str | None, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
    return max(lower, min(parsed, upper))


def _float_value(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _auroc(y_true: list[int], y_score: list[float]) -> float:
    positives = [score for truth, score in zip(y_true, y_score) if truth == 1]
    negatives = [score for truth, score in zip(y_true, y_score) if truth == 0]
    if not positives or not negatives:
        return 0.0

    wins = 0.0
    total = len(positives) * len(negatives)
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / total


def _best_f1_threshold(y_true: list[int], y_score: list[float]) -> dict[str, float]:
    if not y_true:
        return {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    thresholds = sorted({round(score, 3) for score in y_score})
    if not thresholds:
        thresholds = [0.0]

    best = {"threshold": thresholds[0], "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in thresholds:
        predictions = [1 if score >= threshold else 0 for score in y_score]
        tp = sum(1 for truth, pred in zip(y_true, predictions) if truth == 1 and pred == 1)
        fp = sum(1 for truth, pred in zip(y_true, predictions) if truth == 0 and pred == 1)
        fn = sum(1 for truth, pred in zip(y_true, predictions) if truth == 1 and pred == 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best["f1"]:
            best = {
                "threshold": round(threshold, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            }
    return best
