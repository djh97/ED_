from __future__ import annotations

import csv
import gzip
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from app.models import EDRequest
from app.tools.patient_risk import run_patient_risk


@dataclass(frozen=True)
class MimicEDCase:
    stay_id: str
    subject_id: str
    request: EDRequest
    labels: dict[str, bool | float | str]


def load_mimic_iv_ed_cases(data_dir: str | Path, limit: int | None = None) -> list[MimicEDCase]:
    """Loads MIMIC-IV-ED edstays and triage tables into EDRequest cases.

    Expected files:
    - edstays.csv.gz or edstays.csv
    - triage.csv.gz or triage.csv

    The labels come from the real dataset where available. Operational context
    is derived from timestamps and should be described as proxy context.
    """

    root = Path(data_dir)
    triage_by_stay = {row["stay_id"]: row for row in _read_table(root, "triage")}
    ed_rows = list(_read_table(root, "edstays"))
    if limit is not None:
        ed_rows = ed_rows[:limit]

    hour_counts = Counter(_arrival_hour(row.get("intime", "")) for row in ed_rows)
    cases: list[MimicEDCase] = []
    for row in ed_rows:
        stay_id = row.get("stay_id", "")
        triage = triage_by_stay.get(stay_id, {})
        request = _build_request_from_mimic(row, triage, hour_counts)
        labels = _labels_from_mimic(row, triage)
        cases.append(MimicEDCase(stay_id=stay_id, subject_id=row.get("subject_id", ""), request=request, labels=labels))
    return cases


def evaluate_mimic_patient_risk(
    data_dir: str | Path,
    label_name: str = "admitted",
    limit: int | None = 5000,
    threshold: float = 0.55,
) -> dict[str, float | int | str]:
    """Evaluates the patient-risk tool against a real MIMIC-IV-ED label."""

    cases = load_mimic_iv_ed_cases(data_dir, limit=limit)
    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []

    for case in cases:
        label = bool(case.labels[label_name])
        risk = run_patient_risk(case.request)
        score = risk.flagged_patients[0].risk_score if risk.flagged_patients else 0.0
        y_true.append(1 if label else 0)
        y_score.append(score)
        y_pred.append(1 if score >= threshold else 0)

    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)

    return {
        "dataset": "MIMIC-IV-ED",
        "task": f"patient_risk_to_{label_name}",
        "cases": len(cases),
        "threshold": threshold,
        "accuracy": round((tp + tn) / max(len(cases), 1), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3),
        "auroc": round(_auroc(y_true, y_score), 3),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def _read_table(root: Path, name: str) -> Iterable[dict[str, str]]:
    path = _resolve_table(root, name)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _resolve_table(root: Path, name: str) -> Path:
    for candidate in (root / f"{name}.csv.gz", root / f"{name}.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {name}.csv.gz or {name}.csv in {root}")


def _build_request_from_mimic(row: dict[str, str], triage: dict[str, str], hour_counts: Counter[str]) -> EDRequest:
    hour_key = _arrival_hour(row.get("intime", ""))
    arrivals = max(hour_counts.get(hour_key, 1), 1)
    los_minutes = _length_of_stay_minutes(row.get("intime", ""), row.get("outtime", ""))
    admitted = _is_admitted(row.get("disposition", ""))
    acuity = _int_value(triage.get("acuity"), default=3, lower=1, upper=5)

    return EDRequest(
        timestamp=row.get("intime") or "2100-01-01T00:00:00",
        current_queue_length=min(40, max(3, arrivals * 2)),
        arrivals_last_hour=min(35, arrivals),
        average_wait_minutes=min(180, max(15, int(los_minutes / 6) if los_minutes else 35)),
        boarding_patients=1 if admitted and los_minutes >= 360 else 0,
        patients=[
            {
                "patient_id": f"MIMIC-{row.get('stay_id', 'unknown')}",
                "age": 60,
                "sex": _sex_from_gender(row.get("gender", "")),
                "triage_level": acuity,
                "chief_complaint": triage.get("chiefcomplaint") or "missing chief complaint",
                "triage_notes": f"arrival={row.get('arrival_transport', '')}; disposition={row.get('disposition', '')}",
                "heart_rate": _int_value(triage.get("heartrate"), default=88, lower=20, upper=250),
                "systolic_bp": _int_value(triage.get("sbp"), default=120, lower=40, upper=300),
                "respiratory_rate": _int_value(triage.get("resprate"), default=18, lower=5, upper=80),
                "oxygen_saturation": _int_value(triage.get("o2sat"), default=97, lower=50, upper=100),
                "temperature_c": _temperature_c(triage.get("temperature")),
                "has_abnormal_labs": False,
                "suspected_sepsis": "sepsis" in (triage.get("chiefcomplaint") or "").lower(),
                "pain_score": _pain_score(triage.get("pain")),
                "waiting_minutes": 0,
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
            "occupied_beds": min(24, max(10, arrivals + 12)),
            "discharge_ready_beds": 2,
            "high_acuity_beds_available": 1 if acuity <= 2 else 2,
        },
    )


def _labels_from_mimic(row: dict[str, str], triage: dict[str, str]) -> dict[str, bool | float | str]:
    los_minutes = _length_of_stay_minutes(row.get("intime", ""), row.get("outtime", ""))
    acuity = _int_value(triage.get("acuity"), default=3, lower=1, upper=5)
    admitted = _is_admitted(row.get("disposition", ""))
    return {
        "admitted": admitted,
        "high_acuity": acuity <= 2,
        "prolonged_los": los_minutes >= 360,
        "critical_proxy": acuity <= 2 or admitted and los_minutes >= 360,
        "ed_los_minutes": float(los_minutes),
        "disposition": row.get("disposition", ""),
    }


def _arrival_hour(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:00")
    except ValueError:
        return "unknown"


def _length_of_stay_minutes(intime: str, outtime: str) -> int:
    try:
        start = datetime.fromisoformat(intime)
        end = datetime.fromisoformat(outtime)
        return max(0, int((end - start).total_seconds() / 60))
    except ValueError:
        return 0


def _is_admitted(disposition: str) -> bool:
    disposition_upper = disposition.upper()
    return "ADMIT" in disposition_upper or "TRANSFER" in disposition_upper


def _sex_from_gender(value: str) -> str:
    value = value.upper()
    if value == "F":
        return "female"
    if value == "M":
        return "male"
    return "unknown"


def _int_value(value: str | None, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
    return max(lower, min(parsed, upper))


def _temperature_c(value: str | None) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return 37.0
    if parsed > 45.0:
        parsed = (parsed - 32.0) * 5.0 / 9.0
    return max(30.0, min(round(parsed, 1), 45.0))


def _pain_score(value: str | None) -> int:
    if not value:
        return 0
    match = "".join(character for character in str(value) if character.isdigit())
    if not match:
        return 0
    return max(0, min(int(match[:2]), 10))


def _auroc(y_true: list[int], y_score: list[float]) -> float:
    positives = [(score, truth) for truth, score in zip(y_true, y_score) if truth == 1]
    negatives = [(score, truth) for truth, score in zip(y_true, y_score) if truth == 0]
    if not positives or not negatives:
        return 0.0

    wins = 0.0
    total = len(positives) * len(negatives)
    for positive_score, _ in positives:
        for negative_score, _ in negatives:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / total
