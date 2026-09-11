from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.models import (
    BedToolResult,
    EDRequest,
    FlowToolResult,
    PatientInput,
    PatientRiskToolResult,
    RiskFlag,
    StaffingToolResult,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def _level(score: float, cuts: tuple[float, float, float], names: tuple[str, str, str, str]) -> str:
    if score >= cuts[2]:
        return names[3]
    if score >= cuts[1]:
        return names[2]
    if score >= cuts[0]:
        return names[1]
    return names[0]


@dataclass(frozen=True)
class TextSignal:
    score: float
    factors: list[str]


class NLPIntakeExtractor:
    """Clinical NLP extractor for chief complaint and triage notes."""

    KEYWORDS: tuple[tuple[str, str, float], ...] = (
        ("chest pain", "chest pain", 0.08),
        ("shortness of breath", "shortness of breath", 0.10),
        ("dyspnea", "shortness of breath", 0.10),
        ("confusion", "altered mental status", 0.10),
        ("altered mental", "altered mental status", 0.10),
        ("stroke", "possible stroke", 0.12),
        ("sepsis", "possible sepsis", 0.12),
        ("fever", "fever/infection concern", 0.05),
        ("trauma", "trauma presentation", 0.07),
        ("bleeding", "active bleeding", 0.10),
        ("syncope", "syncope", 0.08),
    )

    def extract(self, patient: PatientInput) -> TextSignal:
        text = f"{patient.chief_complaint} {patient.triage_notes}".lower()
        factors: list[str] = []
        score = 0.0

        for keyword, factor, weight in self.KEYWORDS:
            if keyword in text and factor not in factors:
                factors.append(factor)
                score += weight

        return TextSignal(score=_clamp(score, 0.0, 0.25), factors=factors)


class DeterministicFlowAssessment:
    """Transparent rule-weighted assessment of ED flow pressure."""

    def predict(self, ed_input: EDRequest) -> FlowToolResult:
        occupancy = ed_input.beds.occupied_beds / max(ed_input.beds.total_beds, 1)
        queue = _clamp(ed_input.current_queue_length / 30.0)
        arrivals = _clamp(ed_input.arrivals_last_hour / 25.0)
        boarding = _clamp(ed_input.boarding_patients / 10.0)
        wait = _clamp(ed_input.average_wait_minutes / 180.0)
        absence = 0.10 if ed_input.staffing.staff_absence_flag else 0.0
        hour_pressure = self._hour_pressure(ed_input.timestamp)

        score = _clamp(
            0.24 * occupancy
            + 0.22 * queue
            + 0.20 * arrivals
            + 0.16 * boarding
            + 0.12 * wait
            + 0.06 * hour_pressure
            + absence
        )

        predicted_wait = int(ed_input.average_wait_minutes * (1.0 + score) + ed_input.current_queue_length * 1.25)
        bottleneck = _level(score, (0.40, 0.65, 0.85), ("low", "moderate", "high", "critical"))

        reasoning = self._top_drivers(
            {
                "bed occupancy": occupancy,
                "queue length": queue,
                "recent arrivals": arrivals,
                "boarding": boarding,
                "current wait": wait,
                "time-of-day pressure": hour_pressure,
            }
        )
        adjustment = self._recommend_adjustment(bottleneck, occupancy, queue, boarding)

        return FlowToolResult(
            congestion_score=round(score, 3),
            predicted_wait_minutes=predicted_wait,
            bottleneck_level=bottleneck,
            recommended_adjustment=adjustment,
            reasoning=reasoning,
            rationale=(
                "Deterministic flow assessment estimated congestion from recent demand, "
                "bed occupancy, boarding, current wait, and temporal pressure."
            ),
        )

    def _hour_pressure(self, timestamp: str) -> float:
        try:
            hour = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).hour
        except ValueError:
            return 0.4
        if 10 <= hour <= 22:
            return 1.0
        if 7 <= hour < 10 or 22 < hour <= 23:
            return 0.6
        return 0.25

    def _top_drivers(self, drivers: dict[str, float]) -> list[str]:
        return [name for name, _ in sorted(drivers.items(), key=lambda item: item[1], reverse=True)[:3]]

    def _recommend_adjustment(self, bottleneck: str, occupancy: float, queue: float, boarding: float) -> str:
        if bottleneck == "critical" and occupancy >= 0.9:
            return "Open discharge-ready capacity and reprioritize high-acuity queue."
        if bottleneck in {"high", "critical"} and boarding >= 0.5:
            return "Escalate boarding review and accelerate inpatient transfer decisions."
        if bottleneck in {"high", "critical"} and queue >= 0.6:
            return "Activate queue reprioritization and fast-track low-acuity patients."
        return "Continue routine flow monitoring."


class DeterministicPatientRiskAssessment:
    """Transparent rule-weighted patient-risk assessment with factor contributions."""

    TRIAGE_WEIGHTS = {1: 0.45, 2: 0.32, 3: 0.16, 4: 0.06, 5: 0.02}

    def __init__(self) -> None:
        self._nlp = NLPIntakeExtractor()

    def predict(self, ed_input: EDRequest) -> PatientRiskToolResult:
        flags = [self._score_patient(patient) for patient in ed_input.patients]
        flags.sort(key=lambda flag: flag.risk_score, reverse=True)
        top = flags[0] if flags else None

        adjustment = None
        if top and top.risk_level in {"high", "critical"}:
            adjustment = f"Escalate {top.patient_id} for clinician review."

        return PatientRiskToolResult(
            highest_risk_patient_id=top.patient_id if top else None,
            flagged_patients=flags,
            recommended_adjustment=adjustment,
            rationale=(
                "Deterministic risk assessment combines triage acuity, vitals, labs, "
                "sepsis flags, waiting time, age, and NLP-derived triage symptoms."
            ),
        )

    def _score_patient(self, patient: PatientInput) -> RiskFlag:
        factors: list[tuple[str, float]] = []
        score = self.TRIAGE_WEIGHTS.get(patient.triage_level, 0.0)
        factors.append((f"triage level {patient.triage_level}", score))

        self._add_factor(factors, patient.systolic_bp < 90, "hypotension", 0.18)
        self._add_factor(factors, patient.oxygen_saturation < 92, "low oxygen saturation", 0.16)
        self._add_factor(factors, patient.respiratory_rate > 24 or patient.respiratory_rate < 10, "abnormal respiratory rate", 0.10)
        self._add_factor(factors, patient.heart_rate > 120 or patient.heart_rate < 45, "abnormal heart rate", 0.10)
        self._add_factor(factors, patient.temperature_c >= 38.5 or patient.temperature_c <= 35.0, "temperature abnormality", 0.07)
        self._add_factor(factors, patient.has_abnormal_labs, "abnormal labs", 0.08)
        self._add_factor(factors, patient.suspected_sepsis, "suspected sepsis", 0.18)
        self._add_factor(factors, patient.age >= 75, "older age", 0.05)
        self._add_factor(factors, patient.waiting_minutes > 120 and patient.triage_level <= 3, "prolonged wait", 0.08)

        text_signal = self._nlp.extract(patient)
        if text_signal.score:
            factors.append(("triage text symptoms", text_signal.score))

        total = _clamp(sum(weight for _, weight in factors))
        risk_level = _level(total, (0.30, 0.55, 0.80), ("low", "moderate", "high", "critical"))
        top_factors = [name for name, _ in sorted(factors, key=lambda item: item[1], reverse=True)[:5] if _ > 0]
        if text_signal.factors:
            top_factors.extend([factor for factor in text_signal.factors if factor not in top_factors])

        return RiskFlag(
            patient_id=patient.patient_id,
            risk_score=round(total, 3),
            risk_level=risk_level,
            escalation_needed=risk_level in {"high", "critical"},
            top_risk_factors=top_factors[:6],
            rationale=", ".join(top_factors[:6]) if top_factors else "no major instability signal detected",
        )

    def _add_factor(self, factors: list[tuple[str, float]], condition: bool, name: str, weight: float) -> None:
        if condition:
            factors.append((name, weight))


class DeterministicStaffingAssessment:
    """Transparent demand-to-capacity staffing assessment."""

    def predict(self, ed_input: EDRequest) -> StaffingToolResult:
        staffing = ed_input.staffing
        base_demand = ed_input.arrivals_last_hour + max(ed_input.current_queue_length * 0.55, 0)
        surge_demand = ed_input.boarding_patients * 1.25 + len(ed_input.patients) * 0.75
        high_acuity_load = sum(1 for patient in ed_input.patients if patient.triage_level <= 2) * 2.0
        total_demand = base_demand + surge_demand + high_acuity_load

        capacity = staffing.available_nurses * staffing.nurse_capacity_per_hour + staffing.available_physicians * staffing.physician_capacity_per_hour
        pressure = total_demand / max(capacity, 1)
        if staffing.staff_absence_flag:
            pressure += 0.15
        pressure = _clamp(pressure)
        level = _level(pressure, (0.55, 0.85, 0.95), ("adequate", "strained", "critical", "critical"))

        reasoning = [
            f"base demand={base_demand:.1f}",
            f"surge demand={surge_demand:.1f}",
            f"high-acuity load={high_acuity_load:.1f}",
            f"capacity={capacity}",
        ]

        if level == "critical":
            adjustment = "Activate surge staffing or reassign staff to triage/resuscitation."
        elif level == "strained":
            adjustment = "Move flexible staff toward triage and fast-track coverage."
        else:
            adjustment = "Current staffing is adequate for estimated demand."

        return StaffingToolResult(
            staffing_pressure_score=round(pressure, 3),
            staffing_level=level,
            estimated_hourly_capacity=capacity,
            recommended_adjustment=adjustment,
            reasoning=reasoning,
            rationale="Deterministic staffing assessment combines base demand with surge and acuity load.",
        )


class DeterministicBedCapacityAssessment:
    """Transparent bed-capacity assessment with explicit feasibility checks."""

    def predict(self, ed_input: EDRequest) -> BedToolResult:
        beds = ed_input.beds
        free_now = max(beds.total_beds - beds.occupied_beds, 0)
        occupancy = beds.occupied_beds / max(beds.total_beds, 1)
        effective_capacity = free_now + beds.discharge_ready_beds
        high_acuity_need = sum(1 for patient in ed_input.patients if patient.triage_level <= 2)
        high_acuity_gap = max(high_acuity_need - beds.high_acuity_beds_available, 0)

        if occupancy >= 0.95 and (effective_capacity <= 1 or high_acuity_gap > 0):
            window = "critical"
        elif occupancy >= 0.85 or effective_capacity <= 3 or high_acuity_gap > 0:
            window = "tight"
        else:
            window = "open"

        reasoning = [
            f"occupancy={occupancy:.0%}",
            f"immediate free beds={free_now}",
            f"effective capacity={effective_capacity}",
            f"high-acuity gap={high_acuity_gap}",
        ]

        if window == "critical":
            adjustment = "Run urgent bed reassignment and reserve monitored capacity for high-risk patients."
        elif window == "tight":
            adjustment = "Review discharge-ready beds and high-acuity placement constraints."
        else:
            adjustment = "No bed reassignment required."

        return BedToolResult(
            occupancy_rate=round(occupancy, 3),
            available_beds_now=free_now,
            action_window=window,
            recommended_adjustment=adjustment,
            reasoning=reasoning,
            rationale="Deterministic bed-capacity assessment evaluates occupancy, discharge-ready capacity, acuity fit, and feasibility.",
        )
