from __future__ import annotations

from app.enhanced_models import DeterministicPatientRiskAssessment
from app.models import EDRequest, PatientRiskToolResult


_MODEL = DeterministicPatientRiskAssessment()


def run_patient_risk(ed_input: EDRequest) -> PatientRiskToolResult:
    return _MODEL.predict(ed_input)
