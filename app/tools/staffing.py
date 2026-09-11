from __future__ import annotations

from app.enhanced_models import DeterministicStaffingAssessment
from app.models import EDRequest, StaffingToolResult


_MODEL = DeterministicStaffingAssessment()


def run_staffing_availability(ed_input: EDRequest) -> StaffingToolResult:
    return _MODEL.predict(ed_input)
